import os
import re
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from .utils import save_image, EMA
from metrics.fid_score import InceptionStatistics, get_precomputed, calc_fd
from tqdm import tqdm
from contextlib import nullcontext
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter  
writer = SummaryWriter()

def get_data_inverse_scaler():
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.

class DummyScheduler:
    @staticmethod
    def step():
        pass

    def load_state_dict(self, state_dict):
        pass

    @staticmethod
    def state_dict():
        return None


class RunningStatistics:
    def __init__(self, **kwargs):
        self.count = 0
        self.stats = []
        for k, v in kwargs.items():
            self.stats.append((k, v or 0))
        self.stats = dict(self.stats)

    def reset(self):
        self.count = 0
        for k in self.stats:
            self.stats[k] = 0

    def update(self, n, **kwargs):
        self.count += n
        for k, v in kwargs.items():
            self.stats[k] = self.stats.get(k, 0) + v

    def extract(self):
        avg_stats = []
        for k, v in self.stats.items():
            avg_stats.append((k, v/self.count))
        return dict(avg_stats)

    def __repr__(self):
        out_str = "Count(s): {}\n"
        out_str += "Statistics:\n"
        for k in self.stats:
            out_str += f"\t{k} = {{{k}}}\n"  # double curly-bracket to escape
        return out_str.format(self.count, **self.stats)


class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            diffusion,
            timesteps,
            epochs,
            trainloader,
            sampler=None,
            scheduler=None,
            use_cfg=False,
            use_ema=False,
            grad_norm=1.0,
            num_accum=1,
            shape=None,
            device=torch.device("cpu"),
            chkpt_intv=5,  # save a checkpoint every [chkpt_intv] epochs
            image_intv=1,  # generate images every [image_intv] epochs
            num_save_images=64,
            ema_decay=0.9999,
            distributed=False,
            rank=0  # process id for distributed training
    ):
        self.model = model
        self.optimizer = optimizer
        self.diffusion = diffusion
        self.timesteps = timesteps
        self.epochs = epochs
        self.start_epoch = 0
        self.trainloader = trainloader
        self.sampler = sampler
        if shape is None:
            shape = next(iter(trainloader))[0].shape[1:]
        self.shape = shape
        self.scheduler = DummyScheduler() if scheduler is None else scheduler

        self.grad_norm = grad_norm
        self.num_accum = num_accum
        self.device = device
        self.chkpt_intv = chkpt_intv
        self.image_intv = image_intv
        self.num_save_images = num_save_images

        if distributed:
            assert sampler is not None
        self.distributed = distributed
        self.is_main = rank == 0

        self.use_cfg = use_cfg
        self.use_ema = use_ema
        if self.is_main and use_ema:
            self.ema = EMA(self.model, decay=ema_decay)
        else:
            self.ema = nullcontext()

        self.stats = RunningStatistics(loss=None)

    def loss(self, x, y):
        B = x.shape[0]
        T = self.timesteps
        if T > 0:
            t = torch.randint(
                T, size=(B, ), dtype=torch.float32, device=self.device
            ).add(1).div(self.timesteps)
        else:
            t = torch.rand((B, ), dtype=torch.float32, device=self.device)
        loss = self.diffusion.train_losses(self.model, x_0=x, t=t, y=y)
        assert loss.shape == (B, )
        return loss

    def step(self, x, y, update=True):
        B = x.shape[0]
        loss = self.loss(x, y).mean()
        loss.div(self.num_accum).backward()
        if update:
            # gradient clipping by global norm
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            # adjust learning rate every step (warming up)
            self.scheduler.step()
            if self.is_main and self.use_ema:
                self.ema.update()
        self.stats.update(B, loss=loss.item() * B)
        

    def sample_fn(
            self, noises, labels,
            diffusion=None, use_ddim=False, batch_size=-1):
        if diffusion is None:
            diffusion = self.diffusion
        shape = noises.shape
        with self.ema:
            if batch_size == -1:
                sample = diffusion.p_sample(
                    denoise_fn=self.model, shape=shape, device=self.device,
                    noise=noises, label=labels, use_ddim=use_ddim)
            else:
                sample = []
                for i in range(0, noises.shape[0], batch_size):
                    _slice = slice(i, i+batch_size)
                    _shape = (min(noises.shape[0] - i, batch_size), ) + tuple(shape[1:])
                    sample.append(diffusion.p_sample(
                        denoise_fn=self.model, shape=_shape, device=self.device,
                        noise=noises[_slice], label=labels[_slice], use_ddim=use_ddim))
                sample = torch.cat(sample, dim=0)
        assert sample.grad is None
        return sample

    def random_labels(self):
        if self.multitags:
            inds = torch.randint(
                len(self.trainloader.dataset),
                size=(self.num_save_images,))
            labels = torch.as_tensor(
                self.trainloader.dataset.targets[inds], dtype=torch.float32)
        else:
            labels = torch.arange(self.num_classes, dtype=torch.float32) + 1
            repeats = torch.as_tensor([
                (self.num_save_images // self.num_classes
                 + int(i < self.num_save_images % self.num_classes))
                for i in range(self.num_classes)])
            labels = labels.repeat_interleave(repeats)
        return labels

    @property
    def num_classes(self):
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module.num_classes
        else:
            return self.model.num_classes

    @property
    def multitags(self):
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module.multitags
        else:
            return self.model.multitags

    def train(
            self,
            evaluator=None,
            noises=None,
            labels=None,
            chkpt_path=None,
            image_dir=None,
            use_ddim=False,
            sample_bsz=-1
    ):

        if self.is_main and self.num_save_images:
            if noises is None:
                # fixed x_T for image generation
                noises = torch.randn((self.num_save_images, ) + self.shape)
            if labels is None and self.num_classes:
                labels = self.random_labels()

        total_batches = 0
        ls = []
        for e in range(self.start_epoch, self.epochs):
            self.stats.reset()
            self.model.train()
            if isinstance(self.sampler, DistributedSampler):
                self.sampler.set_epoch(e)
            with tqdm(
                    self.trainloader,
                    desc=f"{e+1}/{self.epochs} epochs",
                    disable=not self.is_main
            ) as t:
                for i, (x, y) in enumerate(t):
                    total_batches += 1
                    if not self.use_cfg:
                        y = None
                    self.step(
                        x.to(self.device),
                        y.float().to(self.device)
                        if y is not None else y,
                        update=total_batches % self.num_accum == 0
                    )
                    loss = self.current_stats['loss']
                    ls.append(loss)
                    # print("Loss: {}".format(loss))
                    postfix = {'loss' : '{:^.10f}'.format(loss)}
                    t.set_postfix(postfix)
                    if i == len(self.trainloader) - 1:
                        self.model.eval()
                        if evaluator is not None:
                            eval_results = evaluator.eval(self.sample_fn)
                        else:
                            eval_results = dict()
                        results = dict()
                        results.update(self.current_stats)
                        results.update(eval_results)
                        t.set_postfix(results)
            writer.add_scalar('Loss/train', loss, e)
            torch.save(ls, 'losses_uncond.pt')
            if self.is_main:
                if not (e + 1) % self.image_intv and self.num_save_images and image_dir:
                    x = self.sample_fn(
                        noises, labels, use_ddim=use_ddim, batch_size=sample_bsz)
                    save_image(x, os.path.join(image_dir, f"{e+1}.jpg"))
                if not (e + 1) % self.chkpt_intv and chkpt_path:
                    self.save_checkpoint(chkpt_path, epoch=e+1, **results)
            if self.distributed:
                dist.barrier()  # synchronize all processes here

    @property
    def trainees(self):
        roster = ["model", "optimizer"]
        if self.use_ema:
            roster.append("ema")
        if self.scheduler is not None:
            roster.append("scheduler")
        return roster

    @property
    def current_stats(self):
        return self.stats.extract()

    def load_checkpoint(self, chkpt_path, map_location):
        chkpt = torch.load(chkpt_path, map_location=map_location)
        for trainee in self.trainees:
            try:
                getattr(self, trainee).load_state_dict(chkpt[trainee])
            except RuntimeError:
                _chkpt = chkpt[trainee]["shadow"] if trainee == "ema" else chkpt[trainee]
                for k in list(_chkpt.keys()):
                    if k.split(".")[0] == "module":
                        _chkpt[".".join(k.split(".")[1:])] = _chkpt.pop(k)
                    else:
                        _chkpt[".".join(['module'] + k.split("."))] = _chkpt.pop(k)
                getattr(self, trainee).load_state_dict(chkpt[trainee])
            except AttributeError:
                continue
        self.start_epoch = chkpt["epoch"]

    def save_checkpoint(self, chkpt_path, **extra_info):
        chkpt = []
        for k, v in self.named_state_dicts():
            chkpt.append((k, v))
        for k, v in extra_info.items():
            chkpt.append((k, v))
        if "epoch" in extra_info:
            chkpt_path = re.sub(r"(_\d+)?\.pt", f"_{extra_info['epoch']}.pt", chkpt_path)
        torch.save(dict(chkpt), chkpt_path)

    def named_state_dicts(self):
        for k in self.trainees:
            yield k, getattr(self, k).state_dict()


class Evaluator:
    def __init__(
            self,
            dataset,
            diffusion=None,
            eval_batch_size=256,
            max_eval_count=30000,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.diffusion = diffusion
        # inception stats
        self.istats = InceptionStatistics(device=device)
        self.eval_batch_size = eval_batch_size
        self.max_eval_count = max_eval_count
        self.device = device
        self.target_mean, self.target_var = get_precomputed(dataset)

    def eval(self, sample_fn):
        self.istats.reset()
        for _ in range(0, self.max_eval_count + self.eval_batch_size, self.eval_batch_size):
            x = sample_fn(self.eval_batch_size, diffusion=self.diffusion)
            # x = sample_fn(torch.randn(shape, device=self.device), labels, diffusion=self.diffusion, 
            #                 use_ddim=True, batch_size=self.eval_batch_size)
            self.istats(x.to(self.device))
        gen_mean, gen_var = self.istats.get_statistics()
        return {"fid": calc_fd(gen_mean, gen_var, self.target_mean, self.target_var)}
    

class DistillationTrainer:
    def __init__(
            self,
            teacher_model,
            student_model,
            z_model,
            student_optimizer,
            z_optimizer,
            teacher_diffusion,
            student_diffusion,
            speed_up,
            timesteps,
            epochs,
            trainloader,
            sampler=None,
            scheduler=None,
            use_cfg=False,
            use_ema=False,
            grad_norm=1.0,
            num_accum=1,
            shape=None,
            device=torch.device("cpu"),
            chkpt_intv=5,  # save a checkpoint every [chkpt_intv] epochs
            image_intv=1,  # generate images every [image_intv] epochs
            num_save_images=64,
            ema_decay=0.9999,
            distributed=False,
            rank=0  # process id for distributed training
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.z_model = z_model
        self.student_optimizer = student_optimizer
        self.z_optimizer = z_optimizer
        self.teacher_diffusion = teacher_diffusion
        self.student_diffusion = student_diffusion
        self.speed_up = speed_up
        self.timesteps = timesteps
        self.epochs = epochs
        self.start_epoch = 0
        self.trainloader = trainloader
        self.sampler = sampler
        if shape is None:
            shape = next(iter(trainloader))[0].shape[1:]
        self.shape = shape
        self.scheduler = DummyScheduler() if scheduler is None else scheduler

        self.grad_norm = grad_norm
        self.num_accum = num_accum
        self.device = device
        self.chkpt_intv = chkpt_intv
        self.image_intv = image_intv
        self.num_save_images = num_save_images

        if distributed:
            assert sampler is not None
        self.distributed = distributed
        self.is_main = rank == 0

        self.use_cfg = use_cfg
        self.use_ema = use_ema
        if self.is_main and use_ema:
            self.ema = EMA(self.student_model, decay=ema_decay)
        else:
            self.ema = nullcontext()

        self.stats = RunningStatistics(loss=None)

    def loss(self, x, y):
        B = x.shape[0]
        S = self.timesteps    # distillate steps
        T = self.teacher_diffusion.sample_timesteps  # self.timesteps
        if S > 0:
            t = torch.randint(
                S, size=(B, ), dtype=torch.float32, device=self.device
            ).add(1).div(S)
        else:
            t = torch.rand((B, ), dtype=torch.float32, device=self.device)    #[0,1)
        loss = self.teacher_diffusion.distill_losses_enc(self.student_diffusion, self.teacher_model, 
                                        self.student_model, x, t, y, self.speed_up)
        assert loss.shape == (B, )
        return loss

    def distill_step(self, x, y, update=True):
        B = x.shape[0]
        loss = self.loss(x, y).mean()
        loss.div(self.num_accum).backward()
        if update:
            # gradient clipping by global norm
            nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=self.grad_norm)
            self.student_optimizer.step()
            self.student_optimizer.zero_grad(set_to_none=True)
            # adjust learning rate every step (warming up)
            self.scheduler.step()
            if self.is_main and self.use_ema:
                self.ema.update()
        self.stats.update(B, loss=loss.item() * B)
        

    def sample_fn(
            self, noises, labels,
            diffusion=None, use_ddim=False, batch_size=-1):
        if diffusion is None:
            diffusion = self.student_diffusion
        shape = noises.shape
        with self.ema:
            if batch_size == -1:
                sample = diffusion.p_sample(
                    denoise_fn=self.student_model, shape=shape, device=self.device,
                    noise=noises, label=labels, use_ddim=use_ddim)
            else:
                sample = []
                for i in range(0, noises.shape[0], batch_size):
                    _slice = slice(i, i+batch_size)
                    _shape = (min(noises.shape[0] - i, batch_size), ) + tuple(shape[1:])
                    sample.append(diffusion.p_sample(
                        denoise_fn=self.student_model, shape=_shape, device=self.device,
                        noise=noises[_slice], label=labels[_slice], use_ddim=use_ddim))
                sample = torch.cat(sample, dim=0)
        assert sample.grad is None
        return sample

    def random_labels(self):
        if self.multitags:
            inds = torch.randint(
                len(self.trainloader.dataset),
                size=(self.num_save_images,))
            labels = torch.as_tensor(
                self.trainloader.dataset.targets[inds], dtype=torch.float32)
        else:
            labels = torch.arange(self.num_classes, dtype=torch.float32) + 1
            repeats = torch.as_tensor([
                (self.num_save_images // self.num_classes
                 + int(i < self.num_save_images % self.num_classes))
                for i in range(self.num_classes)])
            labels = labels.repeat_interleave(repeats)
        return labels

    @property
    def num_classes(self):
        if isinstance(self.student_model, DistributedDataParallel):
            return self.student_model.module.num_classes
        else:
            return self.student_model.num_classes

    @property
    def multitags(self):
        if isinstance(self.student_model, DistributedDataParallel):
            return self.student_model.module.multitags
        else:
            return self.student_model.multitags

    def train_G(
            self,
            evaluator=None,
            noises=None,
            labels=None,
            student_chkpt_path=None,
            image_dir=None,
            use_ddim=False,
            sample_bsz=-1
    ):

        if self.is_main and self.num_save_images:
            if noises is None:
                # fixed x_T for image generation
                noises = torch.randn((self.num_save_images, ) + self.shape)
            if labels is None and self.num_classes:
                labels = self.random_labels()

        total_batches = 0
        N = 0
        L_tot = 0
        # self.save_checkpoint(student_chkpt_path, name="student", epoch=1200+1)
        for e in range(self.start_epoch, self.epochs):
            self.stats.reset()
            self.student_model.train()
            self.teacher_model.eval()
            if isinstance(self.sampler, DistributedSampler):
                self.sampler.set_epoch(e)
            with tqdm(
                    self.trainloader,
                    desc=f"{e+1}/{self.epochs} epochs",
                    disable=not self.is_main
            ) as t:
                for i, (x, y) in enumerate(t):
                    total_batches += 1
                    if not self.use_cfg:
                        y = None
                    self.distill_step(
                        x.to(self.device),
                        y.float().to(self.device)
                        if y is not None else y,
                        update=total_batches % self.num_accum == 0
                    )
                    loss = self.current_stats['loss']
                    L_tot += loss
                    N += 1
                    # print("Loss: {}".format(loss.item()))
                    # postfix = {'loss' : '{:^.10f}'.format(L_tot / N)}
                    postfix = 'loss_batch:{:^.8f} loss_single:{:^.8f}'.format(L_tot / N, loss)
                    t.set_postfix_str(postfix)
                    # postfix = {'loss' : '{:^.12f}'.format(loss)}
                    # t.set_postfix(postfix)
                    if i == len(self.trainloader) - 1:
                        self.student_model.eval()
                        if evaluator is not None:
                            eval_results = evaluator.eval(self.sample_fn)
                        else:
                            eval_results = dict()
                        results = dict()
                        results.update(self.current_stats)
                        results.update(eval_results)
                        t.set_postfix(results)
            if self.is_main:
                if not (e + 1) % self.image_intv and self.num_save_images and image_dir:
                    x = self.sample_fn(
                        noises, labels, use_ddim=use_ddim, batch_size=sample_bsz)
                    save_image(x, os.path.join(image_dir, f"{e+1}.jpg"))
                if not (e + 1) % self.chkpt_intv and student_chkpt_path:
                    self.save_checkpoint(student_chkpt_path, name="student", epoch=e+1, **results)
            if self.distributed:
                dist.barrier()  # synchronize all processes here

    @property
    def trainees(self):
        roster = ["model", "optimizer"]
        if self.use_ema:
            roster.append("ema")
        if self.scheduler is not None:
            roster.append("scheduler")
        return roster

    @property
    def current_stats(self):
        return self.stats.extract()

    def load_checkpoint(self, model, chkpt_path, map_location):
        chkpt = torch.load(chkpt_path, map_location=map_location)
        for trainee in self.trainees:
            try:
                trainee_model = model + "_" + trainee
                getattr(self, trainee_model).load_state_dict(chkpt[trainee])
            except RuntimeError:
                _chkpt = chkpt[trainee]["shadow"] if trainee == "ema" else chkpt[trainee]
                for k in list(_chkpt.keys()):
                    if k.split(".")[0] == "module":
                        _chkpt[".".join(k.split(".")[1:])] = _chkpt.pop(k)
                    else:
                        _chkpt[".".join(['module'] + k.split("."))] = _chkpt.pop(k)
                getattr(self, trainee_model).load_state_dict(chkpt[trainee])
            except AttributeError:
                continue
        self.start_epoch = chkpt["epoch"]

    def save_checkpoint(self, chkpt_path, name, **extra_info):
        chkpt = []
        for k, v in self.named_state_dicts(name):
            chkpt.append((k, v))
        for k, v in extra_info.items():
            chkpt.append((k, v))
        if "epoch" in extra_info:
            chkpt_path = re.sub(r"(_\d+)?\.pt", f"_{extra_info['epoch']}.pt", chkpt_path)
        torch.save(dict(chkpt), chkpt_path)

    def named_state_dicts(self, name):
        for k in self.trainees:
            model = k
            if k == "model" or  k == "optimizer":
                model = name + "_" + k
            yield k, getattr(self, model).state_dict()

