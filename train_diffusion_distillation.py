import os
import json
import torch
from torch import nn
from datetime import datetime
from torch.optim import AdamW, lr_scheduler
from diffusion import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.distributed.elastic.multiprocessing import errors
from functools import partial
import copy
from scripts import image_datasets, dist_util, logger
from scripts.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_semantic_encoder,
    add_dict_to_argparser,
    args_to_dict,
)

@errors.record
def main(args):

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    teacher_model, teacher_diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    student_model, student_diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # z_model includes the semantic encoder and gradient predictor
    z_model = create_semantic_net(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    teacher_model = teacher_model.to(dist_util.dev())
    student_model = student_model.to(dist_util.dev())
    z_model = z_model.to(dist_util.dev())

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, teacher_diffusion)
    
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        z_model=z_model,
        student_optimizer=student_optimizer,
        z_optimizer=z_optimizer,
        teacher_diffusion=teacher_diffusion,
        student_diffusion=student_diffusion,
        speed_up=speed_up,
        timesteps=train_timesteps,
        epochs=epochs,
        trainloader=trainloader,
        sampler=sampler,
        scheduler=scheduler,
        use_cfg=args.use_cfg,
        use_ema=args.use_ema,
        grad_norm=grad_norm,
        num_accum=args.num_accum,
        shape=image_shape,
        device=train_device,
        chkpt_intv=chkpt_intv,
        image_intv=image_intv,
        num_save_images=num_save_images,
        ema_decay=args.ema_decay,
        rank=rank,
        distributed=distributed
    )
    evaluator = Evaluator(dataset=dataset, device=eval_device) if args.eval else None

    # use cudnn benchmarking algorithm to select the best conv algorithm
    if torch.backends.cudnn.is_available():  # noqa
        torch.backends.cudnn.benchmark = True  # noqa
        logger(f"cuDNN benchmark: ON")

    logger("Training starts...", flush=True)
    trainer.train_G(
        evaluator,
        image_dir=image_dir,
        use_ddim=args.use_ddim,
        sample_bsz=args.sample_bsz
    )

    

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "celeba"], default="cifar10")
    parser.add_argument("--root", default="~/datasets", type=str, help="root directory of datasets")
    parser.add_argument("--epochs", default=3000, type=int, help="total number of training epochs")
    parser.add_argument("--lr", default=0.0002, type=float, help="learning rate")
    parser.add_argument("--beta1", default=0.9, type=float, help="beta_1 in Adam")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta_2 in Adam")
    parser.add_argument("--weight-decay", default=0., type=float,
                        help="decoupled weight_decay factor in Adam")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-accum", default=1, type=int, help=(
        "number of batches before weight update, a.k.a. gradient a                                                ccumulation"))
    parser.add_argument("--teacher-sample-timesteps", default=256, type=int, help="number of teacher diffusion steps for sampling")
    parser.add_argument("--train-timesteps", default=128, type=int, help=(
        "number of student diffusion steps for training (0 indicates continuous training)"))
    parser.add_argument("--logsnr-schedule", choices=["linear", "sigmoid", "cosine", "legacy"], default="cosine")
    parser.add_argument("--logsnr-max", default=20., type=float)
    parser.add_argument("--logsnr-min", default=-20., type=float)
    parser.add_argument("--model-out-type", choices=["x_0", "eps", "both", "v"], default="v", type=str)
    parser.add_argument("--model-var-type", choices=["fixed_small", "fixed_large", "fixed_medium"], default="fixed_large", type=str)
    parser.add_argument("--reweight-type", choices=["constant", "snr", "truncated_snr", "alpha2"], default="truncated_snr", type=str)
    parser.add_argument("--loss-type", choices=["kl", "mse"], default="mse", type=str)
    parser.add_argument("--intp-frac", default=0., type=float)
    parser.add_argument("--use-cfg", default=True, help="whether to use classifier-free guidance")
    parser.add_argument("--w-guide", default=0.3, type=float, help="classifier-free guidance strength")
    parser.add_argument("--p-uncond", default=0.1, type=float, help="probability of unconditional training")
    parser.add_argument("--num-workers", default=4, type=int, help="number of workers for data loading")
    parser.add_argument("--train-device", default="cuda:0", type=str)
    parser.add_argument("--eval-device", default="cuda:0", type=str)
    parser.add_argument("--image-dir", default="./images/train", type=str)
    parser.add_argument("--image-intv", default=10, type=int)
    parser.add_argument("--num-save-images", default=64, type=int, help="number of images to generate & save")
    parser.add_argument("--sample-bsz", default=-1, type=int, help="batch size for sampling")
    parser.add_argument("--config-dir", default="./configs", type=str)
    parser.add_argument("--chkpt-dir", default="./checkpoints", type=str)
    parser.add_argument("--teacher-chkpt-name", default="teacher/v_diffusion_256steps_600.pt", type=str)
    parser.add_argument("--student-chkpt-name", default="teacher/v_diffusion_256steps_600.pt", type=str)
    parser.add_argument("--chkpt-intv", default=300, type=int, help="frequency of saving a checkpoint")
    parser.add_argument("--seed", default=1234, type=int, help="random seed")
    parser.add_argument("--resume", default=True, help="to resume training from a checkpoint")
    parser.add_argument("--eval", action="store_true", help="whether to evaluate fid during training")
    parser.add_argument("--use-ema", default=True, help="whether to use exponential moving average")
    parser.add_argument("--use-ddim", default=True, help="whether to use DDIM sampler")
    parser.add_argument("--ema-decay", default=0.9999, type=float, help="decay factor of ema")
    parser.add_argument("--distributed", action="store_true", help="whether to use distributed training")
    # Distillation
    parser.add_argument("--speed_up", type=int, default=2)

    main(parser.parse_args())

    # python train.py --use-ema --use-ddim --use-cfg --eval --resume
    # python -m torch.distributed.run --standalone --nproc_per_node 3 --rdzv_backend c10d distillate_G.py --distributed




