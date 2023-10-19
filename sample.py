"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.utils import save_image
from scripts import dist_util, logger
from scripts.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    student_model, student_diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    latent_denoise_fn, _ = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    student_model.to(dist_util.dev())
    latent_denoise_fn.to(dist_util.dev())
    if args.use_fp16:
        student_model.convert_to_fp16()
        latent_denoise_fn.convert_to_fp16()
    student_model.eval()
    latent_denoise_fn.eval()
    inferred_latents = torch.load(args.inferred_latents_path, map_location=torch.device('cpu'))
    latents_mean = inferred_latents["mean"].to(self.device)
    latents_std = inferred_latents["std"].to(self.device)

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:    
        sample_fn = (
            diffusion.latent_diffusion_sample if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            latent_denoise_fn=latent_denoise_fn,
            decoder=student_model.gradient_predictor,
            x_T=torch.randn(num, self.config["image_channel"],self.config["image_size"],self.config["image_size"]).cuda(),
            latents_mean=latents_mean,
            latents_std=latents_std,
        )
        print(sample)
        to_range_0_1 = lambda x: (x + 1.) / 2
        # save_image(to_range_0_1(sample), '1.png')

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_31.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=50000,
        batch_size=100,
        use_ddim=False,
        use_gradient_predictor=True,
        student_model_path="./models/students/SFERD_four_step.pt",
        inferred_latents_path= "./models/latents/imagenet64.pt"
        latent_diffusion_path="./models/latents_denoise_fn/imagenet64_checkpoints.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
