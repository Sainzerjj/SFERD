"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
import math
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms as T
from functools import partial
from .ddim import DDIM


def broadcast_to(
        arr, x,
        dtype=None, device=None, ndim=None):
    if x is not None:
        dtype = dtype or x.dtype
        device = device or x.device
        ndim = ndim or x.ndim
    out = torch.as_tensor(arr, dtype=dtype, device=device)
    return out.reshape((-1,) + (1,) * (ndim - 1))


def get_logsnr_schedule(schedule, logsnr_min: float = -20., logsnr_max: float = 20.):
    """
    schedule is named according to the relationship between alpha2 and t,
    i.e. alpha2 as a XX function of affine transformation of t (except for legacy)
    """

    logsnr_min, logsnr_max = torch.as_tensor(logsnr_min), torch.as_tensor(logsnr_max)
    if schedule == "linear":
        def logsnr2t(logsnr):
            return torch.sigmoid(logsnr)

        def t2logsnr(t):
            return torch.logit(t)
    elif schedule == "sigmoid":
        logsnr_range = logsnr_max - logsnr_min

        def logsnr2t(logsnr):
            return (logsnr_max - logsnr) / logsnr_range

        def t2logsnr(t):
            return logsnr_max - t * logsnr_range
    elif schedule == "cosine":
        def logsnr2t(logsnr):
            return torch.atan(torch.exp(-0.5 * logsnr)).div(0.5 * math.pi)
            
        def t2logsnr(t):
            return -2 * torch.log(torch.tan(t * math.pi * 0.5))
    elif schedule == "legacy":
        """
        continuous version of the (discrete) linear schedule used by \
          Ho, Jonathan, Ajay Jain, and Pieter Abbeel. \
            "Denoising diffusion probabilistic models." \
              Advances in Neural Information Processing Systems 33 (2020): 6840-6851.
        """
        delta_max, delta_min = (
            torch.as_tensor(1 - 0.0001),
            torch.as_tensor(1 - 0.02))
        delta_max_m1 = torch.as_tensor(-0.0001)
        log_delta_max = torch.log1p(delta_max_m1)
        log_delta_min = torch.log1p(torch.as_tensor(-0.02))
        delta_range = delta_max - delta_min
        log_alpha_range = (delta_max * log_delta_max -
                           delta_min * log_delta_min) / delta_range - 1

        def schedule_fn(t):
            tau = delta_max - delta_range * t
            tau_m1 = delta_max_m1 - delta_range * t
            log_alpha = (
                    (delta_max * log_delta_max - tau * torch.log1p(tau_m1))
                    / delta_range - t).mul(-20. / log_alpha_range).add(-2.0612e-09)
            return log_alpha - stable_log1mexp(log_alpha)

        return schedule_fn
    else:
        raise NotImplementedError
    b = logsnr2t(logsnr_max)
    a = logsnr2t(logsnr_min) - b

    def schedule_fn(t):
        _a, _b = broadcast_to(a, t), broadcast_to(b, t)
        return t2logsnr(_a * t + _b)

    return schedule_fn


def stable_log1mexp(x):
    """
    numerically stable version of log(1-exp(x)), x<0
    """
    assert torch.all(x < 0.)
    return torch.where(
        x < -9,
        torch.log1p(torch.exp(x).neg()),
        torch.log(torch.expm1(x).neg()))

# SNR = log(alpha^2/sigma^2)   z_t = alpha * x + sigma * noise
def logsnr_to_posterior(logsnr_s, logsnr_t, var_type: str, intp_frac=None):
    # upcast to double precision to reduce precision loss
    logsnr_s, logsnr_t = (
        logsnr_s.to(torch.float64), logsnr_t.to(torch.float64))

    log_alpha_st = 0.5 * (F.logsigmoid(logsnr_s) - F.logsigmoid(logsnr_t))
    logr = logsnr_t - logsnr_s
    log_one_minus_r = stable_log1mexp(logr)
    mean_coef1 = (logr + log_alpha_st).exp()
    mean_coef2 = (log_one_minus_r + 0.5 * F.logsigmoid(logsnr_s)).exp()

    # strictly speaking, only when var_type == "small",
    # does `logvar` calculated here represent the logarithm
    # of the true posterior variance
    if var_type == "fixed_large":
        logvar = log_one_minus_r + F.logsigmoid(-logsnr_t)
    elif var_type == "fixed_small":
        logvar = log_one_minus_r + F.logsigmoid(-logsnr_s)
    elif var_type == "fixed_medium":
        # linear interpolation in log-space
        assert isinstance(intp_frac, (float, torch.Tensor))
        logvar = (
                intp_frac * (log_one_minus_r + F.logsigmoid(-logsnr_t)) +
                (1. - intp_frac) * (log_one_minus_r + F.logsigmoid(-logsnr_s))
        )
    else:
        raise NotImplementedError(var_type)

    return tuple(map(lambda x: x.to(torch.float32), (mean_coef1, mean_coef2, logvar)))


DEBUG = False

# upcast to double precision to reduce precision loss
def logsnr_to_posterior_ddim(logsnr_s, logsnr_t, eta: float):
    logsnr_s, logsnr_t = (
        logsnr_s.to(torch.float64), logsnr_t.to(torch.float64))

    if not DEBUG and eta == 1.:
        return logsnr_to_posterior(logsnr_s, logsnr_t, "fixed_small")
    else:
        if DEBUG:
            print("Debugging mode...")
        log_alpha_st = 0.5 * (F.logsigmoid(logsnr_s) - F.logsigmoid(logsnr_t))
        logr = logsnr_t - logsnr_s
        if eta == 0:
            log_one_minus_sqrt_r = stable_log1mexp(0.5 * logr)
            mean_coef1 = (F.logsigmoid(-logsnr_s) - F.logsigmoid(-logsnr_t)).mul(0.5).exp()
            mean_coef2 = (log_one_minus_sqrt_r + 0.5 * F.logsigmoid(logsnr_s)).exp()
            logvar = torch.as_tensor(-math.inf)
        else:
            log_one_minus_r = stable_log1mexp(logr)
            logvar = log_one_minus_r + F.logsigmoid(-logsnr_s) + 2 * math.log(eta)
            mean_coef1 = stable_log1mexp(
                logvar - F.logsigmoid(-logsnr_s))
            mean_coef1 += F.logsigmoid(-logsnr_s) - F.logsigmoid(-logsnr_t)
            mean_coef1 *= 0.5
            mean_coef2 = stable_log1mexp(mean_coef1 - log_alpha_st).add(
                0.5 * F.logsigmoid(logsnr_s))
            mean_coef1, mean_coef2 = mean_coef1.exp(), mean_coef2.exp()

        return tuple(map(lambda x: x.to(torch.float32), (mean_coef1, mean_coef2, logvar)))


@torch.jit.script
def pred_x0_from_eps(x_t, eps, logsnr_t):
    return x_t.div(torch.sigmoid(logsnr_t).sqrt()) - eps.mul(logsnr_t.neg().mul(.5).exp())


def pred_x0_from_x0eps(x_t, x0eps, logsnr_t):
    x_0, eps = x0eps.chunk(2, dim=1)
    _x_0 = pred_x0_from_eps(x_t, eps, logsnr_t)
    return x_0.mul(torch.sigmoid(-logsnr_t)) + _x_0.mul(torch.sigmoid(logsnr_t))


@torch.jit.script
def pred_eps_from_x0(x_t, x_0, logsnr_t):
    return x_t.mul(torch.sigmoid(-logsnr_t).sqrt()) - x_0.mul(logsnr_t.mul(.5).exp())


@torch.jit.script
def pred_v_from_x0eps(x_0, eps, logsnr_t):
    return -x_0.mul(torch.sigmoid(-logsnr_t).sqrt()) + eps.mul(torch.sigmoid(logsnr_t).sqrt())


@torch.jit.script
def pred_x0_from_v(x_t, v, logsnr_t):
    return x_t.mul(torch.sigmoid(logsnr_t).sqrt()) - v.mul(torch.sigmoid(-logsnr_t).sqrt())


@torch.jit.script
def pred_eps_from_v(x_t, v, logsnr_t):
    return x_t.mul(torch.sigmoid(-logsnr_t).sqrt()) + v.mul(torch.sigmoid(logsnr_t).sqrt())


def q_sample(x_0, logsnr_t, eps=None):
    """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
    """
    if eps is None:
        eps = torch.randn_like(x_0)
    return x_0.mul(torch.sigmoid(logsnr_t).sqrt()) + eps.mul(torch.sigmoid(-logsnr_t).sqrt())


@torch.jit.script
def q_mean_var(x_0, logsnr_t):
    return x_0.mul(torch.sigmoid(logsnr_t).sqrt()), F.logsigmoid(-logsnr_t)


def raise_error_with_msg(msg):
    def raise_error(*args, **kwargs):
        raise NotImplementedError(msg)

    return raise_error


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :logsnr_fn: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """
    def __init__(
            self,
            logsnr_fn,
            sample_timesteps,
            model_out_type,
            model_var_type,
            reweight_type,
            loss_type,
            intp_frac=None,
            w_guide=0.1,
            p_uncond=0.1,
            use_ddim=False
    ):
        self.logsnr_fn = logsnr_fn
        self.sample_timesteps = sample_timesteps

        self.model_out_type = model_out_type
        self.model_var_type = model_var_type
        # self.pre_out = 0
        # self.pre_x_0 = 0
        # from mse_target to re-weighting strategy
        # x0 -> constant
        # eps -> SNR
        # both -> truncated_SNR, i.e. max(1, SNR)
        self.reweight_type = reweight_type
        self.loss_type = loss_type
        self.intp_frac = intp_frac
        self.w_guide = w_guide
        self.p_uncond = p_uncond

        self.sel_attn_depth = 8
        self.sel_attn_block = "output"
        self.num_heads = 1
        self.blur_sigma = 3
        

    def extract_coef_at_t(schedule, t, x_shape):
        return torch.gather(schedule, -1, t).reshape([x_shape[0]] + [1] * (len(x_shape) - 1))

    def t2logsnr(self, *ts, x=None):
        _broadcast_to = lambda t: broadcast_to(
            self.logsnr_fn(t), x=x)
        return tuple(map(_broadcast_to, ts))

    def predicted_noise_to_predicted_x_0(self, x_t, t, predicted_noise):
        shape = x_t.shape
        return self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t \
               - self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise

    def noise_p_sample(self, x_t, t, predicted_noise):
        shape = x_t.shape
        predicted_mean = \
            self.extract_coef_at_t(self.noise_posterior_mean_x_t_coef, t, shape) * x_t - \
            self.extract_coef_at_t(self.noise_posterior_mean_noise_coef, t, shape) * predicted_noise
        log_variance_clipped = self.extract_coef_at_t(self.posterior_log_variance_clipped, t, shape)
        noise = torch.randn(shape, device=self.device)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape([shape[0]] + [1] * (len(shape) - 1))
        return predicted_mean + nonzero_mask * (0.5 * log_variance_clipped).exp() * noise
    
    def q_posterior_mean_var(
            self, x_0, x_t, logsnr_s, logsnr_t, model_var_type=None, intp_frac=None):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        model_var_type = model_var_type or self.model_var_type
        intp_frac = self.intp_frac or intp_frac
        mean_coef1, mean_coef2, posterior_logvar = logsnr_to_posterior(
            logsnr_s, logsnr_t, var_type=model_var_type, intp_frac=intp_frac)
        posterior_mean = mean_coef1 * x_t + mean_coef2 * x_0
        return posterior_mean, posterior_logvar

    def q_posterior_mean_var_ddim(self, x_0, x_t, logsnr_s, logsnr_t):
        """
        Compute the mean and variance of the diffusion posterior of DDIM:

            q(x_{t-1} | x_t, x_0)

        """
        mean_coef1, mean_coef2, posterior_logvar = logsnr_to_posterior_ddim(
            logsnr_s, logsnr_t, eta=0.)
        posterior_mean = mean_coef1 * x_t + mean_coef2 * x_0
        return posterior_mean, posterior_logvar

    def p_mean_var(
            self, denoise_fn, x_t, s, t, y, clip_denoised, return_pred, use_ddim=False):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param denoise_fn: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: 
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'pred_xstart': the prediction for x_0.
        """
        out = denoise_fn(x_t, t, y=y)
        # save_image_cuda(out, "out.png", nrow=8, normalize=True, value_range=(-1., 1.))
        logsnr_s, logsnr_t = self.t2logsnr(s, t, x=x_t)

        if self.model_var_type == "learned":
            out, intp_frac = out.chunk(2, dim=1)
            intp_frac = torch.sigmoid(intp_frac)  # re-scale to (0, 1)
        else:
            intp_frac = None

        # calculate the mean estimate
        _clip = (lambda x: x.clamp(-1., 1.)) if clip_denoised else (lambda x: x)
        _raise_error = raise_error_with_msg(self.model_out_type)
        pred_x_0 = _clip({
                             "x0": lambda arg1, arg2, arg3: arg2,
                             "eps": pred_x0_from_eps,
                             "both": pred_x0_from_x0eps,
                             "v": pred_x0_from_v
                         }.get(self.model_out_type, _raise_error)(x_t, out, logsnr_t))
        # save_image_cuda(pred_x_0, "1.png", nrow=8, normalize=True, value_range=(-1., 1.))
        if use_ddim:
            model_mean, model_logvar = self.q_posterior_mean_var_ddim(
                x_0=pred_x_0, x_t=x_t,
                logsnr_s=logsnr_s, logsnr_t=logsnr_t)
        else:
            model_mean, model_logvar = self.q_posterior_mean_var(
                x_0=pred_x_0, x_t=x_t,
                logsnr_s=logsnr_s, logsnr_t=logsnr_t, intp_frac=intp_frac)

        if return_pred:
            return model_mean, model_logvar, pred_x_0
        else:
            return model_mean, model_logvar
    
    def get_ddim_betas_and_timestep_map(ddim_style, original_alphas_cumprod):
        original_timesteps = original_alphas_cumprod.shape[0]
        dim_step = int(ddim_style[len("ddim"):])
        use_timesteps = set([int(s) for s in list(np.linspace(0, original_timesteps - 1, dim_step + 1))])
        timestep_map = []

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(original_alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)

        return np.array(new_betas), torch.tensor(timestep_map, dtype=torch.long)
    
    def p_mean_var_enc(
            self, denoise_fn, x_t, s, t, y, x_0, clip_denoised, return_pred, use_ddim=False):
        """
        Apply the SFERD to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param denoised_fn: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param x_0: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: 
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'pred_xstart': the prediction for x_0.
        """
        all_out = denoise_fn(x_t, t, y=y, x_start=x_0)
        # get U-Net prediction
        out = all_out.pred
        # get atention map
        attn_map = all_out.attention
        # save_image_cuda(out, "out.png", nrow=8, normalize=True, value_range=(-1., 1.))
        logsnr_s, logsnr_t = self.t2logsnr(s, t, x=x_t)

        if self.model_var_type == "learned":
            out, intp_frac = out.chunk(2, dim=1)
            intp_frac = torch.sigmoid(intp_frac)  # re-scale to (0, 1)
        else:
            intp_frac = None

        # calculate the mean estimate
        _clip = (lambda x: x.clamp(-1., 1.)) if clip_denoised else (lambda x: x)
        _raise_error = raise_error_with_msg(self.model_out_type)
        pred_x_0 = _clip({
                             "x0": lambda arg1, arg2, arg3: arg2,
                             "eps": pred_x0_from_eps,
                             "both": pred_x0_from_x0eps,
                             "v": pred_x0_from_v
                         }.get(self.model_out_type, _raise_error)(x_t, out, logsnr_t))
        pred_eps = pred_eps_from_v(x_t, out, logsnr_t)
        guide_scale = 1.3
        blur_sigma = 3
        mask_blurred = self.attention_masking(
                pred_x_0,
                t,
                attn_map,
                # prev_noise=pred_eps,
                x_real = x_0, 
                blur_sigma=blur_sigma,
            )
        # save_image_cuda(mask_blurred, "result_pred_x.png", nrow=8, normalize=True, value_range=(-1., 1.))
        mask_blurred = q_sample(mask_blurred, logsnr_t, eps=pred_eps)
        _, _, uncond_eps = self.p_pred_x_0_enc(denoise_fn, mask_blurred, t, y, pred_x_0, clip_denoised=True)
        guided_eps = uncond_eps + guide_scale * (pred_eps - uncond_eps)
        pred_x_0 = pred_x0_from_eps(x_t, guided_eps, logsnr_t)
        # save_image_cuda(pred_x_0, "2.png", nrow=8, normalize=True, value_range=(-1., 1.))
        if use_ddim:
            model_mean, model_logvar = self.q_posterior_mean_var_ddim(
                x_0=pred_x_0, x_t=x_t,
                logsnr_s=logsnr_s, logsnr_t=logsnr_t)
        else:
            model_mean, model_logvar = self.q_posterior_mean_var(
                x_0=pred_x_0, x_t=x_t,
                logsnr_s=logsnr_s, logsnr_t=logsnr_t, intp_frac=intp_frac)

        if return_pred:
            return model_mean, model_logvar, pred_x_0   # , guided_eps
        else:
            return model_mean, model_logvar
    
    def p_pred_x_0(
            self, denoise_fn, x_t, t, y, clip_denoised):
        """
        Apply the original model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param denoised_fn: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param x_0: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: 
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'pred_xstart': the prediction for x_0.
        """
        out = denoise_fn(x_t, t, y=y)
        logsnr_t, = self.t2logsnr(t, x=x_t)

        if self.model_var_type == "learned":
            out, intp_frac = out.chunk(2, dim=1)
            intp_frac = torch.sigmoid(intp_frac)  # re-scale to (0, 1)
        else:
            intp_frac = None

        # calculate the mean estimate
        _clip = (lambda x: x.clamp(-1., 1.)) if clip_denoised else (lambda x: x)
        _raise_error = raise_error_with_msg(self.model_out_type)
        pred_x_0 = _clip({
                             "x0": lambda arg1, arg2, arg3: arg2,
                             "eps": pred_x0_from_eps,
                             "both": pred_x0_from_x0eps,
                             "v": pred_x0_from_v
                         }.get(self.model_out_type, _raise_error)(x_t, out, logsnr_t))
        
        return pred_x_0
    

    def attention_masking(
        self, x, t, attn_map, x_real, blur_sigma, model_kwargs=None,
    ):
        """
        Apply the self-attention mask to produce bar{x_0^t}

        :param x: the predicted x_0 [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param attn_map: the attention map tensor at time t.
        :param prev_noise: the previously predicted epsilon to inject
            the same noise as x_t.
        :param blur_sigma: a sigma of Gaussian blur.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: the bar{x_0^t}
        """
        B, C, H, W = x.shape
        assert t.shape == (B,)
        
        if self.sel_attn_depth in [0, 1, 2] or self.sel_attn_block == "middle":
            attn_res = 8
        elif self.sel_attn_depth in [3, 4, 5]:
            attn_res = 16
        elif self.sel_attn_depth in [6, 7, 8]:
            attn_res = 32
        else:
            raise ValueError("sel_attn_depth must be in [0, 1, 2, 3, 4, 5, 6, 7, 8]")
        
        # attn_mask = attn_map.reshape(B, self.num_heads, attn_res ** 2, attn_res ** 2).mean(1, keepdim=False).sum(1, keepdim=False)
        # attn_mask = attn_mask.reshape(B, attn_res, attn_res).unsqueeze(1).repeat(1, 3, 1, 1).float()
        # attn_mask = F.interpolate(attn_mask, (H, W))
        # save_image_cuda(attn_mask, '1.png')

        # Generating attention mask
        attn_mask = attn_map.reshape(B, self.num_heads, attn_res ** 2, attn_res ** 2).mean(1, keepdim=False).sum(1, keepdim=False) > 1.0 #0.8 1.2
        attn_mask = attn_mask.reshape(B, attn_res, attn_res).unsqueeze(1).repeat(1, 3, 1, 1).int().float()
        attn_mask = F.interpolate(attn_mask, (H, W))

        # Gaussian blur
        transform = T.GaussianBlur(kernel_size=15, sigma=blur_sigma)
        x_curr = transform(x)

        # Apply attention masking
        x_curr = x_curr * (attn_mask) + x * (1 - attn_mask)
        # x_curr = x * (attn_mask) + x_real * (1 - attn_mask)
        
        return x_curr



    def p_pred_x_0_enc_student(
            self, denoise_fn, x_t, t, x_start, clip_denoised):
        """
        denoised prediction bar{x_0^t} from the distillation student model with gradient predictor
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param x_start: real data x_0 used for E_\varphi to get z_sem.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: denoised prediction
        :return: predicted gradient using G_\tau
        :return: predicted eps
        """
        z_sem = denoise_fn.encoder(x_start)
        all_out = denoise_fn(x_t, t, z_sem=z_sem)
        out = all_out.pred
        gradient = all_out.gradient
        logsnr_t, = self.t2logsnr(t, x=x_t)

        if self.model_var_type == "learned":
            out, intp_frac = out.chunk(2, dim=1)
            intp_frac = torch.sigmoid(intp_frac)  # re-scale to (0, 1)
        else:
            intp_frac = None

        # calculate the mean estimate
        _clip = (lambda x: x.clamp(-1., 1.)) if clip_denoised else (lambda x: x)
        _raise_error = raise_error_with_msg(self.model_out_type)
        pred_x_0 = _clip({
                             "x0": lambda arg1, arg2, arg3: arg2,
                             "eps": pred_x0_from_eps,
                             "both": pred_x0_from_x0eps,
                             "v": pred_x0_from_v
                         }.get(self.model_out_type, _raise_error)(x_t, out, logsnr_t))

        pred_eps = pred_eps_from_v(x_t, out, logsnr_t)
        return pred_x_0, gradient, pred_eps
    
    def p_pred_x_0_enc(
            self, denoise_fn, x_t, t, y, clip_denoised):
        """
        denoised prediction bar{x_0^t} from the distillation teacher model with gradient predictor
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: denoised prediction
        :return: attention map
        :return: predicted eps
        """
        all_out = denoise_fn(x_t, t, y=y)
        out = all_out.pred
        attn_map = all_out.attention
        logsnr_t, = self.t2logsnr(t, x=x_t)

        if self.model_var_type == "learned":
            out, intp_frac = out.chunk(2, dim=1)
            intp_frac = torch.sigmoid(intp_frac)  # re-scale to (0, 1)
        else:
            intp_frac = None

        # calculate the mean estimate
        _clip = (lambda x: x.clamp(-1., 1.)) if clip_denoised else (lambda x: x)
        _raise_error = raise_error_with_msg(self.model_out_type)
        pred_x_0 = _clip({
                             "x0": lambda arg1, arg2, arg3: arg2,
                             "eps": pred_x0_from_eps,
                             "both": pred_x0_from_x0eps,
                             "v": pred_x0_from_v
                         }.get(self.model_out_type, _raise_error)(x_t, out, logsnr_t))

        pred_eps = pred_eps_from_v(x_t, out, logsnr_t)
        return pred_x_0, attn_map, pred_eps
    # === sample ===
    def p_sample_step(
            self, denoise_fn, x_t, step, y, x_start=None,
            clip_denoised=True, return_pred=False, use_ddim=False):
        """
        Sample x_{t-1} from the original model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param attn_guidance: if True, self-attention guidance is given.
        :param guidance_kwargs: if not None, a dict of the parameters of self-attention guidance.
        :return: 
                 - 'sample': a random sample from the model.
                 - 'pred_x_0': a prediction of x_0.
        """
        s, t = step.div(self.sample_timesteps), \
               step.add(1).div(self.sample_timesteps)
        cond = broadcast_to(step > 0, x_t, dtype=torch.bool)
        model_mean, model_logvar, pred_x_0 = self.p_mean_var_enc(
            denoise_fn, x_t, s, t, y, x_start, 
            clip_denoised=clip_denoised, return_pred=True, use_ddim=use_ddim)
        # if t[0]!=1:
        #     delta_out = out - self.pre_out
        #     delta_x_0 = pred_x_0 - self.pre_x_0
        # self.pre_out = out
        # self.pre_x_0 = pred_x_0
        to_range_0_1 = lambda x: (x + 1.) / 2.
        save_image_cuda(pred_x_0, "1.png")
        # model_mean = torch.where(cond, model_mean, pred_x_0)
        if self.w_guide and y is not None:
            # classifier-free guidance
            _model_mean, _, _pred_x_0 = self.p_mean_var(
                denoise_fn, x_t, s, t, torch.zeros_like(y),
                clip_denoised=clip_denoised, return_pred=True, use_ddim=use_ddim)
            _model_mean = torch.where(cond, _model_mean, _pred_x_0)
            model_mean += self.w_guide * (model_mean - _model_mean)

        noise = torch.randn_like(x_t)
        sample = model_mean + cond.float() * torch.exp(0.5 * model_logvar) * noise

        return (sample, pred_x_0) if return_pred else sample   #, pred_eps

    def p_sample_distill_step(
            self, denoise_fn, x_t, step, y, x_0,
            clip_denoised=True, return_pred=True, use_ddim=False):
        """
        Sample x_{t-1} from the SFERD model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param attn_guidance: if True, self-attention guidance is given.
        :param guidance_kwargs: if not None, a dict of the parameters of self-attention guidance.
        :return: 
                 - 'sample': a random sample from the model.
                 - 'pred_x_0': a prediction of x_0.
        """
        s, t = step.sub(1).div(self.sample_timesteps), \
               step.div(self.sample_timesteps)
        cond = broadcast_to(step > 0, x_t, dtype=torch.bool)
        model_mean, model_logvar, pred_x_0 = self.p_mean_var_enc(
            denoise_fn, x_t, s, t, y, x_0, 
            clip_denoised=clip_denoised, return_pred=True, use_ddim=use_ddim)
        to_range_0_1 = lambda x: (x + 1.) / 2.
        # save_image(to_range_0_1(pred_x_0), "1.png", )
        # model_mean = torch.where(cond, model_mean, pred_x_0)
        if self.w_guide and y is not None:
            # classifier-free guidance
            _model_mean, _, _pred_x_0 = self.p_mean_var(
                denoise_fn, x_t, s, t, torch.zeros_like(y),
                clip_denoised=clip_denoised, return_pred=True, use_ddim=use_ddim)
            _model_mean = torch.where(cond, _model_mean, _pred_x_0)
            model_mean += self.w_guide * (model_mean - _model_mean)

        noise = torch.randn_like(x_t)
        sample = model_mean + cond.float() * torch.exp(0.5 * model_logvar) * noise

        return (sample, pred_x_0) if return_pred else sample

    @torch.inference_mode()
    def p_sample(
            self, denoise_fn, shape, x_start=None, 
            noise=None, label=None, device="cpu", use_ddim=False):
        B = shape[0]
        t = torch.empty((B,), device=device)
        if noise is None:
            x_t = torch.randn(shape, device=device)
        else:
            x_t = noise.to(device)
        if label is not None:
            label = label.to(device)
        for ti in reversed(range(self.sample_timesteps)):
            t.fill_(ti)
            x_t = self.p_sample_step(
                denoise_fn, x_t, step=t, y=label, x_start=x_start, use_ddim=use_ddim)
        return x_t.cpu()

    @torch.inference_mode()
    def p_sample_progressive(
            self, denoise_fn, shape,
            noise=None, label=None, device="cpu", use_ddim=False, pred_freq=50):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        B = shape[0]
        t = torch.empty(B, device=device)
        if noise is None:
            x_t = torch.randn(shape, device=device)
        else:
            x_t = noise.to(device)
        L = self.sample_timesteps // pred_freq
        preds = torch.zeros((L, B) + shape[1:], dtype=torch.float32)
        idx = L
        for ti in reversed(range(self.sample_timesteps)):
            t.fill_(ti)
            x_t, pred = self.p_sample_step(
                denoise_fn, x_t, step=t, y=label, return_pred=True, use_ddim=use_ddim)
            if (ti + 1) % pred_freq == 0:
                idx -= 1
                preds[idx] = pred.cpu()
        return x_t.cpu(), preds

    # === log likelihood ===
    # bpd: bits per dimension

    def _loss_term_bpd(
            self, denoise_fn, x_0, x_t, s, t, y, clip_denoised, return_pred):
        logsnr_s, logsnr_t = self.t2logsnr(s, t, x=x_0)
        # calculate L_t
        # t = 0: negative log likelihood of decoder, -\log p(x_0 | x_1)
        # t > 0: variational lower bound loss term, KL term
        true_mean, true_logvar = self.q_posterior_mean_var(
            x_0=x_0, x_t=x_t,
            logsnr_s=logsnr_s, logsnr_t=logsnr_t, model_var_type="fixed_small")
        model_mean, model_logvar, pred_x_0 = self.p_mean_var_enc(
            denoise_fn, x_t=x_t, s=s, t=t, y=y, x_0=x_0,
            clip_denoised=clip_denoised, return_pred=True, use_ddim=False)
        kl = normal_kl(true_mean, true_logvar, model_mean, model_logvar)
        kl = flat_mean(kl) / math.log(2.)  # natural base to base 2
        decoder_nll = discretized_gaussian_loglik(
            x_0, pred_x_0, log_scale=0.5 * model_logvar).neg()
        decoder_nll = flat_mean(decoder_nll) / math.log(2.)
        output = torch.where(s.to(kl.device) > 0, kl, decoder_nll)
        return (output, pred_x_0) if return_pred else output

    def from_model_out_to_pred(self, x_t, model_out, logsnr_t):
        assert self.model_out_type in {"x0", "eps", "both", "v"}
        if self.model_out_type == "v":
            v = model_out
            x_0 = pred_x0_from_v(x_t, v, logsnr_t)
            eps = pred_eps_from_v(x_t, v, logsnr_t)
        else:
            if self.model_out_type == "x0":
                x_0 = model_out
                eps = pred_eps_from_x0(x_t, x_0, logsnr_t)
            elif self.model_out_type == "eps":
                eps = model_out
                x_0 = pred_x0_from_eps(x_t, eps, logsnr_t)
            elif self.model_out_type == "both":
                x_0, eps = model_out.chunk(2, dim=1)
            else:
                raise NotImplementedError(self.model_out_type)
            v = pred_v_from_x0eps(x_0, eps, logsnr_t)
        return {"constant": x_0, "snr": eps, "truncated_snr": (x_0, eps), "alpha2": v}

    '''
        training distillation model without gradient predictor
    '''
    def distill_losses(self, student_diffusion, teacher_denoise_fn, student_denoise_fn, 
                x_0, t, y, speed_up, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        with torch.no_grad():
            logsnr_t, = self.t2logsnr(t, x=x_0)
            x_t = q_sample(x_0, logsnr_t, eps=noise)
            t = t.mul(self.sample_timesteps)
            # save_image_cuda(x_t, "result_xt.png", nrow=8, normalize=True, value_range=(-1., 1.))
            x_tm1 = self.p_sample_distill_step(teacher_denoise_fn, x_t, step=t, y=None, 
                                speed_up=speed_up, clip_denoised=True, return_pred=False, use_ddim=True)
            # save_image_cuda(x_tm1, "result_xtm1.png", nrow=8, normalize=True, value_range=(-1., 1.))
            tm1 = t.sub(1).div(self.sample_timesteps)
            pred_x_0 = self.p_pred_x_0(teacher_denoise_fn, x_tm1, tm1, y, clip_denoised=True)
            save_image_cuda(pred_x_0, "result_pred_x.png", nrow=8, normalize=True, value_range=(-1., 1.))
            w = 1 + torch.sigmoid(logsnr_t)/torch.sigmoid(-logsnr_t)
            # w = torch.sigmoid(logsnr_t)/torch.sigmoid(-logsnr_t).sqrt()
        # calculate the loss
        # mse: re-weighted
        if self.loss_type == "mse":
            if self.p_uncond and y is not None:
                y *= broadcast_to(
                    torch.rand((y.shape[0],)) > self.p_uncond, y)
            t = t.div(self.sample_timesteps)
            model_out = student_diffusion.p_pred_x_0(student_denoise_fn, x_t, t, y=y, clip_denoised=True)
            save_image_cuda(model_out, "result.png", nrow=8, normalize=True, value_range=(-1., 1.))
            losses = flat_mean((w * pred_x_0 - w * model_out).pow(2))
        else:
            raise NotImplementedError(self.loss_type)
        return losses
    def distill_losses_enc(self, student_diffusion, teacher_denoise_fn, student_denoise_fn, 
                x_0, t, y, speed_up, noise=None):
    """
        Compute training losses of SFERD model for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
    """
        if noise is None:
            noise = torch.randn_like(x_0)
        with torch.no_grad():
            logsnr_t, = self.t2logsnr(t, x=x_0)
            shape =x_0.shape
            x_t = q_sample(x_0, logsnr_t, eps=noise)
            s = None
            if self.loss_type == "kl":
                t = torch.ceil(t * self.sample_timesteps)
                s = t.sub(1).div(self.sample_timesteps)
                t = t.div(self.sample_timesteps)
            tm1 = t.sub(1/self.sample_timesteps)
            logsnr_s, = self.t2logsnr(tm1, x=x_0)
            x_tm1 = q_sample(x_0, logsnr_s, eps=noise)
            # pred_x_0 = self.p_pred_x_0(teacher_denoise_fn, x_tm1, tm1, y, clip_denoised=True)
            pred_x_0, attn_map, pred_eps = self.p_pred_x_0_enc(teacher_denoise_fn, x_tm1, tm1, y=None, x_start=x_0, clip_denoised=True)
            save_image_cuda(pred_x_0, "result_pred_x_1.png", nrow=8, normalize=True, value_range=(-1., 1.))
            blur_sigma = self.blur_sigma
            '''
            Apply the self-attention mask to produce mask_blurred bar{x_0^t}
            '''
            mask_blurred = self.attention_masking(
                    pred_x_0,
                    tm1,
                    attn_map,
                    # prev_noise=pred_eps,
                    x_real=x_0,
                    blur_sigma=blur_sigma,
                )
            '''
            apply forward-diffusion to obtain \bm{\tilde x}_{t}^{attn}
            '''
            mask_blurred = q_sample(mask_blurred, logsnr_s, eps=pred_eps)
            _, _, uncond_eps = self.p_pred_x_0_enc(teacher_denoise_fn, mask_blurred, tm1, None, pred_x_0, clip_denoised=True)
            guided_eps = uncond_eps + self.w_guide * (pred_eps - uncond_eps)
            '''
            get \bm {\tilde x}_0^{\mathrm {target}} 
            '''
            pred_x_0_target = pred_x0_from_eps(x_tm1, guided_eps, logsnr_s)
            # save_image_cuda(pred_x_0_target, "result_pred_x_2.png", nrow=8, normalize=True, value_range=(-1., 1.))
            # pred_x_0 += self.w_guide * (pred_x_0 - pred_x_0_attn)
            # save_image_cuda(pred_x_0_attn, "result_pred_x_3.png", nrow=8, normalize=True, value_range=(-1., 1.))
            w = 1 + torch.sigmoid(logsnr_t)/torch.sigmoid(-logsnr_t)

            # w = torch.sigmoid(logsnr_t)/torch.sigmoid(-logsnr_t).sqrt()
        # calculate the loss
        # mse: re-weighted
        if self.loss_type == "mse":
            if self.p_uncond and y is not None:
                y *= broadcast_to(
                    torch.rand((y.shape[0],)) > self.p_uncond, y)
            # t = t.div(self.sample_timesteps)
            pred_x_0, gradient, pred_eps = student_diffusion.p_pred_x_0_enc_student(student_denoise_fn, x_t, t, x_start=x_0, clip_denoised=True)
            shift_coef = torch.sigmoid(-logsnr_s)/torch.sigmoid(logsnr_s).sqrt()
            shift_coef = self.extract_coef_at_t(shift_coef, t, shape)
            # weight = None
            # the re-written losses of the pre-trained student model with gradient predictor
            losses = flat_mean((w * pred_x_0_target - w * (pred_x_0 + shift_coef * gradient)).pow(2))
        elif self.loss_type == "kl":
            losses = self._loss_term_bpd(
                student_denoise_fn, x_0=x_0, x_t=x_t, s=s, t=t, y=y,
                clip_denoised=False, return_pred=False)
        else:
            raise NotImplementedError(self.loss_type)
        return losses
    def train_losses(self, denoise_fn, x_0, t, y, noise=None):
    """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
    """
        if noise is None:
            noise = torch.randn_like(x_0)

        s = None
        if self.loss_type == "kl":
            t = torch.ceil(t * self.sample_timesteps)
            s = t.sub(1).div(self.sample_timesteps)
            t = t.div(self.sample_timesteps)

        # calculate the loss
        # kl: un-weighted
        # mse: re-weighted

        logsnr_t, = self.t2logsnr(t, x=x_0)
        x_t = q_sample(x_0, logsnr_t, eps=noise)
        if self.loss_type == "kl":
            losses = self._loss_term_bpd(
                denoise_fn, x_0=x_0, x_t=x_t, s=s, t=t, y=y,
                clip_denoised=False, return_pred=False)
        elif self.loss_type == "mse":
            assert self.model_var_type != "learned"
            assert self.reweight_type in {"constant", "snr", "truncated_snr", "alpha2"}
            target = {
                "constant": x_0,
                "snr": noise,
                "truncated_snr": (x_0, noise),
                "alpha2": pred_v_from_x0eps(x_0, noise, logsnr_t)
            }[self.reweight_type]
            
            if self.p_uncond and y is not None:
                y *= broadcast_to(
                    torch.rand((y.shape[0],)) > self.p_uncond, y)

            model_out = denoise_fn(x_t, t, y=y, x_start=x_0).pred
            predict = self.from_model_out_to_pred(
                x_t, model_out, logsnr_t
            )[self.reweight_type]

            if isinstance(target, tuple):
                assert len(target) == 2
                losses = torch.maximum(*[
                    flat_mean((tgt - pred).pow(2))
                    for tgt, pred in zip(target, predict)])
            else:
                losses = flat_mean((target - model_out).pow(2))
        else:
            raise NotImplementedError(self.loss_type)

        return losses

    def _prior_bpd(self, x_0):
        B = x_0.shape[0]
        t = torch.ones([B, ], dtype=torch.float32)
        logsnr_t, = self.t2logsnr(t, x=x_0)
        T_mean, T_logvar = q_mean_var(x_0=x_0, logsnr_t=logsnr_t)
        kl_prior = normal_kl(T_mean, T_logvar, mean2=0., logvar2=0.)
        return flat_mean(kl_prior) / math.log(2.)

    def calc_all_bpd(self, denoise_fn, x_0, y, clip_denoised=True):
        B, T = x_0.shape, self.sample_timesteps
        s = torch.empty([B, ], dtype=torch.float32)
        t = torch.empty([B, ], dtype=torch.float32)
        losses = torch.zeros([B, T], dtype=torch.float32)
        mses = torch.zeros([B, T], dtype=torch.float32)

        for i in range(T - 1, -1, -1):
            s.fill_(i / self.sample_timesteps)
            t.fill_((i + 1) / self.sample_timesteps)
            logsnr_t, = self.t2logsnr(t)
            x_t = q_sample(x_0, logsnr_t=logsnr_t)
            loss, pred_x_0 = self._loss_term_bpd(
                denoise_fn, x_0, x_t=x_t, s=s, t=t, y=y,
                clip_denoised=clip_denoised, return_pred=True)
            losses[:, i] = loss
            mses[:, i] = flat_mean((pred_x_0 - x_0).pow(2))

        prior_bpd = self._prior_bpd(x_0)
        total_bpd = torch.sum(losses, dim=1) + prior_bpd
        return total_bpd, losses, prior_bpd, mses


    """
        representation learning
    """

    def representation_learning_ddpm_sample(self, encoder, decoder, x_0, x_T, z=None):
        shape = x_0.shape
        batch_size = shape[0]

        if z is None:
            z = encoder(x_0)
        img = x_T

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            predicted_noise, gradient = decoder(img, t, z)
            shift_coef = self.extract_coef_at_t(self.shift_coef, t, shape)
            img = self.noise_p_sample(img, t, predicted_noise + shift_coef * gradient)
        return img


    def representation_learning_ddim_sample(self, ddim_style, encoder, decoder, x_0, x_T, z=None, stop_percent=0.0):
        if z is None:
            z = encoder(x_0)
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.shift_ddim_sample_loop(decoder, z, x_T, stop_percent=stop_percent)

    def representation_learning_ddim_encode(self, ddim_style, encoder, decoder, x_0, z=None):
        if z is None:
            z = encoder(x_0)
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.shift_ddim_encode_loop(decoder, z, x_0)

    def representation_learning_autoencoding(self, encoder_ddim_style, decoder_ddim_style, encoder, decoder, x_0):
        z = encoder(x_0)
        inferred_x_T = self.representation_learning_ddim_encode(encoder_ddim_style, encoder, decoder, x_0, z)
        return self.representation_learning_ddim_sample(decoder_ddim_style, None, decoder, None, inferred_x_T, z)

    
    def representation_learning_denoise_one_step(self, encoder, decoder, x_0, timestep_list):
        shape = x_0.shape

        t = torch.tensor(timestep_list, device=self.device, dtype=torch.long)
        x_t = self.q_sample(x_0, t, noise=torch.randn_like(x_0))
        z = encoder(x_0)
        predicted_noise, gradient = decoder(x_t, t, z)

        predicted_x_0 = self.predicted_noise_to_predicted_x_0(x_t, t, predicted_noise)

        shift_coef = self.extract_coef_at_t(self.shift_coef, t, shape)
        autoencoder_predicted_noise = predicted_noise + shift_coef * gradient
        autoencoder_predicted_x_0 = self.predicted_noise_to_predicted_x_0(x_t, t, autoencoder_predicted_noise)

        return predicted_x_0, autoencoder_predicted_x_0
    """
        z_latent
    """
    @property
    def latent_diffusion_config(self):
        timesteps = 1000
        betas = np.array([0.008] * timesteps)
        # betas = np.linspace(0.0001, 0.02, timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)
        loss_type = "l1"

        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)
        return {
            "timesteps": timesteps,
            "betas": betas,
            "alphas_cumprod": to_torch(alphas_cumprod),
            "sqrt_alphas_cumprod": to_torch(sqrt_alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": to_torch(sqrt_one_minus_alphas_cumprod),
            "loss_type": loss_type,
        }

    def normalize(self, z, mean, std):
        return (z - mean) / std


    def denormalize(self, z, mean, std):
        return z * std + mean


    def latent_diffusion_train_one_batch(self, latent_denoise_fn, encoder, x_0, latents_mean, latents_std):
        timesteps = self.latent_diffusion_config["timesteps"]

        sqrt_alphas_cumprod = self.latent_diffusion_config["sqrt_alphas_cumprod"]
        sqrt_one_minus_alphas_cumprod = self.latent_diffusion_config["sqrt_one_minus_alphas_cumprod"]

        z_0 = encoder(x_0)
        z_0 = z_0.detach()
        z_0 = self.normalize(z_0, latents_mean, latents_std)

        shape = z_0.shape
        batch_size = shape[0]

        t = torch.randint(0, timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(z_0)

        z_t = self.extract_coef_at_t(sqrt_alphas_cumprod, t, shape) * z_0 \
                + self.extract_coef_at_t(sqrt_one_minus_alphas_cumprod, t, shape) * noise

        predicted_noise = latent_denoise_fn(z_t, t)

        prediction_loss = self.p_loss(noise, predicted_noise, loss_type=self.latent_diffusion_config["loss_type"])

        return {
            'prediction_loss': prediction_loss,
        }

    def latent_diffusion_sample(self, latent_ddim_style, decoder_ddim_style, latent_denoise_fn, decoder, x_T, latents_mean, latents_std):
        alphas_cumprod = self.latent_diffusion_config["alphas_cumprod"]

        batch_size = x_T.shape[0]
        input_channel = latent_denoise_fn.module.input_channel
        z_T = torch.randn((batch_size, input_channel), device=self.device)

        z_T.clamp_(-1.0, 1.0) # may slightly improve sample quality

        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(latent_ddim_style, alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        z = ddim.latent_ddim_sample_loop(latent_denoise_fn, z_T)

        z = self.denormalize(z, latents_mean, latents_std)

        return self.representation_learning_ddim_sample(decoder_ddim_style, None, decoder, None, x_T, z, stop_percent=0.3)


if __name__ == "__main__":
    DEBUG = True


    def test_logsnr_to_posterior():
        logsnr_schedule = get_logsnr_schedule("cosine")
        logsnr_s = logsnr_schedule(torch.as_tensor(0.))
        logsnr_t = logsnr_schedule(torch.as_tensor(1. / 1000))
        print(logsnr_to_posterior(logsnr_s, logsnr_t, "fixed_small"))
        logsnr_s = logsnr_schedule(torch.as_tensor(999. / 1000))
        logsnr_t = logsnr_schedule(torch.as_tensor(1.))
        print(logsnr_to_posterior(logsnr_s, logsnr_t, "fixed_small"))


    def test_logsnr_to_posterior_ddim():
        logsnr_schedule = get_logsnr_schedule("cosine")
        t = torch.linspace(0, 1, 1001, dtype=torch.float32)
        print(logsnr_schedule(t))
        logsnr_s = logsnr_schedule(t[:-1])
        logsnr_t = logsnr_schedule(t[1:])
        mean_coef1, mean_coef2, logvar = logsnr_to_posterior(
            logsnr_s, logsnr_t, "fixed_small")
        mean_coef1_, mean_coef2_, logvar_ = logsnr_to_posterior_ddim(
            logsnr_s, logsnr_t, eta=1.)
        print(
            torch.allclose(mean_coef1, mean_coef1_),
            torch.allclose(mean_coef2, mean_coef2_),
            torch.allclose(logvar, logvar_))


    def test_legacy():
        logsnr_schedule = get_logsnr_schedule("legacy")
        t = torch.linspace(0, 1, 1000, dtype=torch.float32)
        print(torch.sigmoid(logsnr_schedule(t))[::10])
        print(logsnr_schedule(t)[::10])
        t = torch.rand(10000, dtype=torch.float32)
        print(logsnr_schedule(t))

    # run tests
    TESTS = [test_logsnr_to_posterior, test_logsnr_to_posterior_ddim, test_legacy]
    TEST_INDICES = []
    for i in TEST_INDICES:
        TESTS[i]()
    
import torchvision.utils as tu
from torchvision.utils import make_grid
def save_image_cuda(x, path, nrow=8, normalize=True, value_range=(-1., 1.)):
    img = make_grid(x, nrow=nrow, normalize=normalize, value_range=value_range)
    tu.save_image(img, path)

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple


DEFAULT_DTYPE = torch.float32


@torch.jit.script
def get_timestep_embedding(
        timesteps, embed_dim: int, dtype: torch.dtype = DEFAULT_DTYPE, scale: float = 1000.):
    """
    Adapted from fairseq/fairseq/modules/sinusoidal_positional_embedding.py
    The implementation is slightly different from the decription in Section 3.5 of [1]
    [1] Vaswani, Ashish, et al. "Attention is all you need."
     Advances in neural information processing systems 30 (2017).
    """
    timesteps = scale * timesteps.ravel()
    half_dim = embed_dim // 2
    embed = math.log(10000) / (half_dim - 1)
    embed = torch.exp(-torch.arange(half_dim, dtype=dtype, device=timesteps.device) * embed)
    embed = torch.outer(timesteps.to(dtype), embed)
    embed = torch.cat([torch.sin(embed), torch.cos(embed)], dim=1)
    if embed_dim % 2 == 1:
        embed = F.pad(embed, [0, 1])  # padding the last dimension
    assert embed.dtype == dtype
    return embed


@torch.jit.script
def normal_kl(mean1, logvar1, mean2, logvar2):
    diff_logvar = logvar1 - logvar2
    kl = (-1.0 - diff_logvar).add(
        (mean1 - mean2).pow(2) * torch.exp(-logvar2)).add(
        torch.exp(diff_logvar)).mul(0.5)
    return kl


@torch.jit.script
def approx_std_normal_cdf(x):
    """
    Reference:
    Page, E. “Approximations to the Cumulative Normal Function and Its Inverse for Use on a Pocket Calculator.”
     Applied Statistics 26.1 (1977): 75–76. Web.
    """
    return 0.5 * (1. + torch.tanh(math.sqrt(2. / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


@torch.jit.script
def discretized_gaussian_loglik(
        x, means, log_scale, precision: float = 1./255,
        tol: float = 1e-12):
    # if isinstance(cutoff, float):
    #     cutoff = (-cutoff, cutoff)
    # Assumes data is integers [0, 255] rescaled to [-1, 1]
    cutoff = (-0.999, 0.999)
    x_centered = x - means
    inv_stdv = torch.exp(-log_scale)
    upper = inv_stdv * (x_centered + precision)
    cdf_upper = torch.where(
        x > cutoff[1], torch.as_tensor(1, dtype=torch.float32, device=x.device), approx_std_normal_cdf(upper))
    lower = inv_stdv * (x_centered - precision)
    cdf_lower = torch.where(
        x < cutoff[0], torch.as_tensor(0, dtype=torch.float32, device=x.device), approx_std_normal_cdf(lower))
    log_probs = torch.log(torch.clamp(cdf_upper - cdf_lower - tol, min=0).add(tol))
    return log_probs


@torch.jit.script
def continuous_gaussian_loglik(x, mean, logvar):
    x_centered = x - mean
    inv_var = torch.exp(-logvar)
    log_probs = x_centered.pow(2) * inv_var + math.log(2 * math.pi) + logvar
    return log_probs.mul(0.5).neg()


def discrete_klv2d(hist1, hist2, eps=1e-9):
    """
    compute the discretized (empirical) Kullback-Leibler divergence between P_data1 and P_data2
    """
    return np.sum(hist2 * (np.log(hist2 + eps) - np.log(hist1 + eps)))


def hist2d(data, bins, value_range=None):
    """
    compute the 2d histogram matrix for a set of data points
    """
    if bins == "auto":
        bins = math.floor(math.sqrt(len(data) // 10))
    if value_range is not None:
        if isinstance(value_range, (int, float)):
            value_range = ((-value_range, value_range), ) * 2
        if hasattr(value_range, "__iter__"):
            if not hasattr(next(iter(value_range)), "__iter__"):
                value_range = (value_range, ) * 2
    x, y = np.split(data, 2, axis=1)
    x, y = x.squeeze(1), y.squeeze(1)
    return np.histogram2d(x, y, bins=bins, range=value_range)[0]


def flat_mean(x, start_dim=1):
    reduce_dim = [i for i in range(start_dim, x.ndim)]
    return torch.mean(x, dim=reduce_dim)


def flat_sum(x, start_dim=1):
    reduce_dim = [i for i in range(start_dim, x.ndim)]
    return torch.sum(x, dim=reduce_dim)


