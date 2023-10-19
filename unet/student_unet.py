import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from abc import abstractmethod

class View(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, input):
        return input.view(self.size)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return nn.GroupNorm(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :]  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings (and class) as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepZBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings (and class) as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, emb_z):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, emb_z=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, TimestepZBlock):
                x = layer(x, emb, emb_z)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels)
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # Adaptive Group Normalization
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1. + scale) + shift
        h = out_rest(h)

        return self.skip_connection(x) + h

class ResBlockShift(TimestepZBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels),
        )
        self.emb_z_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, emb_z):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        emb_z_out = self.emb_z_layers(emb_z)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        while len(emb_z_out.shape) < len(h.shape):
            emb_z_out = emb_z_out[..., None]

        # Adaptive Group Normalization
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        z_scale, z_shift = torch.chunk(emb_z_out, 2, dim=1)
        h = (1. + z_scale) * (out_norm(h) * (1. + scale) + shift) + z_shift
        h = out_rest(h)

        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )
        weight = torch.softmax(weight, dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight, dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

class UNet(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param input_channel: channels in the input Tensor.
    :param base_channel: base channel count for the model.
    :param num_residual_blocks_of_a_block: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_multiplier: channel multiplier for each level of the UNet.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_class: if specified (as an int), then this model will be
        class-conditional with `num_class` classes.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        input_channel,
        base_channel,
        channel_multiplier,
        num_residual_blocks_of_a_block,
        attention_resolutions,
        num_heads,
        head_channel,
        use_new_attention_order,
        dropout,
        num_class=None,
        dims=2,
        learn_sigma=False,
        **kwargs
    ):
        super().__init__()
        self.num_class = num_class
        self.base_channel = base_channel
        output_channel = input_channel * 2 if learn_sigma else input_channel
        time_embed_dim = base_channel * 4
        self.time_embed = nn.Sequential(
            linear(base_channel, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_class is not None:
            # original class label
            # self.label_emb = nn.Embedding(num_class, time_embed_dim)

            # free
            # self.label_emb = nn.Linear(num_class, time_embed_dim)

            self.label_emb = nn.Sequential(
                nn.Linear(num_class, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        ch = input_ch = int(channel_multiplier[0] * base_channel)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, input_channel, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_multiplier):
            for _ in range(num_residual_blocks_of_a_block):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * base_channel),
                        dims=dims
                    )
                ]
                ch = int(mult * base_channel)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=head_channel,
                            use_new_attention_order=use_new_attention_order
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_multiplier) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            down=True
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=head_channel,
                use_new_attention_order=use_new_attention_order
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_multiplier))[::-1]:
            for i in range(num_residual_blocks_of_a_block + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(base_channel * mult),
                        dims=dims
                    )
                ]
                ch = int(base_channel * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=head_channel,
                            use_new_attention_order=use_new_attention_order
                        )
                    )
                if level and i == num_residual_blocks_of_a_block:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            up=True
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, output_channel, 3, padding=1)),
        )

    def forward(self, input, time, condition=None):
        """
        Apply the model to an input batch.

        :param input: an [N x C x ...] Tensor of inputs.
        :param time: a 1-D batch of timesteps.
        :param condition: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        emb = self.time_embed(timestep_embedding(time, self.base_channel))

        if self.num_class is not None:
            # assert condition.shape == (input.shape[0],)
            emb = emb + self.label_emb(condition)

        h = input
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)

class Base_model(nn.Module):
    def __init__(
        self,
        semantic_encoder,
        gradient_predictor,
    ):
        super().__init__()
        self.semantic_encoder = base_channel
        self.gradient_predictor = gradient_predictor

class ShiftUNet(nn.Module):
    """
    ShiftUNet based on UNet with additive trainable label_emb, shift_middle_block, shift_output_blocks and shift_out.

    :param input_channel: channels in the input Tensor.
    :param base_channel: base channel count for the model.
    :param num_residual_blocks_of_a_block: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_multiplier: channel multiplier for each level of the UNet.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param latent_dim: latent dim
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        input_channel,
        base_channel,
        channel_multiplier,
        num_residual_blocks_of_a_block,
        attention_resolutions,
        num_heads,
        head_channel,
        use_new_attention_order,
        dropout,
        latent_dim,
        dims=2,
        learn_sigma=False,
        **kwargs
    ):
        super().__init__()
        self.base_channel = base_channel
        output_channel = input_channel * 2 if learn_sigma else input_channel

        time_embed_dim = base_channel * 4

        # this layer is freeze, which is trained in diffusion model
        self.time_embed = nn.Sequential(
            linear(base_channel, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # original class label
        # self.label_emb = nn.Embedding(latent_dim, time_embed_dim)

        # free representation learning
        # this layer is trainable
        self.label_emb = nn.Linear(latent_dim, time_embed_dim)

        ch = input_ch = int(channel_multiplier[0] * base_channel)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, input_channel, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        shift_input_block_chans = [ch]
        ds = 1
        shift_ds = 1
        for level, mult in enumerate(channel_multiplier):
            for _ in range(num_residual_blocks_of_a_block):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * base_channel),
                        dims=dims
                    )
                ]
                ch = int(mult * base_channel)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=head_channel,
                            use_new_attention_order=use_new_attention_order
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
                shift_input_block_chans.append(ch)
            if level != len(channel_multiplier) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            down=True
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                shift_input_block_chans.append(ch)
                ds *= 2
                shift_ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=head_channel,
                use_new_attention_order=use_new_attention_order
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims
            ),
        )

        self.shift_middle_block = TimestepEmbedSequential(
            ResBlockShift(
                ch,
                time_embed_dim,
                dropout,
                dims=dims
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=head_channel,
                use_new_attention_order=use_new_attention_order
            ),
            ResBlockShift(
                ch,
                time_embed_dim,
                dropout,
                dims=dims
            ),
        )

        memory_ch = ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_multiplier))[::-1]:
            for i in range(num_residual_blocks_of_a_block + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(base_channel * mult),
                        dims=dims
                    )
                ]
                ch = int(base_channel * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=head_channel,
                            use_new_attention_order=use_new_attention_order
                        )
                    )
                if level and i == num_residual_blocks_of_a_block:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            up=True
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        ch = memory_ch
        self.shift_output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_multiplier))[::-1]:
            for i in range(num_residual_blocks_of_a_block + 1):
                ich = shift_input_block_chans.pop()
                layers = [
                    ResBlockShift(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(base_channel * mult),
                        dims=dims
                    )
                ]
                ch = int(base_channel * mult)
                if shift_ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=head_channel,
                            use_new_attention_order=use_new_attention_order
                        )
                    )
                if level and i == num_residual_blocks_of_a_block:
                    out_ch = ch
                    layers.append(
                        ResBlockShift(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            up=True
                        )
                    )
                    shift_ds //= 2
                self.shift_output_blocks.append(TimestepEmbedSequential(*layers))


        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, output_channel, 3, padding=1)),
        )

        self.shift_out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, input_channel, 3, padding=1)),
        )

        self.freeze()

    def forward(self, input, time, condition):
        """
        Apply the model to an input batch.

        :param input: an [N x C x ...] Tensor of inputs.
        :param time: a 1-D batch of timesteps.
        :param condition: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        emb = self.time_embed(timestep_embedding(time, self.base_channel))
        shift_emb = self.label_emb(condition)

        hs = []
        h = input
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        epsilon_h = self.middle_block(h, emb)
        shift_h = self.shift_middle_block(h, emb, shift_emb)

        for module, shift_module in zip(self.output_blocks, self.shift_output_blocks):
            h_previous = hs.pop()

            epsilon_h = torch.cat([epsilon_h, h_previous], dim=1)
            epsilon_h = module(epsilon_h, emb)

            shift_h = torch.cat([shift_h, h_previous], dim=1)
            shift_h = shift_module(shift_h, emb, shift_emb)

        return self.out(epsilon_h), self.shift_out(shift_h)


    def set_train_mode(self):
        self.label_emb.train()
        self.shift_middle_block.train()
        self.shift_output_blocks.train()
        self.shift_out.train()

    def set_eval_mode(self):
        self.label_emb.eval()
        self.shift_middle_block.eval()
        self.shift_output_blocks.eval()
        self.shift_out.eval()

    def freeze(self):
        self.time_embed.eval()
        self.input_blocks.eval()
        self.middle_block.eval()
        self.output_blocks.eval()
        self.out.eval()

        self.time_embed.requires_grad_(requires_grad=False)
        self.input_blocks.requires_grad_(requires_grad=False)
        self.middle_block.requires_grad_(requires_grad=False)
        self.output_blocks.requires_grad_(requires_grad=False)
        self.out.requires_grad_(requires_grad=False)


class MLPSkipNet(nn.Module):
    """
    concat x to hidden layers

    default MLP for the latent DPM in the paper!
    """
    def __init__(
        self,
        input_channel, # latent_channel
        model_channel,
        num_layers,
        time_emb_channel,
        use_norm,
        dropout,
        **kwargs
    ):
        super().__init__()

        self.input_channel = input_channel
        self.skip_layers = list(range(1,num_layers))
        self.time_emb_channel = time_emb_channel

        self.time_embed = nn.Sequential(
            nn.Linear(self.time_emb_channel, input_channel),
            nn.SiLU(),
            nn.Linear(input_channel, input_channel),
        )

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            if i == 0:
                act = "silu"
                norm = use_norm
                cond = True
                a, b = input_channel, model_channel
                dropout = dropout
            elif i == num_layers - 1:
                act = "none"
                norm = False
                cond = False
                a, b = model_channel, input_channel
                dropout = 0
            else:
                act = "silu"
                norm = use_norm
                cond = True
                a, b = model_channel, model_channel
                dropout = dropout

            if i in self.skip_layers:
                a += input_channel

            self.layers.append(
                MLPLNAct(
                    a,
                    b,
                    norm=norm,
                    activation=act,
                    cond_channels=input_channel,
                    use_cond=cond,
                    dropout=dropout,
                ))

    def forward(self, x, t):
        t = timestep_embedding(t, self.time_emb_channel)
        cond = self.time_embed(t)
        h = x
        for i in range(len(self.layers)):
            if i in self.skip_layers:
                # injecting input into the hidden layers
                h = torch.cat([h, x], dim=1)
            h = self.layers[i].forward(x=h, cond=cond)
        output = h
        return output


class MLPLNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm,
        use_cond,
        activation,
        cond_channels,
        dropout,
    ):
        super().__init__()
        self.activation = activation
        self.use_cond = use_cond

        self.linear = nn.Linear(in_channels, out_channels)
        self.act = nn.SiLU() if self.activation == "silu" else nn.Identity()
        if self.use_cond:
            self.linear_emb = nn.Linear(cond_channels, out_channels)
            self.cond_layers = nn.Sequential(self.act, self.linear_emb)
        if norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == "silu":
                    nn.init.kaiming_normal_(module.weight, a=0, nonlinearity='relu')
                else:
                    # leave it as default
                    pass

    def forward(self, x, cond=None):
        x = self.linear(x)
        if self.use_cond:
            # (n, c) or (n, c * 2)
            cond = self.cond_layers(cond)
            cond = (cond, None)

            # scale shift first
            x = x * (1.0 + cond[0])
            if cond[1] is not None:
                x = x + cond[1]
            # then norm
            x = self.norm(x)
        else:
            # no condition
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x