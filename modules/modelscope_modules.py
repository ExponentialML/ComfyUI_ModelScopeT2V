import torch
import torch.nn as nn
import comfy.ops
from einops import rearrange, repeat
from comfy.ldm.modules.attention import SpatialTransformer, BasicTransformerBlock
from comfy.ldm.modules.diffusionmodules.openaimodel import TimestepBlock, Downsample, Upsample

operations = comfy.ops.disable_weight_init
ops = operations

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def get_batch_size(b: int, use_cfg: True):
    return 2 if use_cfg else 1

def use_attn_strength(strength):
    return 1.0 - strength > 0.0

class TemporalConvBlock_v2(nn.Module):

    def __init__(self,
                    in_dim,
                    out_dim=None,
                    dropout=0.0,
                    device='cuda',
                    dtype=torch.float16,
                    ops=operations,
                    temporal_conv_strength=1.0
                 ):
        super(TemporalConvBlock_v2, self).__init__()
        if out_dim is None:
            out_dim = in_dim  # int(1.5*in_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.temporal_conv_strength = temporal_conv_strength

        # conv layers
        self.conv1 = nn.Sequential(
            ops.GroupNorm(32, in_dim, dtype=dtype, device=device), nn.SiLU(),
            ops.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0), dtype=dtype, device=device))
        self.conv2 = nn.Sequential(
            ops.GroupNorm(32, out_dim, dtype=dtype, device=device), nn.SiLU(), nn.Dropout(dropout),
            ops.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0), dtype=dtype, device=device))
        self.conv3 = nn.Sequential(
            ops.GroupNorm(32, out_dim, dtype=dtype, device=device), nn.SiLU(), nn.Dropout(dropout),
            ops.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0), dtype=dtype, device=device))
        self.conv4 = nn.Sequential(
            ops.GroupNorm(32, out_dim, dtype=dtype, device=device), nn.SiLU(), nn.Dropout(dropout),
            ops.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0), dtype=dtype, device=device))

        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        if use_attn_strength(self.temporal_conv_strength):
            x = identity + (self.temporal_conv_strength * 0.1) * x
        else:
            x = identity + x
        return x

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
    :param use_checkpoint: if True, use gradient checkpointing on this module.
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
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        kernel_size=3,
        exchange_temb_dims=False,
        skip_t_emb=False,
        dtype=None,
        device=None,
        operations=ops,
        use_temporal_conv=True,
        temporal_conv_strength=1.0,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims
        self.use_temporal_conv = use_temporal_conv
        self.use_cfg = True

        if isinstance(kernel_size, list):
            padding = [k // 2 for k in kernel_size] 
        else:
            padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            operations.GroupNorm(32, channels, dtype=dtype, device=device),
            nn.SiLU(),
            operations.conv_nd(dims, channels, self.out_channels, 3, padding=1, dtype=dtype, device=device),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, dtype=dtype, device=device)
            self.x_upd = Upsample(channels, False, dims, dtype=dtype, device=device)
        elif down:
            self.h_upd = Downsample(channels, False, dims, dtype=dtype, device=device)
            self.x_upd = Downsample(channels, False, dims, dtype=dtype, device=device)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        if self.skip_t_emb:
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                operations.Linear(
                    emb_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels, dtype=dtype, device=device
                ),
            )
        self.out_layers = nn.Sequential(
            operations.GroupNorm(32, self.out_channels, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(operations.conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, dtype=dtype, device=device))
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = operations.conv_nd(
                dims, channels, self.out_channels, 3, padding=1, dtype=dtype, device=device
            )
        else:
            self.skip_connection = operations.conv_nd(dims, channels, self.out_channels, 1, dtype=dtype, device=device)

        if use_temporal_conv:
            self.temopral_conv = TemporalConvBlock_v2(
                self.out_channels,
                self.out_channels,
                dropout=0.1,
                temporal_conv_strength=temporal_conv_strength
            )

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return self._forward(x, emb)

    def _forward(self, x, emb):
        b = x.shape[0]
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = None
        if not self.skip_t_emb:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            h = out_norm(h)
            if emb_out is not None:
                scale, shift = th.chunk(emb_out, 2, dim=1)
                h *= (1 + scale)
                h += shift
            h = out_rest(h)
        else:
            if emb_out is not None:
                if self.exchange_temb_dims:
                    emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
                h = h + emb_out
            h = self.out_layers(h)

        h = self.skip_connection(x) + h

        if self.use_temporal_conv:
            batch_size = get_batch_size(b, self.use_cfg)
            h = rearrange(h, '(b f) c h w -> b c f h w', b=batch_size)
            h = self.temopral_conv(h)
            h = rearrange(h, 'b c f h w -> (b f) c h w')
            
        return h

class TemporalTransformer(nn.Module): # nn.Module -> SpatialTransformer
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head,
                 depth=1,
                 dropout=0.0,
                 context_dim=None,
                 disable_self_attn=False,
                 use_linear=False,
                 use_checkpoint=True,
                 only_self_att=True,
                 multiply_zero=False,
                 device='cuda',
                 dtype=torch.float16,
                 ops=operations,
                 use_temporal_attention=True,
                 temporal_attn_strength=1.0
                 ):
        super().__init__()
        self.multiply_zero = multiply_zero
        self.only_self_att = only_self_att
        self.use_temporal_attention = use_temporal_attention
        self.use_cfg = True
        self.temporal_attn_strength = temporal_attn_strength

        if self.only_self_att:
            context_dim = None
        if not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = ops.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, device=device, dtype=dtype)
        if not use_linear:
            self.proj_in = nn.Conv1d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0, device=device, dtype=dtype)
        else:
            self.proj_in = ops.Linear(in_channels, inner_dim, device=device, dtype=dtype)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim[d],
                checkpoint=use_checkpoint,
                device=device,
                dtype=dtype
                ) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv1d(
                    inner_dim, in_channels, kernel_size=1, stride=1,
                    padding=0)).to(dtype=dtype)
        else:
            self.proj_out = zero_module(ops.Linear(in_channels, inner_dim, device=device, dtype=dtype))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        if not self.use_temporal_attention:
            return x
        # note: if no context is given, cross-attention defaults to self-attention
        if self.only_self_att:
            context = None
        if not isinstance(context, list):
            context = [context]
        x_in = x
        b = get_batch_size(x_in.shape[0], self.use_cfg)
        x = rearrange(x, '(b f) c h w -> b c f h w', b=b)
        b, c, f, h, w = x.shape
        
        x = self.norm(x)

        if not self.use_linear:
            x = rearrange(x, 'b c f h w -> (b h w) c f').contiguous()
            x = self.proj_in(x)
        if self.use_linear:
            x = rearrange(
                x, '(b f) c h w -> b (h w) f c', f=self.frames).contiguous()
            x = self.proj_in(x)

        if self.only_self_att:
            x = rearrange(x, 'bhw c f -> bhw f c').contiguous()
            for i, block in enumerate(self.transformer_blocks):
                x = block(x)
            x = rearrange(x, '(b hw) f c -> b hw f c', b=b).contiguous()
        else:
            x = rearrange(x, '(b hw) c f -> b hw f c', b=b).contiguous()
            for i, block in enumerate(self.transformer_blocks):
                context[i] = rearrange(
                    context[i], '(b f) l con -> b f l con',
                    f=self.frames).contiguous()
                # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    context_i_j = repeat(
                        context[i][j],
                        'f l con -> (f r) l con',
                        r=(h * w) // self.frames,
                        f=self.frames).contiguous()
                    x[j] = block(x[j], context=context_i_j)

        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, 'b (h w) f c -> b f c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = rearrange(x, 'b hw f c -> (b hw) c f').contiguous()
            x = self.proj_out(x)
            x = rearrange(
                x, '(b h w) c f -> b c f h w', b=b, h=h, w=w).contiguous()

        x = rearrange(x, 'b c f h w -> (b f) c h w')

        if use_attn_strength(self.temporal_attn_strength):
            x = (self.temporal_attn_strength * 0.1) * x + x_in
        else:
            x = x + x_in
        return x
