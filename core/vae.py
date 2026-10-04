# Adapted from DiffSynth-Studio: diffsynth/models/wan_video_vae.py (Wan2.1 VAE only).
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from tqdm import tqdm

CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        return super().forward(F.pad(x, padding))


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode == "upsample2d":
            self.resample = nn.Sequential(Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == "upsample3d":
            self.resample = nn.Sequential(Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(), CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout), CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity


def _conv_with_cache(layer, x, feat_cache, feat_idx):
    idx = feat_idx[0]
    cache_x = x[:, :, -CACHE_T:, :, :].clone()
    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
        cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
    x = layer(x, feat_cache[idx])
    feat_cache[idx] = cache_x
    feat_idx[0] += 1
    return x


class Encoder3d(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_downsample=[True, True, False], dropout=0.0):
        super().__init__()
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                downsamples.append(Resample(out_dim, mode="downsample3d" if temperal_downsample[i] else "downsample2d"))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim), ResidualBlock(out_dim, out_dim, dropout))
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x = _conv_with_cache(self.conv1, x, feat_cache, feat_idx) if feat_cache is not None else self.conv1(x)
        for layer in self.downsamples:
            x = layer(x, feat_cache, feat_idx) if feat_cache is not None else layer(x)
        for layer in self.middle:
            x = layer(x, feat_cache, feat_idx) if isinstance(layer, ResidualBlock) and feat_cache is not None else layer(x)
        for layer in self.head:
            x = _conv_with_cache(layer, x, feat_cache, feat_idx) if isinstance(layer, CausalConv3d) and feat_cache is not None else layer(x)
        return x


class Decoder3d(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_upsample=[False, True, True], dropout=0.0):
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0], dropout))
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i in (1, 2, 3):
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                upsamples.append(Resample(out_dim, mode="upsample3d" if temperal_upsample[i] else "upsample2d"))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x = _conv_with_cache(self.conv1, x, feat_cache, feat_idx) if feat_cache is not None else self.conv1(x)
        for layer in self.middle:
            x = layer(x, feat_cache, feat_idx) if isinstance(layer, ResidualBlock) and feat_cache is not None else layer(x)
        for layer in self.upsamples:
            x = layer(x, feat_cache, feat_idx) if feat_cache is not None else layer(x)
        for layer in self.head:
            x = _conv_with_cache(layer, x, feat_cache, feat_idx) if isinstance(layer, CausalConv3d) and feat_cache is not None else layer(x)
        return x


def count_conv3d(model):
    return sum(isinstance(m, CausalConv3d) for m in model.modules())


class VideoVAE_(nn.Module):
    def __init__(self, dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks, attn_scales, temperal_downsample[::-1], dropout)

    def encode(self, x, scale):
        self.clear_cache()
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
        return (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)

    def decode(self, z, scale):
        self.clear_cache()
        scale = [s.to(dtype=z.dtype, device=z.device) for s in scale]
        z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        x = self.conv2(z)
        for i in range(z.shape[2]):
            self._conv_idx = [0]
            out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            out = out_ if i == 0 else torch.cat([out, out_], 2)
        return out

    def clear_cache(self):
        self._conv_idx = [0]
        self._feat_map = [None] * count_conv3d(self.decoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * count_conv3d(self.encoder)


class WanVideoVAE(nn.Module):
    def __init__(self, z_dim=16):
        super().__init__()
        mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
        std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]
        self.model = VideoVAE_(z_dim=z_dim).eval().requires_grad_(False)
        self.upsampling_factor = 8
        self.z_dim = z_dim

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])
        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)
        mask = torch.stack([h, w]).min(dim=0).values
        return rearrange(mask, "H W -> 1 1 1 H W")

    def _tiles(self, H, W, size_h, size_w, stride_h, stride_w):
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                tasks.append((h, h + size_h, w, w + size_w))
        return tasks

    def tiled_decode(self, hidden_states, device, tile_size, tile_stride):
        _, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride
        up = self.upsampling_factor
        out_T = T * 4 - 3
        weight = torch.zeros((1, 1, out_T, H * up, W * up), dtype=hidden_states.dtype, device="cpu")
        values = torch.zeros((1, 3, out_T, H * up, W * up), dtype=hidden_states.dtype, device="cpu")
        for h, h_, w, w_ in tqdm(self._tiles(H, W, size_h, size_w, stride_h, stride_w), desc="VAE decoding"):
            batch = self.model.decode(hidden_states[:, :, :, h:h_, w:w_].to(device), self.scale).to("cpu")
            mask = self.build_mask(batch, is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                                   border_width=((size_h - stride_h) * up, (size_w - stride_w) * up)).to(dtype=hidden_states.dtype)
            th, tw = h * up, w * up
            values[:, :, :, th : th + batch.shape[3], tw : tw + batch.shape[4]] += batch * mask
            weight[:, :, :, th : th + batch.shape[3], tw : tw + batch.shape[4]] += mask
        return (values / weight).clamp_(-1, 1)

    def tiled_encode(self, video, device, tile_size, tile_stride):
        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride
        up = self.upsampling_factor
        out_T = (T + 3) // 4
        weight = torch.zeros((1, 1, out_T, H // up, W // up), dtype=video.dtype, device="cpu")
        values = torch.zeros((1, self.z_dim, out_T, H // up, W // up), dtype=video.dtype, device="cpu")
        for h, h_, w, w_ in tqdm(self._tiles(H, W, size_h, size_w, stride_h, stride_w), desc="VAE encoding"):
            batch = self.model.encode(video[:, :, :, h:h_, w:w_].to(device), self.scale).to("cpu")
            mask = self.build_mask(batch, is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                                   border_width=((size_h - stride_h) // up, (size_w - stride_w) // up)).to(dtype=video.dtype)
            th, tw = h // up, w // up
            values[:, :, :, th : th + batch.shape[3], tw : tw + batch.shape[4]] += batch * mask
            weight[:, :, :, th : th + batch.shape[3], tw : tw + batch.shape[4]] += mask
        return values / weight

    def encode(self, videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        videos = [video.to("cpu") for video in videos]
        out = []
        for video in videos:
            video = video.unsqueeze(0)
            if tiled:
                up = self.upsampling_factor
                hidden = self.tiled_encode(video, device, (tile_size[0] * up, tile_size[1] * up), (tile_stride[0] * up, tile_stride[1] * up))
            else:
                hidden = self.model.encode(video.to(device), self.scale)
            out.append(hidden.squeeze(0))
        return torch.stack(out)

    def decode(self, hidden_states, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        hidden_states = [h.to("cpu") for h in hidden_states]
        out = []
        for hidden in hidden_states:
            hidden = hidden.unsqueeze(0)
            if tiled:
                video = self.tiled_decode(hidden, device, tile_size, tile_stride)
            else:
                video = self.model.decode(hidden.to(device), self.scale).clamp_(-1, 1)
            out.append(video.squeeze(0))
        return torch.stack(out)
