# Adapted from DiffSynth-Studio: diffsynth/models/wan_video_image_encoder.py (vision tower of open-clip XLM-RoBERTa ViT-H/14 only).
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import flash_attention

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, proj_dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.proj_dropout = proj_dropout
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        x = flash_attention(q, k, v, num_heads=self.num_heads, compatibility_mode=True)
        return F.dropout(self.proj(x), self.proj_dropout, self.training)


class AttentionBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, num_heads, norm_eps=1e-5):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = LayerNorm(dim, eps=norm_eps)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(0.0))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=14, dim=1280, mlp_ratio=4, out_dim=1024, num_heads=16, num_layers=32, norm_eps=1e-5):
        super().__init__()
        self.image_size = image_size
        num_patches = (image_size // patch_size) ** 2
        gain = 1.0 / math.sqrt(dim)
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(gain * torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(0.0)
        self.pre_norm = LayerNorm(dim, eps=norm_eps)
        self.transformer = nn.Sequential(*[AttentionBlock(dim, mlp_ratio, num_heads, norm_eps) for _ in range(num_layers)])
        self.post_norm = LayerNorm(dim, eps=norm_eps)
        self.head = nn.Parameter(gain * torch.randn(dim, out_dim))

    def forward(self, x):
        b = x.size(0)
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        x = torch.cat([self.cls_embedding.expand(b, -1, -1).to(dtype=x.dtype, device=x.device), x], dim=1)
        x = self.dropout(x + self.pos_embedding.to(dtype=x.dtype, device=x.device))
        x = self.pre_norm(x)
        return self.transformer[:-1](x)  # penultimate-block features, (B, 257, 1280)


class XLMRobertaCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = VisionTransformer()
        self.log_scale = nn.Parameter(math.log(1 / 0.07) * torch.ones([]))


class WanImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = XLMRobertaCLIP()

    def encode_image(self, images):
        # images: list of (1, 3, H, W) tensors in [-1, 1]
        size = (self.model.visual.image_size,) * 2
        x = torch.cat([F.interpolate(u, size=size, mode="bicubic", align_corners=False) for u in images])
        x = x.mul_(0.5).add_(0.5)
        mean = torch.tensor(CLIP_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        x = x.sub_(mean).div_(std)
        return self.model.visual(x)
