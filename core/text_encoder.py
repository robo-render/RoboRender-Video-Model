# Adapted from DiffSynth-Studio: diffsynth/models/wan_video_text_encoder.py (umT5-XXL encoder + tokenizer).
import html
import math

import ftfy
import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


def fp16_clamp(x):
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)
        return self.weight * x


class T5Attention(nn.Module):
    def __init__(self, dim, dim_attn, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, pos_bias=None):
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim
        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            mask = mask.view(b, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)
        attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum("bnij,bjnc->binc", attn, v)
        return self.dropout(self.o(x.reshape(b, -1, n * c)))


class T5FeedForward(nn.Module):
    def __init__(self, dim, dim_ffn, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.fc1(x) * self.gate(x))
        return self.dropout(self.fc2(x))


class T5RelativeEmbedding(nn.Module):
    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.bidirectional = bidirectional
        self.max_dist = max_dist
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq, lk):
        device = self.embedding.weight.device
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(lq, device=device).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        return self.embedding(rel_pos).permute(2, 0, 1).unsqueeze(0).contiguous()

    def _relative_position_bucket(self, rel_pos):
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (torch.log(rel_pos.float() / max_exact) / math.log(self.max_dist / max_exact) * (num_buckets - max_exact)).long()
        rel_pos_large = torch.min(rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        return rel_buckets + torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)


class T5SelfAttention(nn.Module):
    def __init__(self, dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos=True, dropout=0.1):
        super().__init__()
        self.shared_pos = shared_pos
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)

    def forward(self, x, mask=None, pos_bias=None):
        e = pos_bias if self.shared_pos else self.pos_embedding(x.size(1), x.size(1))
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        return fp16_clamp(x + self.ffn(self.norm2(x)))


class WanTextEncoder(nn.Module):
    def __init__(self, vocab=256384, dim=4096, dim_attn=4096, dim_ffn=10240, num_heads=64, num_layers=24, num_buckets=32, shared_pos=False, dropout=0.1):
        super().__init__()
        self.shared_pos = shared_pos
        self.token_embedding = nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True) if shared_pos else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos, dropout) for _ in range(num_layers)])
        self.norm = T5LayerNorm(dim)

    def forward(self, ids, mask=None):
        x = self.dropout(self.token_embedding(ids))
        e = self.pos_embedding(x.size(1), x.size(1)) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        return self.dropout(self.norm(x))


def _clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()


class HuggingfaceTokenizer:
    def __init__(self, name, seq_len=512):
        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(name)

    def __call__(self, text):
        ids = self.tokenizer([_clean(text)], return_tensors="pt", padding="max_length", truncation=True, max_length=self.seq_len)
        return ids.input_ids, ids.attention_mask
