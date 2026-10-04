# Adapted from DiffSynth-Studio: WanVideoPipeline in diffsynth/pipelines/wan_video.py, reduced to the
# multi-view dual-control (depth + mask) + previous-frame-reference path used by RoboRender.
import os
from typing import List, Optional

import torch
from PIL import Image
from tqdm import tqdm

from .clip import WanImageEncoder
from .dit import TeaCache, WanModel
from .loader import convert_clip, convert_dit, convert_vae, fuse_lora, load_model, load_state_dict
from .scheduler import FlowMatchScheduler
from .text_encoder import HuggingfaceTokenizer, WanTextEncoder
from .vae import WanVideoVAE
from .video import image_to_tensor, tensor_to_video, video_to_tensor

DIT_CONFIG = dict(
    has_image_input=True, patch_size=[1, 2, 2], in_dim=48, dim=1536, ffn_dim=8960, freq_dim=256,
    text_dim=4096, out_dim=16, num_heads=12, num_layers=30, eps=1e-6, has_ref_conv=True,
)
TILE_SIZE = (30, 52)
TILE_STRIDE = (15, 26)


def _find(model_dir, *candidates):
    for name in candidates:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of {candidates} found in {model_dir}")


class RoboRenderPipeline:
    def __init__(self, model_dir, lora_path=None, lora_alpha=1.0, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.scheduler = FlowMatchScheduler()
        print("Loading models...")
        self.tokenizer = HuggingfaceTokenizer(_find(model_dir, "google/umt5-xxl"), seq_len=512)
        self.text_encoder = load_model(
            WanTextEncoder, _find(model_dir, "models_t5_umt5-xxl-enc-bf16.safetensors", "models_t5_umt5-xxl-enc-bf16.pth"),
            dtype=dtype, device=device)
        self.dit = load_model(
            WanModel, _find(model_dir, "diffusion_pytorch_model.safetensors"), DIT_CONFIG, convert_dit, dtype, device)
        self.vae = load_model(
            WanVideoVAE, _find(model_dir, "Wan2.1_VAE.safetensors", "Wan2.1_VAE.pth"), None, convert_vae, dtype, device)
        self.image_encoder = load_model(
            WanImageEncoder,
            _find(model_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
            None, convert_clip, dtype, device)
        if lora_path is not None:
            print(f"Loading LoRA from: {lora_path}")
            fuse_lora(self.dit, load_state_dict(lora_path, dtype, device), alpha=lora_alpha, dtype=dtype, device=device)

    # ---------------------------------------------------------------- encoders
    def encode_prompt(self, prompt):
        ids, mask = self.tokenizer(prompt)
        ids, mask = ids.to(self.device), mask.to(self.device)
        emb = self.text_encoder(ids, mask)
        for i, v in enumerate(mask.gt(0).sum(dim=1).long()):
            emb[i, v:] = 0
        return emb

    def encode_video(self, frames):
        x = video_to_tensor(frames, self.dtype, self.device)
        return self.vae.encode(x, device=self.device, tiled=True, tile_size=TILE_SIZE, tile_stride=TILE_STRIDE).to(dtype=self.dtype, device=self.device)

    def encode_control(self, control_views, mask_views):
        """Per-view VAE latents stacked along height, depth and mask concatenated along channels -> (1, 32, T, H, W)."""
        latents = [torch.cat([self.encode_video(v) for v in control_views], dim=3)]
        if mask_views is not None:
            latents.append(torch.cat([self.encode_video(v) for v in mask_views], dim=3))
        return torch.cat(latents, dim=1)

    def encode_reference(self, ref_views, width, view_height):
        """Previous-frame references (one per view, None allowed) -> (reference_latents, clip_feature)."""
        if ref_views is None or all(r is None for r in ref_views):
            return None, torch.zeros((1, 257, 1280), dtype=self.dtype, device=self.device)
        latents, clips = [], []
        for ref in ref_views:
            if ref is None:
                latents.append(None)
                clips.append(None)
                continue
            ref = ref.resize((width, view_height))
            latents.append(self.vae.encode(video_to_tensor([ref], self.dtype, self.device), device=self.device))
            clips.append(self.image_encoder.encode_image([image_to_tensor(ref, self.dtype, self.device)]))
        lat_ref = next(l for l in latents if l is not None)
        clip_ref = next(c for c in clips if c is not None)
        latents = [torch.zeros(lat_ref.shape, dtype=self.dtype, device=self.device) if l is None else l for l in latents]
        clips = [torch.zeros(clip_ref.shape, dtype=self.dtype, device=self.device) if c is None else c for c in clips]
        return torch.cat(latents, dim=3), torch.cat(clips, dim=1)

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        control_views: List[List[Image.Image]],
        mask_views: Optional[List[List[Image.Image]]] = None,
        ref_views: Optional[List[Optional[Image.Image]]] = None,
        negative_prompt: str = "",
        height: int = 240,
        width: int = 416,
        num_frames: int = 81,
        seed: int = 42,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        teacache_thresh: Optional[float] = None,
    ):
        num_views = len(control_views)
        total_height = height * num_views
        assert total_height % 16 == 0 and width % 16 == 0 and num_frames % 4 == 1, \
            f"need (height*num_views) % 16 == 0, width % 16 == 0, num_frames % 4 == 1; got {total_height}x{width}, {num_frames} frames"

        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)
        shape = (1, self.vae.z_dim, (num_frames - 1) // 4 + 1, total_height // 8, width // 8)
        noise = torch.randn(shape, generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32)
        latents = noise.to(dtype=self.dtype, device=self.device)

        context_posi = self.encode_prompt(prompt)
        context_nega = self.encode_prompt(negative_prompt) if cfg_scale != 1.0 else None
        y = self.encode_control(control_views, mask_views)
        reference_latents, clip_feature = self.encode_reference(ref_views, width, height)
        tea_posi = TeaCache(num_inference_steps, teacache_thresh) if teacache_thresh else None
        tea_nega = TeaCache(num_inference_steps, teacache_thresh) if teacache_thresh else None

        for i, timestep in enumerate(tqdm(self.scheduler.timesteps, desc="Denoising")):
            t = timestep.unsqueeze(0).to(dtype=self.dtype, device=self.device)
            pred = self.dit(latents, t, context_posi, clip_feature, y, reference_latents, tea_posi)
            if cfg_scale != 1.0:
                pred_nega = self.dit(latents, t, context_nega, clip_feature, y, reference_latents, tea_nega)
                pred = pred_nega + cfg_scale * (pred - pred_nega)
            latents = self.scheduler.step(pred, self.scheduler.timesteps[i], latents)

        video = self.vae.decode(latents, device=self.device, tiled=True, tile_size=TILE_SIZE, tile_stride=TILE_STRIDE)
        return tensor_to_video(video)
