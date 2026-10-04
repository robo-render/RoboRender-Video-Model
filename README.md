# RoboRender

## Setup

```bash
pip install -r requirements.txt

hf download alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control --local-dir models/Wan2.1-Fun-V1.1-1.3B-Control \
    --include "diffusion_pytorch_model.safetensors" "Wan2.1_VAE.pth" "models_t5_umt5-xxl-enc-bf16.pth" \
              "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" "google/umt5-xxl/*"
hf download RoboRender/roborender-video roborender_lora.safetensors --local-dir ckpt
```

## Data

```
sample/
  original_instruction.txt              # prompt
  <view>/depth/original_depth_vda.mp4   # depth control
  <view>/mask/rgb_robot.mp4             # robot mask control
  <view>/pre_frame.png                  # optional real frame for clip 0
```

416x240 at 15 fps. `height * num_views` must be divisible by 16.

## Inference

```bash
python infer.py --sample_dir data/test_long_sample_00502 --views hand_left hand_right head_color \
    --output_dir outputs/demo [--use_prev_frame] [--teacache]
```

| flag | default | |
|---|---|---|
| `--views` | | view names, or `name:depth.mp4:mask.mp4` |
| `--use_prev_frame` | off | condition clip 0 on `pre_frame.png`; later clips use the previous clip's last frame |
| `--first_clip_refs` | | explicit clip-0 references, `name:image.png` |
| `--teacache` | off | ~2.5x faster, minor quality loss (`--teacache_thresh 0.10`) |
| `--height --width` | 240 416 | per-view resolution |
| `--clip_length --overlap` | 81 1 | |
| `--num_inference_steps --cfg_scale --seed` | 50 5.0 42 | |
| `--save_clips` | off | also save each clip |

Writes `generated_<view>.mp4` and `comparison.mp4` (depth | mask | generated). Three views need about 20 GB GPU memory.

## Acknowledgements

Model code adapted from [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and
[Wan2.1](https://github.com/Wan-Video/Wan2.1); each file in `core/` notes its source.
