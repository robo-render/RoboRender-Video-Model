"""Long-video sim-to-real transfer: multi-view, depth + mask control, autoregressive previous-frame conditioning."""
import argparse
import os
import time
from pathlib import Path

from PIL import Image

from core import RoboRenderPipeline
from core.video import hstack, load_video, save_video, vstack

DEFAULT_MODEL_DIR = "models/Wan2.1-Fun-V1.1-1.3B-Control"
DEFAULT_LORA = "ckpt/roborender_lora.safetensors"
DEPTH_FILE = "depth/original_depth_vda.mp4"
MASK_FILE = "mask/rgb_robot.mp4"
PREV_FRAME_FILE = "pre_frame.png"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", nargs="+", required=True,
                   help="'name:depth.mp4[:mask.mp4]' per view, or just 'name' when --sample_dir is given "
                        f"(then {DEPTH_FILE} / {MASK_FILE} under <sample_dir>/<name>/ are used)")
    p.add_argument("--sample_dir", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None, help="defaults to <sample_dir>/original_instruction.txt")
    p.add_argument("--use_prev_frame", action="store_true",
                   help=f"condition clip 0 on <sample_dir>/<view>/{PREV_FRAME_FILE} for every view")
    p.add_argument("--first_clip_refs", nargs="*", default=None,
                   help="explicit clip-0 references, 'name:image.png'; unlisted views get zero reference")
    p.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    p.add_argument("--lora_path", type=str, default=DEFAULT_LORA)
    p.add_argument("--output_dir", type=str, default="outputs/demo")
    p.add_argument("--height", type=int, default=240, help="per-view height")
    p.add_argument("--width", type=int, default=416)
    p.add_argument("--clip_length", type=int, default=81)
    p.add_argument("--overlap", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--teacache", action="store_true", help="enable TeaCache (~2x faster, minor quality loss)")
    p.add_argument("--teacache_thresh", type=float, default=0.10)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--save_clips", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def parse_view(spec, sample_dir):
    parts = spec.split(":")
    if len(parts) == 1:
        assert sample_dir, f"view '{spec}' has no paths; pass --sample_dir or use name:depth[:mask]"
        root = os.path.join(sample_dir, parts[0])
        mask = os.path.join(root, MASK_FILE)
        return parts[0], os.path.join(root, DEPTH_FILE), (mask if os.path.exists(mask) else None)
    if len(parts) == 2:
        return parts[0], parts[1], None
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(f"bad view spec '{spec}'")


def split_into_clips(frames, clip_length, overlap):
    """Overlapping clips; the last one is padded by repeating its final frame."""
    if len(frames) <= clip_length:
        clip = list(frames) + [frames[-1]] * (clip_length - len(frames))
        return [clip]
    clips, start = [], 0
    while start < len(frames):
        clip = list(frames[start : start + clip_length])
        clip += [clip[-1]] * (clip_length - len(clip))
        clips.append(clip)
        if start + clip_length >= len(frames):
            break
        start += clip_length - overlap
    return clips


def split_views(frames, num_views):
    h = frames[0].height // num_views
    return [[f.crop((0, i * h, f.width, (i + 1) * h)) for f in frames] for i in range(num_views)]


def main():
    args = parse_args()
    views = [parse_view(v, args.sample_dir) for v in args.views]
    names = [v[0] for v in views]
    has_mask = any(v[2] is not None for v in views)
    if args.prompt is None:
        assert args.sample_dir, "--prompt is required without --sample_dir"
        args.prompt = open(os.path.join(args.sample_dir, "original_instruction.txt")).read().strip()
    print(f"Views: {names}\nPrompt: {args.prompt}\nTeaCache: {args.teacache_thresh if args.teacache else 'off'}")

    # Load sources, truncate to the shortest stream, split into clips.
    depth = [load_video(v[1], args.height, args.width) for v in views]
    masks = [load_video(v[2], args.height, args.width) if v[2] else None for v in views]
    total = min([len(d) for d in depth] + [len(m) for m in masks if m is not None])
    depth = [d[:total] for d in depth]
    masks = [m[:total] if m is not None else None for m in masks]
    depth_clips = [split_into_clips(d, args.clip_length, args.overlap) for d in depth]
    mask_clips = [split_into_clips(m, args.clip_length, args.overlap) if m is not None else None for m in masks]
    num_clips = len(depth_clips[0])
    print(f"Source frames: {total}, clips: {num_clips} x {args.clip_length} (overlap {args.overlap})")

    # Clip-0 references.
    refs = None
    if args.use_prev_frame:
        assert args.sample_dir, "--use_prev_frame needs --sample_dir"
        refs = [Image.open(os.path.join(args.sample_dir, n, PREV_FRAME_FILE)).convert("RGB") for n in names]
    if args.first_clip_refs:
        ref_map = dict(s.split(":", 1) for s in args.first_clip_refs)
        refs = [Image.open(ref_map[n]).convert("RGB") if n in ref_map else None for n in names]

    pipe = RoboRenderPipeline(args.model_dir, args.lora_path, device=args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = [[] for _ in names]
    for ci in range(num_clips):
        print(f"\n=== clip {ci + 1}/{num_clips} | prev-frame conditioning: {'yes' if refs else 'no'} ===")
        t0 = time.perf_counter()
        video = pipe(
            prompt=args.prompt,
            control_views=[depth_clips[vi][ci] for vi in range(len(names))],
            mask_views=[mask_clips[vi][ci] for vi in range(len(names))] if has_mask else None,
            ref_views=refs,
            height=args.height, width=args.width, num_frames=args.clip_length,
            seed=args.seed, cfg_scale=args.cfg_scale, num_inference_steps=args.num_inference_steps,
            teacache_thresh=args.teacache_thresh if args.teacache else None,
        )
        print(f"clip {ci + 1} done in {time.perf_counter() - t0:.1f}s")
        clip_views = split_views(video, len(names))
        skip = args.overlap if ci > 0 else 0
        for vi in range(len(names)):
            generated[vi].extend(clip_views[vi][skip:])
        refs = [cv[-1].copy() for cv in clip_views]
        if args.save_clips:
            for vi, n in enumerate(names):
                save_video(clip_views[vi], str(out_dir / f"clip_{ci:03d}_{n}.mp4"), fps=args.fps)

    generated = [g[:total] for g in generated]
    for vi, n in enumerate(names):
        save_video(generated[vi], str(out_dir / f"generated_{n}.mp4"), fps=args.fps)
    comparison = []
    for fi in range(total):
        rows = []
        for vi in range(len(names)):
            parts = [depth[vi][fi]] + ([masks[vi][fi]] if masks[vi] is not None else []) + [generated[vi][fi]]
            rows.append(hstack(parts))
        comparison.append(vstack(rows))
    save_video(comparison, str(out_dir / "comparison.mp4"), fps=args.fps)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
