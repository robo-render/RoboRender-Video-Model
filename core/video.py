# Adapted from DiffSynth-Studio: diffsynth/utils/data/__init__.py and the image/video tensor helpers in diffsynth/diffusion/base_pipeline.py.
import imageio
import numpy as np
import torch
from einops import rearrange, reduce, repeat
from PIL import Image
from tqdm import tqdm


def crop_and_resize(image, height, width):
    image = np.array(image)
    h, w, _ = image.shape
    if h / w < height / width:
        cw = int(h / height * width)
        left = (w - cw) // 2
        image = image[:, left : left + cw]
    else:
        ch = int(w / width * height)
        top = (h - ch) // 2
        image = image[top : top + ch, :]
    return Image.fromarray(image).resize((width, height))


def load_video(path, height, width):
    reader = imageio.get_reader(path)
    frames = []
    for i in range(reader.count_frames()):
        frame = Image.fromarray(np.array(reader.get_data(i))).convert("RGB")
        if frame.size != (width, height):
            frame = crop_and_resize(frame, height, width)
        frames.append(frame)
    reader.close()
    return frames


def save_video(frames, path, fps=15, quality=5):
    writer = imageio.get_writer(path, fps=fps, quality=quality)
    for frame in tqdm(frames, desc="Saving video"):
        writer.append_data(np.array(frame))
    writer.close()


def image_to_tensor(image, dtype, device):
    """PIL -> (1, C, H, W) in [-1, 1]."""
    x = torch.Tensor(np.array(image, dtype=np.float32)).to(dtype=dtype, device=device)
    x = x * (2 / 255) + (-1)
    return repeat(x, "H W C -> B C H W", B=1)


def video_to_tensor(frames, dtype, device):
    """list of PIL -> (1, C, T, H, W) in [-1, 1]."""
    return torch.stack([image_to_tensor(f, dtype, device) for f in frames], dim=2)


def tensor_to_video(x):
    """(1, C, T, H, W) in [-1, 1] -> list of PIL."""
    x = reduce(x, "B C T H W -> T H W C", reduction="mean")
    frames = []
    for frame in x:
        frame = ((frame - (-1)) * (255 / 2)).clip(0, 255).to(device="cpu", dtype=torch.uint8)
        frames.append(Image.fromarray(frame.numpy()))
    return frames


def hstack(images):
    h = max(im.height for im in images)
    ims = [im if im.height == h else im.resize((int(im.width * h / im.height), h)) for im in images]
    out = Image.new("RGB", (sum(im.width for im in ims), h))
    x = 0
    for im in ims:
        out.paste(im, (x, 0))
        x += im.width
    return out


def vstack(images):
    w = max(im.width for im in images)
    ims = [im if im.width == w else im.resize((w, int(im.height * w / im.width))) for im in images]
    out = Image.new("RGB", (w, sum(im.height for im in ims)))
    y = 0
    for im in ims:
        out.paste(im, (0, y))
        y += im.height
    return out
