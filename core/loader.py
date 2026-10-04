# Adapted from DiffSynth-Studio: diffsynth/core/loader, diffsynth/utils/state_dict_converters and diffsynth/utils/lora/general.py.
from contextlib import contextmanager

import torch
from safetensors import safe_open


@contextmanager
def skip_init():
    """Register parameters on the meta device so model construction is free; weights are assigned on load."""
    old = torch.nn.Module.register_parameter

    def register(module, name, param):
        old(module, name, param)
        if param is not None:
            p = module._parameters[name]
            module._parameters[name] = type(p)(p.to("meta"), requires_grad=param.requires_grad)

    torch.nn.Module.register_parameter = register
    try:
        yield
    finally:
        torch.nn.Module.register_parameter = old


def load_state_dict(path, dtype=None, device="cpu"):
    if str(path).endswith(".safetensors"):
        sd = {}
        with safe_open(path, framework="pt", device=str(device)) as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
    else:
        sd = torch.load(path, map_location=device, weights_only=True)
        if len(sd) == 1:
            for key in ("state_dict", "module", "model_state"):
                if key in sd:
                    sd = sd[key]
                    break
    if dtype is not None:
        sd = {k: (v.to(dtype) if isinstance(v, torch.Tensor) else v) for k, v in sd.items()}
    return sd


def convert_dit(sd):
    return {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items() if not k.startswith("vace")}


def convert_vae(sd):
    if "model_state" in sd:
        sd = sd["model_state"]
    return {"model." + k: v for k, v in sd.items()}


def convert_clip(sd):
    return {"model." + k: v for k, v in sd.items() if not k.startswith("textual.")}


def load_model(model_class, path, kwargs=None, converter=None, dtype=torch.bfloat16, device="cuda"):
    with skip_init():
        model = model_class(**(kwargs or {}))
    sd = load_state_dict(path, dtype, device)
    if converter is not None:
        sd = converter(sd)
    model.load_state_dict(sd, assign=True)
    return model.to(dtype=dtype, device=device).eval()


def _lora_pairs(lora):
    pairs = {}
    for key in lora:
        a_key, b_key = ("lora_down", "lora_up") if ".lora_up." in key else ("lora_A", "lora_B")
        if b_key not in key:
            continue
        keys = key.split(".")
        if len(keys) > keys.index(b_key) + 2:
            keys.pop(keys.index(b_key) + 1)
        keys.pop(keys.index(b_key))
        if keys[0] == "diffusion_model":
            keys.pop(0)
        keys.pop(-1)
        pairs[".".join(keys)] = (key, key.replace(b_key, a_key))
    return pairs


def fuse_lora(model, lora, alpha=1.0, dtype=torch.bfloat16, device="cuda"):
    pairs = _lora_pairs(lora)
    fused = 0
    for name, module in model.named_modules():
        if name not in pairs:
            continue
        up = lora[pairs[name][0]].to(device=device, dtype=dtype)
        down = lora[pairs[name][1]].to(device=device, dtype=dtype)
        if up.dim() == 4:
            delta = alpha * torch.mm(up.squeeze(3).squeeze(2), down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            delta = alpha * torch.mm(up, down)
        module.weight.data = module.weight.data.to(device=device, dtype=dtype) + delta
        fused += 1
    print(f"{fused} tensors are fused by LoRA.")
    return fused
