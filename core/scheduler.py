# Adapted from DiffSynth-Studio: diffsynth/diffusion/flow_match.py (Wan template only).
import torch


class FlowMatchScheduler:
    def __init__(self, num_train_timesteps=1000):
        self.num_train_timesteps = num_train_timesteps

    def set_timesteps(self, num_inference_steps=50, shift=5.0):
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.sigmas = sigmas
        self.timesteps = sigmas * self.num_train_timesteps

    def step(self, model_output, timestep, sample):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sigma_ = 0 if timestep_id + 1 >= len(self.timesteps) else self.sigmas[timestep_id + 1]
        return sample + model_output * (sigma_ - sigma)
