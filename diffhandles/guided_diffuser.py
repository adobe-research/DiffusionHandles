from typing import List

import torch

class GuidedDiffuser:
    
    def __init__(self, conf):
        self.conf = conf

    def to(self, device: torch.device = None, dtype: torch.dtype = None):
        raise NotImplementedError

    @staticmethod
    def get_depth_intrinsics(device: torch.device = None):
        """
        Return intrinsics suitable for the input depth.
        """
        raise NotImplementedError

    def encode_latent_image(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode_latent_image(self, latent_image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @torch.no_grad()
    def initial_inference(self, init_latents: torch.Tensor, depth: torch.Tensor, uncond_embeddings: torch.Tensor, prompt: str):
        raise NotImplementedError

    @torch.no_grad()
    def guided_inference(
            self, latents: torch.Tensor, depth: torch.Tensor, uncond_embeddings: torch.Tensor, prompt: str,
            activations_orig: list[torch.Tensor],
            correspondences: torch.Tensor, save_denoising_steps: bool = False):
        raise NotImplementedError
