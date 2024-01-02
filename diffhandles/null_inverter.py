import torch

from diffhandles.diffuser import Diffuser

class NullInverter:

    def __init__(self, model: Diffuser):
        self.model = model
        
    def to(self, device: torch.device):
        self.model.to(device)
        return self

    def invert(self, target_img: torch.Tensor, depth: torch.Tensor, prompt: str, num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        raise NotImplementedError("Null inverter must implement invert method.")
