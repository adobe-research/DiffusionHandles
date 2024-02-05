import torch

class DepthEstimator:

    def __init__(self):
        pass

    def to(self, device: torch.device):
        raise NotImplementedError("Depth estimator must implement to method.")
    
    def estimate_depth(self, img: torch.Tensor):
        raise NotImplementedError("Depth estimator must implement estimate_depth method.")

    @staticmethod
    def get_intrinsics(self, h: int, w: int):
        raise NotImplementedError("Depth estimator must implement get_intrinsics method.")
