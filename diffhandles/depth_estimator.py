import torch

class DepthEstimator:

    def __init__(self):
        pass

    def to(self, device: torch.device):
        raise NotImplementedError("Depth estimator must implement to method.")
    
    def estimate_depth(self, img: torch.Tensor):
        raise NotImplementedError("Depth estimator must implement estimate_depth method.")

    def get_intrinsics(self, h: int, w: int):
        raise NotImplementedError("Depth estimator must implement get_intrinsics method.")

    def depth_to_points(self, depth: torch.Tensor, R=None, t=None):
        raise NotImplementedError("Depth estimator must implement depth_to_points method.")

    def points_to_depth_merged(self, points, mod_ids, output_size=(512, 512), R=None, t=None, max_depth_value=float('inf')):
        raise NotImplementedError("Depth estimator must implement depth_to_points method.")
