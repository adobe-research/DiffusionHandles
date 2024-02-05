import torch
import numpy as np
from PIL import Image
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

from depth_estimator import DepthEstimator

class ZoeDepthEstimator(DepthEstimator):

    def __init__(self):
        super().__init__()
        conf = get_config("zoedepth_nk", "infer")
        self.model = build_model(conf)

    def to(self, device: torch.device):
        self.model.to(device)
        return self
    
    def estimate_depth(self, img: torch.Tensor):
        return self.model.infer(img)

    @staticmethod
    def get_intrinsics(h: int, w: int):
        """
        Intrinsics for a pinhole camera model.
        Assume fov of 55 degrees and central principal point.
        """
        f = 0.5 * w / np.tan(0.5 * 6.24 * np.pi / 180.0) #car benchmark
        #f = 0.5 * W / np.tan(0.5 * 7.18 * np.pi / 180.0) #airplane benchmark
        #f = 0.5 * W / np.tan(0.5 * 14.9 * np.pi / 180.0) #chair, cup, lamp, stool benchmark        
        #f = 0.5 * W / np.tan(0.5 * 7.23 * np.pi / 180.0) #plant benchmark            
        f = 0.5 * w / np.tan(0.5 * 55 * np.pi / 180.0)    
        cx = 0.5 * w
        cy = 0.5 * h
        return torch.tensor([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
            ])
