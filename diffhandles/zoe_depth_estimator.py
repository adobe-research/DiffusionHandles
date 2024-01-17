import torch
import numpy as np
from PIL import Image
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

from diffhandles.depth_estimator import DepthEstimator

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

    @staticmethod
    def depth_to_points(depth: torch.Tensor, R=None, t=None):

        if depth.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported")

        depth = depth.squeeze(dim=0).cpu().numpy()

        K = ZoeDepthEstimator.get_intrinsics(depth.shape[1], depth.shape[2]).numpy()
        Kinv = np.linalg.inv(K)
        if R is None:
            R = np.eye(3)
        if t is None:
            t = np.zeros(3)

        # M converts from your coordinate to PyTorch3D's coordinate system
        M = np.eye(3)
        M[0, 0] = -1.0
        M[1, 1] = -1.0

        height, width = depth.shape[1:3]

        # print(height)
        # print(width)

        x = np.arange(width)
        y = np.arange(height)
        coord = np.stack(np.meshgrid(x, y), -1)
        coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
        coord = coord.astype(np.float32)
        # coord = torch.as_tensor(coord, dtype=torch.float32, device=device)
        coord = coord[None]  # bs, h, w, 3

        D = depth[:, :, :, None, None]
        # print(D.shape, Kinv[None, None, None, ...].shape, coord[:, :, :, :, None].shape )
        pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
        # pts3D_1 live in your coordinate system. Convert them to Py3D's
        pts3D_1 = M[None, None, None, ...] @ pts3D_1
        # from reference to targe tviewpoint
        pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
        # pts3D_2 = pts3D_1
        # depth_2 = pts3D_2[:, :, :, 2, :]  # b,1,h,w
        return torch.from_numpy(pts3D_2[:, :, :, :3, 0][0])
