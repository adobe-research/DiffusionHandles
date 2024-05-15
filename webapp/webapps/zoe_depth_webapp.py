import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import argparse

import torch
import numpy.typing as npt

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

from depth_estimator_webapp import DepthEstimatorWebapp

from utils import crop_and_resize

class ZoeDepthWebapp(DepthEstimatorWebapp):
    def __init__(self, netpath: str, port: int, device: str = 'cuda:0'):
        super().__init__(netpath=netpath, port=port)

        self.img_res = 512

        conf = get_config("zoedepth_nk", "infer")
        self.depth_estimator = build_model(conf)
        self.depth_estimator.to(device)

    def estimate_depth(self, img: npt.NDArray = None) -> npt.NDArray:

        print('run_zoe_depth')

        if img is None:
            raise ValueError('Some inputs are missing.')

        print(f'{img.shape} {img.dtype}')

        # prepare input image (convert to torch tensor, crop, resize)
        img = torch.from_numpy(img).to(device=self.depth_estimator.device, dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        img = crop_and_resize(img=img, size=self.img_res)

        print(f'{img.shape} {img.dtype}')
        
        with torch.no_grad():
            depth = self.depth_estimator.infer(img)

        print(depth.shape)

        depth = depth[0].detach().cpu().permute(1, 2, 0).numpy()[..., 0]

        return depth

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--netpath', type=str, default='/depth_estimator')
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    server = ZoeDepthWebapp(netpath=args.netpath, port=args.port, device=args.device)
    server.start()
