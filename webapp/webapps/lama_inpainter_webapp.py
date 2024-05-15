import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import argparse

import torch
import numpy.typing as npt
import scipy

from saicinpainting import LamaInpainter

from foreground_remover_webapp import ForegroundRemoverWebapp

from utils import crop_and_resize

class LamaInpainterWebapp(ForegroundRemoverWebapp):

    def __init__(self, netpath: str, port: int, device: str = 'cuda:0'):
        super().__init__(netpath=netpath, port=port)

        self.img_res = 512

        self.inpainter = LamaInpainter()
        self.inpainter.to(device)

    def remove_foreground(self, img: npt.NDArray = None, fg_mask: npt.NDArray = None, dilation: int = 3) -> npt.NDArray:

        print('run_lama_inpainter')

        if img is None or fg_mask is None:
            raise ValueError('Some inputs are missing.')

        print(f'{img.shape} {img.dtype}')
        print(f'{fg_mask.shape} {fg_mask.dtype}')

        # prepare inputs (convert to torch tensors, crop, resize)
        img = torch.from_numpy(img).to(device=self.inpainter.model.device, dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        fg_mask = torch.from_numpy(fg_mask).to(device=self.inpainter.model.device, dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        img = crop_and_resize(img=img, size=self.img_res)
        if fg_mask.shape[1] > 1:
            fg_mask = fg_mask.mean(dim=1, keepdim=True) # average channels
        fg_mask = crop_and_resize(img=fg_mask, size=self.img_res)
        fg_mask = (fg_mask>0.5).to(device=self.inpainter.model.device, dtype=torch.float32)

        # inpaint the foreground region to get a background image without the foreground object
        if dilation >= 0:
            fg_mask = fg_mask.cpu().numpy() > 0.5
            fg_mask = scipy.ndimage.binary_dilation(fg_mask[0, 0], iterations=dilation)[None, None, ...]
            fg_mask = torch.from_numpy(fg_mask).to(device=self.inpainter.model.device, dtype=torch.float32)
        bg_img = self.inpainter.inpaint(image=img, mask=fg_mask)

        print(bg_img.shape)

        bg_img = (bg_img * 255.0)[0].permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()

        return bg_img

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--netpath', type=str, default='/foreground_remover')
    parser.add_argument('--port', type=int, default=6008)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    server = LamaInpainterWebapp(netpath=args.netpath, port=args.port, device=args.device)
    server.start()
