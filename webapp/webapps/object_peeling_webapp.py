import argparse
import requests

import torch
import numpy.typing as npt
import numpy as np
import scipy
from PIL import Image

from foreground_remover_webapp import ForegroundRemoverWebapp
from utils import crop_and_resize

class ObjectPeelingWebapp(ForegroundRemoverWebapp):
    def __init__(self, netpath: str, port: int, server_url: str):
        super().__init__(netpath=netpath, port=port)
        
        self.img_res = 512
        self.server_url = server_url

    def remove_foreground(self, img: npt.NDArray = None, fg_mask: npt.NDArray = None, dilation: int = 3):

        print('run_object_peeling')

        if img is None or fg_mask is None:
            raise ValueError('Some inputs are missing.')

        print(f'{img.shape} {img.dtype}')
        print(f'{fg_mask.shape} {fg_mask.dtype}')

        # prepare inputs (convert to torch tensors, crop, resize)
        img = torch.from_numpy(img).to(dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        fg_mask = torch.from_numpy(fg_mask).to(dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        img = crop_and_resize(img=img, size=self.img_res)
        if fg_mask.shape[1] > 1:
            fg_mask = fg_mask.mean(dim=1, keepdim=True) # average channels
        fg_mask = crop_and_resize(img=fg_mask, size=self.img_res)
        fg_mask = (fg_mask>0.5).to(dtype=torch.float32)

        if dilation >= 0:
            fg_mask = fg_mask.cpu().numpy() > 0.5
            fg_mask = scipy.ndimage.binary_dilation(fg_mask[0, 0], iterations=dilation)[None, None, ...]
            fg_mask = torch.from_numpy(fg_mask).to(dtype=torch.float32)

        img = Image.fromarray((img * 255.0)[0].permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()).convert("RGB")
        mask = Image.fromarray((fg_mask * 255.0)[0, [0, 0, 0]].permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()).convert("L")

        str_img = img.tobytes().decode("latin1")
        str_mask = mask.tobytes().decode("latin1")

        mode_img = img.mode
        mode_mask = mask.mode


        img_w, img_h = img.size
        mask_w, mask_h = mask.size

        seed = 0

        data = {
            "str_img": str_img,
            "str_mask": str_mask,
            "mode_img": mode_img,
            "mode_mask": mode_mask,
            "img_w": img_w,
            "img_h": img_h,
            "mask_w": mask_w,
            "mask_h": mask_h,
            "seed": seed,
        }

        r = requests.post(self.server_url, json=data)
        print(r.status_code)
        r_json = r.json()
        rst_w, rst_h = r_json["rst_w"], r_json["rst_h"]
        bg_img = np.array(Image.frombytes("RGB", (rst_w, rst_h), r_json[f"str_result_0"].encode("latin1"), "raw"))

        print(bg_img.shape)

        return bg_img

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--netpath', type=str, default='/foreground_remover')
    parser.add_argument('--port', type=int, default=6008)
    # parser.add_argument('--server_url', type=str, default='https://sensei-eks02.infra.adobesensei.io/ilo-demo/podemo/app2/api') # from devbox
    parser.add_argument('--server_url', type=str, default='http://10.51.208.194:6003/ilo-demo/podemo/app2/api') # from runai
    # parser.add_argument('--server_url', type=str, default='http://10.51.135.240:6003/ilo-demo/podemo/app2/api') # from runai

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    server = ObjectPeelingWebapp(netpath=args.netpath, port=args.port, server_url=args.server_url)
    server.start()
