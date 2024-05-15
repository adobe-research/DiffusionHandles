import argparse
import torch
import torchvision
import numpy.typing as npt

from lang_sam import LangSAM

from foreground_selector_webapp import ForegroundSelectorWebapp
from utils import crop_and_resize

class LangSAMSegmenterWebapp(ForegroundSelectorWebapp):
    def __init__(self, netpath: str, port: int, device: str = 'cuda:0'):
        super().__init__(netpath=netpath, port=port)

        self.img_res = 512

        if device != 'cuda:0':
            # LangSAM currently returns empty tensors on other devices
            # (Probably some hard-coded device somewhere in their code?)
            raise RuntimeError("Currently only 'cuda:0' is supported as device.")

        self.segmenter = LangSAM()
        self.segmenter.sam.model.to(device)
        self.segmenter.device = device

    def select_foreground(self, img: npt.NDArray = None, prompt: str = None):

        print('run_langsam_segmenter')

        if img is None or prompt is None:
            raise ValueError('Some inputs are missing.')

        print(f'{img.shape} {img.dtype}')
        print(f'{prompt}')

        # prepare inputs (convert to torch tensors, crop, resize)
        img = torch.from_numpy(img).to(device=self.segmenter.device, dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        img = crop_and_resize(img=img, size=self.img_res)
        
        # print(prompt)
        # print(img.shape)
        # print(img.max())
        # print(img.min())
        # print(img.dtype)
        
        # TODO: this currently only works on 'cuda:0', on other devices the output is an empty tensor
        masks, boxes, prompts, logits = self.segmenter.predict(
            image_pil=torchvision.transforms.functional.to_pil_image(img[0]),
            text_prompt=prompt)
        print(masks.shape)
        print(boxes)
        print(prompts)
        print(logits)
        if len(prompts) == 0:
            # nothing was selected
            mask = torch.zeros((3, self.img_res, self.img_res), device=self.segmenter.device, dtype=torch.float32)
        else:
            mask = masks[[0, 0, 0], :, :].to(device=self.segmenter.device, dtype=torch.float32)

        print(mask.shape)

        mask = (mask * 255.0).permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()

        return mask

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--netpath', type=str, default='/foreground_selector')
    parser.add_argument('--port', type=int, default=6010)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    server = LangSAMSegmenterWebapp(netpath=args.netpath, port=args.port, device=args.device)
    server.start()
