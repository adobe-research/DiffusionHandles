import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import sys
import argparse

import torch
import numpy.typing as npt
import scipy
import gradio as gr
from fastapi import FastAPI
import uvicorn

from saicinpainting import LamaInpainter

from utils import crop_and_resize

class LamaInpainterWebapp:
    def __init__(self, netpath: str, port: int, device: str = 'cuda:0'):
        self.netpath = netpath
        self.port = port
        self.img_res = 512

        self.inpainter = LamaInpainter()
        self.inpainter.to(device)

    def run_lama_inpainter(self, img: npt.NDArray = None, fg_mask: npt.NDArray = None, dilation: int = 3):

        print('run_lama_inpainter')

        if img is None or fg_mask is None:
            return None

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

    def build_gradio_app(self):

        with gr.Blocks() as gr_app:
            with gr.Row():
                with gr.Column():
                    gr_input_image = gr.Image(label="Input Image")
                    gr_fg_mask = gr.Image(label="Foreground Mask")
                    gr_dilation = gr.Number(label="Forground Mask Dilation", precision=0, value=3, minimum=0, maximum=100)
                    generate_button = gr.Button("Submit")
                with gr.Column():
                    gr_bg = gr.Image(label="Background")

            generate_button.click(
                self.run_lama_inpainter,
                inputs=[gr_input_image, gr_fg_mask, gr_dilation],
                outputs=[gr_bg])

        return gr_app

    def start(self):

        gr_app = self.build_gradio_app()
        gr_app = gr_app.queue()

        app = FastAPI()
        app = gr.mount_gradio_app(app, gr_app, path=self.netpath)

        try:
            uvicorn.run(app, host="0.0.0.0", port=self.port)
        except KeyboardInterrupt:
            sys.exit()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--netpath', type=str, default='/lama_inpainter')
    parser.add_argument('--port', type=int, default=6008)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    server = LamaInpainterWebapp(netpath=args.netpath, port=args.port, device=args.device)
    server.start()
