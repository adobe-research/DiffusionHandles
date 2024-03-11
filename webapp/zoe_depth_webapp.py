import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import sys
import argparse

import torch
import numpy.typing as npt
import gradio as gr
from gradio_hdrimage import HDRImage
from fastapi import FastAPI
import uvicorn

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

from utils import crop_and_resize

class ZoeDepthWebapp:
    def __init__(self, netpath: str, port: int, device: str = 'cuda:0'):
        self.netpath = netpath
        self.port = port
        self.img_res = 512

        conf = get_config("zoedepth_nk", "infer")
        self.depth_estimator = build_model(conf)
        self.depth_estimator.to(device)

    def run_zoe_depth(self, img: npt.NDArray = None):

        print('run_zoe_depth')

        if img is None:
            return None

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

    def build_gradio_app(self):

        with gr.Blocks() as gr_app:
            with gr.Row():
                with gr.Column():
                    gr_input_image = gr.Image(label="Input Image")
                    generate_button = gr.Button("Submit")
                with gr.Column():
                    gr_depth = HDRImage(label="Depth")

            generate_button.click(
                self.run_zoe_depth,
                inputs=[gr_input_image],
                outputs=[gr_depth])

        return gr_app

    def start(self):

        gr_app = self.build_gradio_app()
        gr_app = gr_app.queue()

        app = FastAPI()
        app = gr.mount_gradio_app(app, gr_app, path=self.netpath)

        try:
            uvicorn.run(app, host="0.0.0.0", port=self.port)
        except KeyboardInterrupt:
            del self.diff_handles
            sys.exit()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--netpath', type=str, default='/zoe_depth')
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    server = ZoeDepthWebapp(netpath=args.netpath, port=args.port, device=args.device)
    server.start()
