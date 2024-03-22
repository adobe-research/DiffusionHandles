import sys
import argparse

import torch
import torchvision
import numpy.typing as npt
import gradio as gr
from fastapi import FastAPI
import uvicorn

from lang_sam import LangSAM

from utils import crop_and_resize

class LangSAMSegmenterWebapp:
    def __init__(self, netpath: str, port: int, device: str = 'cuda:0'):
        self.netpath = netpath
        self.port = port
        self.img_res = 512

        self.segmenter = LangSAM()
        self.segmenter.sam.model.to(device)
        self.segmenter.device = device

    def run_langsam_segmenter(self, img: npt.NDArray = None, prompt: str = None):

        print('run_langsam_segmenter')

        if img is None or prompt is None:
            return None

        print(f'{img.shape} {img.dtype}')
        print(f'{prompt}')

        # prepare inputs (convert to torch tensors, crop, resize)
        img = torch.from_numpy(img).to(device=self.segmenter.device, dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        img = crop_and_resize(img=img, size=self.img_res)

        masks, boxes, prompts, logits = self.segmenter.predict(
            image_pil=torchvision.transforms.functional.to_pil_image(img[0]),
            text_prompt=prompt)
        mask = masks[[0, 0, 0], :, :].to(device=self.segmenter.device, dtype=torch.float32)

        print(mask.shape)

        mask = (mask * 255.0).permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()

        return mask

    def build_gradio_app(self):

        with gr.Blocks() as gr_app:
            with gr.Row():
                with gr.Column():
                    gr_input_image = gr.Image(label="Input Image", value="data/sunflower/input.png")
                    gr_segment_prompt = gr.Textbox(label="Segment Prompt", value="sunflower")
                    generate_button = gr.Button("Submit")
                with gr.Column():
                    gr_bg = gr.Image(label="Background")

            generate_button.click(
                self.run_langsam_segmenter,
                inputs=[gr_input_image, gr_segment_prompt],
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
    parser.add_argument('--netpath', type=str, default='/langsam_segmenter')
    parser.add_argument('--port', type=int, default=6010)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    server = LangSAMSegmenterWebapp(netpath=args.netpath, port=args.port, device=args.device)
    server.start()
