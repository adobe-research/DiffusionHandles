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

from diffhandles import DiffusionHandles

from utils import crop_and_resize

class DiffhandlesWebapp:
    def __init__(self, netpath: str, port: int, config_path: str = None, device: str = 'cuda:0'):
        self.netpath = netpath
        self.port = port
        self.config_path = config_path
        self.diff_handles = DiffusionHandles(conf_path=config_path)
        self.diff_handles.to(device)
        self.img_res = 512

    def run_diffhandles(
            self, prompt: str = None, img: npt.NDArray = None, fg_mask: npt.NDArray = None,
            depth: npt.NDArray = None, bg_depth: npt.NDArray = None,
            rot_angle: float = 0.0, rot_axis_x: float = 0.0, rot_axis_y: float = 1.0, rot_axis_z: float = 0.0,
            trans_x: float = 0.0, trans_y: float = 0.0, trans_z: float = 0.0):

        # print('run_diffhandles')

        if any(inp is None for inp in [prompt, img, fg_mask, depth, bg_depth]):
            return None

        # print(prompt)
        # print(f'{img.shape} {img.dtype}')
        # print(f'{fg_mask.shape} {fg_mask.dtype}')
        # print(f'{depth.shape} {depth.dtype}')
        # print(f'{bg_depth.shape} {bg_depth.dtype}')

        # prepare inputs (convert to torch tensors, etc.)
        img = torch.from_numpy(img).to(device=self.diff_handles.device, dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        fg_mask = torch.from_numpy(fg_mask).to(device=self.diff_handles.device, dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        depth = torch.from_numpy(depth).to(device=self.diff_handles.device, dtype=torch.float32)[None, None, ...]
        bg_depth = torch.from_numpy(bg_depth).to(device=self.diff_handles.device, dtype=torch.float32)[None, None, ...]
        img = crop_and_resize(img=img, size=self.img_res)
        if fg_mask.shape[1] > 1:
            fg_mask = fg_mask.mean(dim=1, keepdim=True) # average channels
        fg_mask = crop_and_resize(img=fg_mask, size=self.img_res)
        fg_mask = (fg_mask>0.5).to(device=self.diff_handles.device, dtype=torch.float32)
        depth = crop_and_resize(img=depth, size=self.img_res)
        bg_depth = crop_and_resize(img=bg_depth, size=self.img_res)
        rot_axis = torch.tensor([rot_axis_x, rot_axis_y, rot_axis_z], dtype=torch.float32, device=self.diff_handles.device)
        translation = torch.tensor([trans_x, trans_y, trans_z], dtype=torch.float32, device=self.diff_handles.device)

        # print(prompt)
        # print(img.shape)
        # print(fg_mask.shape)
        # print(depth.shape)
        # print(bg_depth.shape)
        # print(f'rot angle {rot_angle}')
        # print(f'rot axis {rot_axis}')
        # print(f'translation {translation}')

        (bg_depth, inverted_null_text, inverted_noise, activations, activations2, activations3, latent_image
         ) = self.diff_handles.set_foreground(
            img, depth, prompt, fg_mask, bg_depth)
        (edited_img, edited_disparity
         ) = self.diff_handles.transform_foreground(
            depth=depth, prompt=prompt,
            fg_mask=fg_mask, bg_depth=bg_depth,
            inverted_null_text=inverted_null_text, inverted_noise=inverted_noise,
            activations=activations, activations2=activations2, activations3=activations3,
            rot_angle=rot_angle, rot_axis=rot_axis, translation=translation,
            use_input_depth_normalization=False)

        edited_img = (edited_img * 255.0)[0].permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()

        return edited_img

    def build_gradio_app(self):

        with gr.Blocks() as gr_app:
            with gr.Row():
                with gr.Column():
                    gr_text_prompt = gr.Textbox(label="Text Prompt", value="a sunflower in the garden")
                    gr_input_image = gr.Image(label="Input Image", value="data/sunflower/input.png")
                    gr_fg_mask = gr.Image(label="Foreground Mask", value="data/sunflower/mask.png")
                    gt_depth = HDRImage(label="Depth", value="data/sunflower/depth.exr")
                    gt_bg_depth = HDRImage(label="Background Depth")
                    gr_rot_angle = gr.Slider(label="Rotation Angle", value=40.0, minimum=-180.0, maximum=180.0, step=1.0)
                    gr_rot_axis_x = gr.Number(label="Rotation Axis X", value=0.0, minimum=-1.0, maximum=1.0)
                    gr_rot_axis_y = gr.Number(label="Rotation Axis Y", value=1.0, minimum=-1.0, maximum=1.0)
                    gr_rot_axis_z = gr.Number(label="Rotation Axis Z", value=0.0, minimum=-1.0, maximum=1.0)
                    gr_trans_x = gr.Number(label="Translation X", value=0.0, minimum=-100.0, maximum=100.0)
                    gr_trans_y = gr.Number(label="Translation Y", value=0.0, minimum=-100.0, maximum=100.0)
                    gr_trans_z = gr.Number(label="Translation Z", value=0.0, minimum=-100.0, maximum=100.0)
                    generate_button = gr.Button("Submit")
                with gr.Column():
                    gr_edited_image = gr.Image(label="Edited Image")

            generate_button.click(
                self.run_diffhandles,
                inputs=[
                    gr_text_prompt, gr_input_image, gr_fg_mask, gt_depth, gt_bg_depth,
                    gr_rot_angle, gr_rot_axis_x, gr_rot_axis_y, gr_rot_axis_z, gr_trans_x, gr_trans_y, gr_trans_z],
                outputs=[
                    gr_edited_image])

        # gr_app = gr.Interface(
        #     fn=self.run_diffhandles,
        #     inputs=[
        #         gr.Textbox(label="Text Prompt"),
        #         gr.Image(label="Input Image"),
        #         gr.Image(label="Foreground Mask"),
        #         HDRImage(label="Depth"),
        #         HDRImage(label="Background Depth"),
        #         gr.Slider(label="Rotation Angle", value=0.0, minimum=-180.0, maximum=180.0, step=1.0),
        #         gr.Number(label="Rotation Axis X", value=0.0, minimum=-1.0, maximum=1.0),
        #         gr.Number(label="Rotation Axis Y", value=1.0, minimum=-1.0, maximum=1.0),
        #         gr.Number(label="Rotation Axis Z", value=0.0, minimum=-1.0, maximum=1.0),
        #         gr.Number(label="Translation X", value=0.0, minimum=-100.0, maximum=100.0),
        #         gr.Number(label="Translation Y", value=0.0, minimum=-100.0, maximum=100.0),
        #         gr.Number(label="Translation Z", value=0.0, minimum=-100.0, maximum=100.0),
        #         ],
        #     outputs=["image"])
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
    parser.add_argument('--netpath', type=str, default='/diffhandles')
    parser.add_argument('--port', type=int, default=6006)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--config_path', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    server = DiffhandlesWebapp(netpath=args.netpath, port=args.port, config_path=args.config_path, device=args.device)
    server.start()
