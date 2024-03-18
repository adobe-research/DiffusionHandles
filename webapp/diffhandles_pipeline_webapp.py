import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import sys
import argparse
import tempfile
import time
import pathlib

import numpy as np
import numpy.typing as npt
import gradio as gr
from fastapi import FastAPI
import uvicorn
import gradio_client
import imageio

class DiffhandlesPipelineWebapp:
    def __init__(
            self, netpath: str, port: int, lama_inpainter_url: str, zoe_depth_url: str, diffhandles_url: str, timeout_seconds: float = None):
        self.netpath = netpath
        self.port = port
        self.lama_inpainter_client = gradio_client.Client(lama_inpainter_url, upload_files=True, download_files=True)
        self.zoe_depth_client = gradio_client.Client(zoe_depth_url, upload_files=True, download_files=True)
        self.diffhandles_client = gradio_client.Client(diffhandles_url, upload_files=True, download_files=True)
        self.timeout_seconds = timeout_seconds
        imageio.plugins.freeimage.download() # to load exr files

    def run_diffhandles_pipeline(
            self, prompt: str = None, img: npt.NDArray = None, fg_mask: npt.NDArray = None,
            rot_angle: float = 0.0, rot_axis_x: float = 0.0, rot_axis_y: float = 1.0, rot_axis_z: float = 0.0,
            trans_x: float = 0.0, trans_y: float = 0.0, trans_z: float = 0.0,
            fg_mask_dilation: int = 3):

        # print('run_diffhandles')

        if any(inp is None for inp in [prompt, img, fg_mask]):
            return None

        # print(prompt)
        # print(f'{img.shape} {img.dtype}')
        # print(f'{fg_mask.shape} {fg_mask.dtype}')
        # print(f'{depth.shape} {depth.dtype}')
        # print(f'{bg_depth.shape} {bg_depth.dtype}')

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.close()
            img_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.close()
            fg_mask_path = f.name

        imageio.imwrite(img_path, img)
        imageio.imwrite(fg_mask_path, fg_mask)

        # submit jobs to create the background and the foreground depth
        bg_job = self.lama_inpainter_client.submit(
            gradio_client.file(img_path), gradio_client.file(fg_mask_path), fg_mask_dilation)
        depth_job = self.zoe_depth_client.submit(
            gradio_client.file(img_path))
        job_time = 0
        bg_depth_job = None
        while not (bg_job.done() and depth_job.done()):

            # already submit job to create the background depth if the background is already available
            if bg_job.done():
                bg_path = bg_job.outputs()[0]
                bg_depth_job = self.zoe_depth_client.submit(
                    gradio_client.file(bg_path))

            time.sleep(0.1)
            job_time += 0.1
            if self.timeout_seconds is not None and job_time > self.timeout_seconds:
                raise TimeoutError("Image editing job timed out.")

        # get depth
        depth_path = depth_job.outputs()[0]
        # depth = imageio.imread(depth_path)

        # submit job to create the background depth
        if bg_depth_job is None:
            bg_path = bg_job.outputs()[0]
            bg_depth_job = self.zoe_depth_client.submit(
                gradio_client.file(bg_path))
        while not (bg_depth_job.done()):
            time.sleep(0.1)
            job_time += 0.1
            if self.timeout_seconds is not None and job_time > self.timeout_seconds:
                raise TimeoutError("Image editing job timed out.")

        # get background depth
        bg_depth_path = bg_depth_job.outputs()[0]

        edited_image_job = self.diffhandles_client.submit(
            prompt, gradio_client.file(img_path), gradio_client.file(fg_mask_path), gradio_client.file(depth_path), gradio_client.file(bg_depth_path),
            rot_angle, rot_axis_x, rot_axis_y, rot_axis_z,
            trans_x, trans_y, trans_z)

        while not (edited_image_job.done()):
            time.sleep(0.1)
            job_time += 0.1
            if self.timeout_seconds is not None and job_time > self.timeout_seconds:
                raise TimeoutError("Image editing job timed out.")

        edited_image_path = edited_image_job.outputs()[0]
        edited_image = np.asarray(imageio.imread(edited_image_path))

        # delete temporary files
        for temp_path in [img_path, fg_mask_path, depth_path, bg_depth_path, edited_image_path]:
            if pathlib.Path(temp_path).is_file():
                pathlib.Path(temp_path).unlink()

        return edited_image


    def build_gradio_app(self):

        with gr.Blocks() as gr_app:
            with gr.Row():
                gr.HTML(value="""
                        <h1>Diffusion Handles</h1>
                        <ol>
                        <li>Enter a text prompt that describes the input image.</li>
                        <li>Upload the input image.</li>
                        <li>Upload a mask (black&white image) of the region you want to edit.</li>
                        <li>Enter the transformation parameters (rotation angle, rotation axis, tranlation - translation is roughly in meters).</li>
                        <li>Submit to get the edited image.</li>
                        </ol>
                        Processing time is currently approx. 3 minutes. This will be improved in the future.
                        """
                        )
            with gr.Row():
                with gr.Column():
                    gr_text_prompt = gr.Textbox(label="Text Prompt", value="a sunflower in the garden")
                    gr_input_image = gr.Image(label="Input Image", value="data/sunflower/input.png")
                    gt_fg_mask = gr.Image(label="Foreground Mask", value="data/sunflower/mask.png")
                    gr_rot_angle = gr.Slider(label="Rotation Angle", value=40.0, minimum=-180.0, maximum=180.0, step=1.0)
                    gr_rot_axis_x = gr.Number(label="Rotation Axis X", value=0.0, minimum=-1.0, maximum=1.0)
                    gr_rot_axis_y = gr.Number(label="Rotation Axis Y", value=1.0, minimum=-1.0, maximum=1.0)
                    gr_rot_axis_z = gr.Number(label="Rotation Axis Z", value=0.0, minimum=-1.0, maximum=1.0)
                    gr_trans_x = gr.Number(label="Transl    ation X", value=0.0, minimum=-100.0, maximum=100.0)
                    gr_trans_y = gr.Number(label="Translation Y", value=0.0, minimum=-100.0, maximum=100.0)
                    gr_trans_z = gr.Number(label="Translation Z", value=0.0, minimum=-100.0, maximum=100.0)
                    gr_fg_mask_dilation = gr.Number(label="Forground Mask Dilation", precision=0, value=3, minimum=0, maximum=100)
                    generate_button = gr.Button("Submit")
                with gr.Column():
                    gr_edited_image = gr.Image(label="Edited Image")

            generate_button.click(
                self.run_diffhandles_pipeline,
                inputs=[
                    gr_text_prompt, gr_input_image, gt_fg_mask,
                    gr_rot_angle, gr_rot_axis_x, gr_rot_axis_y, gr_rot_axis_z, gr_trans_x, gr_trans_y, gr_trans_z,
                    gr_fg_mask_dilation],
                outputs=[
                    gr_edited_image])
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
    parser.add_argument('--netpath', type=str, default='/dh')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--lama_inpainter_url', type=str, default='http://localhost:6008/lama_inpainter')
    parser.add_argument('--zoe_depth_url', type=str, default='http://localhost:6007/zoe_depth')
    parser.add_argument('--diffhandles_url', type=str, default='http://localhost:6006/diffhandles')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    server = DiffhandlesPipelineWebapp(netpath=args.netpath, port=args.port, lama_inpainter_url=args.lama_inpainter_url, zoe_depth_url=args.zoe_depth_url, diffhandles_url=args.diffhandles_url)
    server.start()
