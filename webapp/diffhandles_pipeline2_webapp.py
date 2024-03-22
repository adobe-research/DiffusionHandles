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

from utils import GradioJob

# Example usage:
# python diffhandles_pipeline_webapp.py --port 8888 --netpath /g3i-demo/diffhandles-demo/dh --lama_inpainter_url http://localhost:8891/g3i-demo/diffhandles-demo/lama_inpainter --zoe_depth_url http://localhost:8890/g3i-demo/diffhandles-demo/zoe_depth --diffhandles_url http://localhost:8889/g3i-demo/diffhandles-demo/diffhandles

class DiffhandlesPipelineWebapp:
    def __init__(
            self, netpath: str, port: int, langsam_segmenter_url: str, lama_inpainter_url: str, zoe_depth_url: str, diffhandles_url: str, timeout_seconds: float = None):
        self.netpath = netpath
        self.port = port
        self.langsam_segmenter_client = gradio_client.Client(langsam_segmenter_url, upload_files=True, download_files=True)
        self.lama_inpainter_client = gradio_client.Client(lama_inpainter_url, upload_files=True, download_files=True)
        self.zoe_depth_client = gradio_client.Client(zoe_depth_url, upload_files=True, download_files=True)
        self.diffhandles_client = gradio_client.Client(diffhandles_url, upload_files=True, download_files=True)
        self.timeout_seconds = timeout_seconds
        imageio.plugins.freeimage.download() # to load exr files

    def run_diffhandles_pipeline(
            self, prompt: str = None, img: npt.NDArray = None, object_prompt: str = None,
            rot_angle: float = 0.0, rot_axis_x: float = 0.0, rot_axis_y: float = 1.0, rot_axis_z: float = 0.0,
            trans_x: float = 0.0, trans_y: float = 0.0, trans_z: float = 0.0,
            fg_mask_dilation: int = 3):

        # print('run_diffhandles')

        if any(inp is None for inp in [prompt, img, object_prompt]):
            return None

        # print(prompt)
        # print(f'{img.shape} {img.dtype}')
        # print(f'{fg_mask.shape} {fg_mask.dtype}')
        # print(f'{depth.shape} {depth.dtype}')
        # print(f'{bg_depth.shape} {bg_depth.dtype}')

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.close()
            img_path = f.name

        imageio.imwrite(img_path, img)

        active_jobs = {}
        
        active_jobs['fg_mask'] = GradioJob(job=self.langsam_segmenter_client.submit(
            # gradio_client.file(img_path), # for gradio version >= 4.21
            img_path,
            object_prompt))
        active_jobs['depth'] = GradioJob(job=self.zoe_depth_client.submit(
            # gradio_client.file(img_path) # for gradio version >= 4.21
            img_path))

        while len(active_jobs) > 0:

            if 'fg_mask' in active_jobs and active_jobs['fg_mask'].job.done():
                fg_mask_path = active_jobs['fg_mask'].job.outputs()[0]
                active_jobs.pop('fg_mask')
                
                active_jobs['bg'] = GradioJob(job=self.lama_inpainter_client.submit(
                    # gradio_client.file(img_path), gradio_client.file(fg_mask_path), # for gradio version >= 4.21
                    img_path, fg_mask_path,
                    fg_mask_dilation))

            if 'bg' in active_jobs and active_jobs['bg'].job.done():
                bg_path = active_jobs['bg'].job.outputs()[0]
                active_jobs.pop('bg')
                
                active_jobs['bg_depth'] = GradioJob(job=self.zoe_depth_client.submit(
                    # gradio_client.file(img_path) # for gradio version >= 4.21
                    bg_path))
            
            if (('depth' in active_jobs and active_jobs['depth'].job.done()) and
                ('bg_depth' in active_jobs and active_jobs['bg_depth'].job.done())):
                depth_path = active_jobs['depth'].job.outputs()[0]
                active_jobs.pop('depth')
                bg_depth_path = active_jobs['bg_depth'].job.outputs()[0]
                active_jobs.pop('bg_depth')
                
                active_jobs['edited_image'] = GradioJob(job=self.diffhandles_client.submit(
                    prompt,
                    # gradio_client.file(img_path), gradio_client.file(fg_mask_path), gradio_client.file(depth_path), gradio_client.file(bg_depth_path), # for gradio version >= 4.21
                    img_path, fg_mask_path, depth_path, bg_depth_path,
                    rot_angle, rot_axis_x, rot_axis_y, rot_axis_z,
                    trans_x, trans_y, trans_z))

            if 'edited_image' in active_jobs and active_jobs['edited_image'].job.done():
                edited_image_path = active_jobs['edited_image'].job.outputs()[0]
                active_jobs.pop('edited_image')

            if len(active_jobs) > 0:
                time.sleep(0.1)
                for job in active_jobs.values():
                    job.time += 0.1
                    if self.timeout_seconds is not None and job.time > self.timeout_seconds:
                        raise TimeoutError("Job timed out.")
        
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
                        <li>Enter a prompt that selects the object to be edited.</li>
                        <li>Enter the transformation parameters (rotation angle, rotation axis, tranlation - translation is roughly in meters).</li>
                        <li>Submit to get the edited image.</li>
                        </ol>
                        Processing time is currently approx. 3 minutes. This will be improved in the future.
                        """
                        )
            with gr.Row():
                with gr.Column():
                    gr.HTML(value="""
                            <h1>Input Image & Prompt</h1>
                            """
                            )
                    gr_text_prompt = gr.Textbox(label="Image Prompt", value="a sunflower in the garden")
                    gr_input_image = gr.Image(label="Input Image", value="data/sunflower/input.png")
                    gr.HTML(value="""
                            <h1>Edit Parameters</h1>
                            """
                            )
                    gr_object_prompt = gr.Textbox(label="Select Object with a Prompt", value="sunflower")
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
                    gr_text_prompt, gr_input_image, gr_object_prompt,
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
    parser.add_argument('--langsam_segmenter_url', type=str, default='http://localhost:6010/langsam_segmenter')
    parser.add_argument('--lama_inpainter_url', type=str, default='http://localhost:6008/lama_inpainter')
    parser.add_argument('--zoe_depth_url', type=str, default='http://localhost:6007/zoe_depth')
    parser.add_argument('--diffhandles_url', type=str, default='http://localhost:6006/diffhandles')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    server = DiffhandlesPipelineWebapp(
        netpath=args.netpath, port=args.port,
        langsam_segmenter_url=args.langsam_segmenter_url, lama_inpainter_url=args.lama_inpainter_url,
        zoe_depth_url=args.zoe_depth_url, diffhandles_url=args.diffhandles_url)
    server.start()
