import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import sys
import argparse
import tempfile
import time
import pathlib

import numpy as np
import numpy.typing as npt
import scipy.ndimage
import torch
import gradio as gr
from fastapi import FastAPI
import uvicorn
import gradio_client
import imageio
from diffhandles.depth_transform import transform_depth
from diffhandles.utils import solve_laplacian_depth

from gradio_job_manager import GradioJob, GradioJobManager
from diffhandles.guided_stable_diffuser import GuidedStableDiffuser
from utils import crop_and_resize

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

    def select_foreground(self, img: npt.NDArray, object_prompt: str):
        if any(inp is None for inp in [img, object_prompt]):
            return None
        
        job_manager = GradioJobManager()
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.close()
            img_path = f.name
        imageio.imwrite(img_path, img)

        fg_mask_path = None
        fg_mask = None
        def read_fg_mask(jobs, job_manager):
            nonlocal fg_mask, fg_mask_path
            fg_mask_path = jobs[0].outputs()[0]
            fg_mask = np.asarray(imageio.imread(fg_mask_path))

        fg_mask_job = GradioJob(job=self.langsam_segmenter_client.submit(
            # gradio_client.file(img_path), # for gradio version >= 4.21
            img_path,
            object_prompt))
        
        job_manager.add_job(fg_mask_job)
        job_manager.add_callback(func=read_fg_mask, when_jobs_done=[fg_mask_job])

        job_manager.run()

        # delete temporary files
        for temp_path in [img_path, fg_mask_path]:
            if pathlib.Path(temp_path).is_file():
                pathlib.Path(temp_path).unlink()

        return fg_mask
    
    def preview_edit(
            self, img: npt.NDArray = None, fg_mask: npt.NDArray = None,
            rot_angle: float = 0.0, rot_axis_x: float = 0.0, rot_axis_y: float = 1.0, rot_axis_z: float = 0.0,
            trans_x: float = 0.0, trans_y: float = 0.0, trans_z: float = 0.0,
            fg_mask_dilation: int = 3):

        if any(inp is None for inp in [img, fg_mask]):
            return None

        job_manager = GradioJobManager()

        # print(prompt)
        # print(f'{img.shape} {img.dtype}')
        # print(f'{fg_mask.shape} {fg_mask.dtype}')
        # print(f'{depth.shape} {depth.dtype}')
        # print(f'{bg_depth.shape} {bg_depth.dtype}')

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.close()
            img_path = f.name
        imageio.imwrite(img_path, img)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.close()
            fg_mask_path = f.name
        imageio.imwrite(fg_mask_path, fg_mask)

        depth_job = GradioJob(job=self.zoe_depth_client.submit(
            # gradio_client.file(img_path) # for gradio version >= 4.21
            img_path))
        bg_job = GradioJob(job=self.lama_inpainter_client.submit(
            # gradio_client.file(img_path), gradio_client.file(fg_mask_path), # for gradio version >= 4.21
            img_path, fg_mask_path,
            fg_mask_dilation))

        depth_path = None
        bg_depth_path = None
        depth = None
        bg_depth = None
        def read_depth(jobs, job_manager):
            nonlocal depth_path, bg_depth_path, depth, bg_depth
            depth_path = jobs[0].outputs()[0]
            bg_depth_path = jobs[1].outputs()[0]
            depth = np.asarray(imageio.imread(depth_path))
            bg_depth = np.asarray(imageio.imread(bg_depth_path))

        bg_path = None
        def run_bg_depth(jobs, job_manager):
            nonlocal bg_path
            bg_path = jobs[0].outputs()[0]
            bg_depth_job = GradioJob(job=self.zoe_depth_client.submit(
                # gradio_client.file(bg_path) # for gradio version >= 4.21
                bg_path))
            job_manager.add_job(bg_depth_job)
            job_manager.add_callback(func=read_depth, when_jobs_done=[depth_job, bg_depth_job])
        
        job_manager.add_job(depth_job)
        job_manager.add_job(bg_job)
        job_manager.add_callback(func=run_bg_depth, when_jobs_done=[bg_job])

        job_manager.run()
        
        # delete temporary files
        for temp_path in [img_path, fg_mask_path, bg_path, depth_path, bg_depth_path]:
            if pathlib.Path(temp_path).is_file():
                pathlib.Path(temp_path).unlink()

        img_res = 512
        depth = torch.from_numpy(depth).to(dtype=torch.float32, device='cpu')[None, None, ...]
        fg_mask = torch.from_numpy(fg_mask).to(dtype=torch.float32, device='cpu').permute(2, 0, 1)[None, ...] / 255.0
        bg_depth = torch.from_numpy(bg_depth).to(dtype=torch.float32, device='cpu')[None, None, ...]
        depth = crop_and_resize(img=depth, size=img_res)
        bg_depth = crop_and_resize(img=bg_depth, size=img_res)
        if fg_mask.shape[1] > 1:
            fg_mask = fg_mask.mean(dim=1, keepdim=True) # average channels
        fg_mask = crop_and_resize(img=fg_mask, size=img_res)
        fg_mask = (fg_mask>0.5).to(dtype=torch.float32, device='cpu')
        rot_axis = torch.tensor([rot_axis_x, rot_axis_y, rot_axis_z], dtype=torch.float32, device='cpu')
        translation = torch.tensor([trans_x, trans_y, trans_z], dtype=torch.float32, device='cpu')

        # infilling bg depth hole
        print('infilling bg depth hole ...')
        bg_depth = solve_laplacian_depth(
            depth[0, 0].cpu().numpy(),
            bg_depth[0, 0].cpu().numpy(),
            scipy.ndimage.binary_dilation(fg_mask[0, 0].cpu().numpy(), iterations=15))
        bg_depth = torch.from_numpy(bg_depth)[None, None, ...]

        # transforming depth
        print('transforming depth ...')
        edited_disparity, target_mask, correspondences, raw_edited_depth = transform_depth(
            depth=depth, bg_depth=bg_depth, fg_mask=fg_mask,
            intrinsics=GuidedStableDiffuser.get_depth_intrinsics(h=img_res, w=img_res),
            rot_angle=rot_angle, rot_axis=rot_axis, translation=translation,
            use_input_depth_normalization=False)
        
        print('done')
        
        edited_disparity = (edited_disparity - edited_disparity.min())/(edited_disparity.max() - edited_disparity.min())
        edited_disparity = (edited_disparity[0, 0].detach().cpu().numpy()*255).round().astype("uint8")
        
        return edited_disparity


    def run_diffhandles_pipeline(
            self, prompt: str = None, img: npt.NDArray = None, fg_mask: npt.NDArray = None,
            rot_angle: float = 0.0, rot_axis_x: float = 0.0, rot_axis_y: float = 1.0, rot_axis_z: float = 0.0,
            trans_x: float = 0.0, trans_y: float = 0.0, trans_z: float = 0.0,
            fg_mask_dilation: int = 3):

        # print('run_diffhandles')

        if any(inp is None for inp in [prompt, img, fg_mask]):
            return None

        job_manager = GradioJobManager()

        # print(prompt)
        # print(f'{img.shape} {img.dtype}')
        # print(f'{fg_mask.shape} {fg_mask.dtype}')
        # print(f'{depth.shape} {depth.dtype}')
        # print(f'{bg_depth.shape} {bg_depth.dtype}')

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.close()
            img_path = f.name
        imageio.imwrite(img_path, img)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.close()
            fg_mask_path = f.name
        imageio.imwrite(fg_mask_path, fg_mask)

        depth_job = GradioJob(job=self.zoe_depth_client.submit(
            # gradio_client.file(img_path) # for gradio version >= 4.21
            img_path))
        bg_job = GradioJob(job=self.lama_inpainter_client.submit(
            # gradio_client.file(img_path), gradio_client.file(fg_mask_path), # for gradio version >= 4.21
            img_path, fg_mask_path,
            fg_mask_dilation))

        edited_image_path = None
        edited_image = None
        def read_edited_image(jobs, job_manager):
            nonlocal edited_image_path, edited_image
            edited_image_path = jobs[0].outputs()[0]
            edited_image = np.asarray(imageio.imread(edited_image_path))
        
        depth_path = None
        bg_depth_path = None
        def run_diffhandles(jobs, job_manager):
            nonlocal depth_path, bg_depth_path
            depth_path = jobs[0].outputs()[0]
            bg_depth_path = jobs[1].outputs()[0]
            diffhandles_job = GradioJob(job=self.diffhandles_client.submit(
                prompt,
                # gradio_client.file(img_path), gradio_client.file(fg_mask_path), gradio_client.file(depth_path), gradio_client.file(bg_depth_path), # for gradio version >= 4.21
                img_path, fg_mask_path, depth_path, bg_depth_path,
                rot_angle, rot_axis_x, rot_axis_y, rot_axis_z,
                trans_x, trans_y, trans_z))
            job_manager.add_job(diffhandles_job)
            job_manager.add_callback(func=read_edited_image, when_jobs_done=[diffhandles_job])

        bg_path = None
        def run_bg_depth(jobs, job_manager):
            nonlocal bg_path
            bg_path = jobs[0].outputs()[0]
            bg_depth_job = GradioJob(job=self.zoe_depth_client.submit(
                # gradio_client.file(bg_path) # for gradio version >= 4.21
                bg_path))
            job_manager.add_job(bg_depth_job)
            job_manager.add_callback(func=run_diffhandles, when_jobs_done=[depth_job, bg_depth_job])
        
        job_manager.add_job(depth_job)
        job_manager.add_job(bg_job)
        job_manager.add_callback(func=run_bg_depth, when_jobs_done=[bg_job])
        
        job_manager.run()
        
        # delete temporary files
        for temp_path in [img_path, fg_mask_path, bg_path, depth_path, bg_depth_path, edited_image_path]:
            if pathlib.Path(temp_path).is_file():
                pathlib.Path(temp_path).unlink()

        return edited_image


    def build_gradio_app(self):

        with gr.Blocks() as gr_app:
            with gr.Row():
                gr.HTML(value="""
                        <h1>Diffusion Handles</h1>
                        """
                        )
            with gr.Row():
                    gr.HTML(value="""
                            <h2>Step 1: Select Object</h2>
                            <ol>
                            <li>Enter a text prompt that describes the input image.</li>
                            <li>Upload the input image.</li>
                            <li>Enter a prompt that selects the object to be edited.</li>
                            <li>'Select Object' to perform selection and view a mask of the selected object.</li>
                            </ol>
                            """
                            )
            with gr.Row():
                with gr.Column():
                    gr_text_prompt = gr.Textbox(label="Image Prompt", value="a sunflower in the garden")
                    gr_input_image = gr.Image(label="Input Image", value="data/sunflower/input.png")
                    gr_object_prompt = gr.Textbox(label="Select Object with a Prompt", value="sunflower")
                    gr_object_button = gr.Button("Select Object")
                with gr.Column():
                    gr_fg_mask = gr.Image(label="Object Mask", value="data/sunflower/mask.png")
            
            with gr.Row():
                    gr.HTML(value="""
                            <h2>Step 2: Edit Object</h2>
                            <li>Enter the transformation parameters (rotation angle, rotation axis, tranlation - translation is roughly in meters).</li>
                            <li>'Preview Edit' to get a fast preview of the depth of the edited image.  (Processing time ~ 50 seconds. This will be improved in the future.)</li>
                            <li>'Perform Edit' to get the edited image. (Processing time ~ 3 min. This will be improved in the future.)</li>
                            """
                            )
            with gr.Row():
                with gr.Column():
                    gr_rot_angle = gr.Slider(label="Rotation Angle", value=40.0, minimum=-180.0, maximum=180.0, step=1.0)
                    gr_rot_axis_x = gr.Number(label="Rotation Axis X", value=0.0, minimum=-1.0, maximum=1.0)
                    gr_rot_axis_y = gr.Number(label="Rotation Axis Y", value=1.0, minimum=-1.0, maximum=1.0)
                    gr_rot_axis_z = gr.Number(label="Rotation Axis Z", value=0.0, minimum=-1.0, maximum=1.0)
                    gr_trans_x = gr.Number(label="Transl    ation X", value=0.0, minimum=-100.0, maximum=100.0)
                    gr_trans_y = gr.Number(label="Translation Y", value=0.0, minimum=-100.0, maximum=100.0)
                    gr_trans_z = gr.Number(label="Translation Z", value=0.0, minimum=-100.0, maximum=100.0)
                    gr_fg_mask_dilation = gr.Number(label="Forground Mask Dilation", precision=0, value=3, minimum=0, maximum=100)
                    with gr.Row():
                        with gr.Column():
                            gr_preview_edit_button = gr.Button("Preview Edit")
                        with gr.Column():
                            gr_edit_button = gr.Button("Perform Edit")
                with gr.Column():
                    gr_edited_image = gr.Image(label="Edited Image")

            gr_object_button.click(
                self.select_foreground,
                inputs=[
                    gr_input_image, gr_object_prompt],
                outputs=[
                    gr_fg_mask])

            gr_preview_edit_button.click(
                self.preview_edit,
                inputs=[
                    gr_input_image, gr_fg_mask,
                    gr_rot_angle, gr_rot_axis_x, gr_rot_axis_y, gr_rot_axis_z, gr_trans_x, gr_trans_y, gr_trans_z],
                outputs=[
                    gr_edited_image]
            )
            
            gr_edit_button.click(
                self.run_diffhandles_pipeline,
                inputs=[
                    gr_text_prompt, gr_input_image, gr_fg_mask,
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
