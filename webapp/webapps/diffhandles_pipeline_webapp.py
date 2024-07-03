import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import argparse
import tempfile
import pathlib
import time
import json

import numpy as np
import numpy.typing as npt
import scipy.ndimage
import torch
import gradio as gr
from gradio_hdrimage import HDRImage
import gradio_client
import imageio.plugins as imageio_plugins
import imageio.v3 as imageio
from omegaconf import OmegaConf, DictConfig
from diffhandles.depth_transform import transform_depth, depth_to_mesh, transform_points
from diffhandles.utils import solve_laplacian_depth
from diffhandles.mesh_io import save_mesh
from diffhandles.guided_stable_diffuser import GuidedStableDiffuser
from diffhandles.renderer import Camera
from diffhandles.pytorch3d_renderer import PyTorch3DRenderer, PyTorch3DRendererArgs

from gradio_job_manager import GradioJob, GradioJobManager
from utils import crop_and_resize
from gradio_webapp import GradioWebapp

# Example usage:
# python diffhandles_pipeline_webapp.py --port 8888 --netpath /g3i-demo/diffhandles-demo/dh --lama_inpainter_url http://localhost:8891/g3i-demo/diffhandles-demo/lama_inpainter --zoe_depth_url http://localhost:8890/g3i-demo/diffhandles-demo/zoe_depth --diffhandles_url http://localhost:8889/g3i-demo/diffhandles-demo/diffhandles

class DiffhandlesPipelineWebapp(GradioWebapp):

    def __init__(
            self, netpath: str, port: int, text2img_url: str, foreground_selector_url: str, foreground_remover_url: str, depth_estimator_url: str, diffhandles_url: str,
            default_edit_path: str = None, timeout_seconds: float = None, debug_images: bool = False, return_meshes: bool = False, device: str = 'cuda:0'):

        super().__init__(netpath=netpath, port=port)

        if default_edit_path is None:
            default_edit_path = 'data/toy_cubes/edit_001'

        self.text2img_client = gradio_client.Client(text2img_url, download_files=True)
        self.foreground_selector_client = gradio_client.Client(foreground_selector_url, download_files=True)
        self.foreground_remover_client = gradio_client.Client(foreground_remover_url, download_files=True)
        self.depth_estimator_client = gradio_client.Client(depth_estimator_url, download_files=True)
        self.diffhandles_client = gradio_client.Client(diffhandles_url, download_files=True)
        self.default_edit_path = default_edit_path
        self.timeout_seconds = timeout_seconds
        self.debug_images = debug_images
        self.return_meshes = return_meshes
        self.device = torch.device(device)
        self.img_res = 512

        imageio_plugins.freeimage.download() # to load exr files

        # paths of generated temporary files that should be deleted when the server is stopped
        self.temp_file_paths = []

    def delete_temp_files(self):
        for temp_file in self.temp_file_paths:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        self.temp_file_paths = []

    def delete_old_temp_files(self):
        """
        Remove temporary files older than a day.
        """
        for temp_file in list(self.temp_file_paths): # list will be mutated
            if time.time() - os.path.getmtime(temp_file) > 60*60*24:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                self.temp_file_paths.remove(temp_file)

    def cleanup(self):
        self.delete_temp_files()

    def generate_input(self, prompt: str):
        if prompt is None:
            raise ValueError('Some inputs are missing.')

        job_manager = GradioJobManager()

        text2img_job = GradioJob(job=self.text2img_client.submit(prompt))
        job_manager.add_job(text2img_job)

        input_image = None
        def read_img(jobs, job_manager):
            nonlocal input_image
            input_image_path = jobs[0].outputs()[0]
            input_image = np.asarray(imageio.imread(input_image_path))

        job_manager.add_callback(func=read_img, when_jobs_done=[text2img_job])

        job_manager.run()

        return input_image

    def set_input_image(self, prompt: str, img: npt.NDArray) -> tuple[str, npt.NDArray]:

        if any(inp is None for inp in [prompt, img]):
            raise ValueError('Some inputs are missing.')

        self.delete_old_temp_files()

        # pre-process inputs
        img = torch.from_numpy(img).to(dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        img = crop_and_resize(img=img, size=self.img_res)
        img = (img * 255.0)[0].permute(1, 2, 0).to(dtype=torch.uint8).numpy()

        job_manager = GradioJobManager()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.close()
            img_path = f.name
        imageio.imwrite(img_path, img)

        depth_job = GradioJob(job=self.depth_estimator_client.submit(
            gradio_client.file(img_path)))

        input_image_identity_path = None
        def read_input_image_identity(jobs, job_manager):
            nonlocal input_image_identity_path
            input_image_identity_path = jobs[0].outputs()[0]
            self.temp_file_paths.append(input_image_identity_path)

        depth_path = None
        depth = None
        def run_set_input_image(jobs, job_manager):
            nonlocal depth_path, depth
            depth_path = jobs[0].outputs()[0]
            depth = np.asarray(imageio.imread(depth_path))
            set_input_job = GradioJob(job=self.diffhandles_client.submit(
                prompt,
                gradio_client.file(depth_path), gradio_client.file(img_path),
                api_name="/set_input_image"))
            job_manager.add_job(set_input_job)
            job_manager.add_callback(func=read_input_image_identity, when_jobs_done=[set_input_job])

        job_manager.add_job(depth_job)
        job_manager.add_callback(func=run_set_input_image, when_jobs_done=[depth_job])

        job_manager.run()

        # delete temporary files
        for temp_path in [img_path, depth_path]:
            if pathlib.Path(temp_path).is_file():
                pathlib.Path(temp_path).unlink()
        
        step1_completed = True

        return input_image_identity_path, depth, step1_completed

    def select_foreground(self, img: npt.NDArray, object_prompt: str):

        if any(inp is None for inp in [img, object_prompt]):
            raise ValueError('Some inputs are missing.')

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

        fg_mask_job = GradioJob(job=self.foreground_selector_client.submit(
            gradio_client.file(img_path),
            object_prompt))

        job_manager.add_job(fg_mask_job)
        job_manager.add_callback(func=read_fg_mask, when_jobs_done=[fg_mask_job])

        job_manager.run()

        # delete temporary files
        for temp_path in [img_path, fg_mask_path]:
            if pathlib.Path(temp_path).is_file():
                pathlib.Path(temp_path).unlink()

        return fg_mask

    def set_foreground(self, prompt: str, img: npt.NDArray, input_image_identity_path:str, depth: npt.NDArray, fg_mask: npt.NDArray, fg_mask_dilation: int) -> npt.NDArray:

        if any(x is None for x in [input_image_identity_path, depth]):
            # Results of the previous steps are not available, compute the previous steps
            input_image_identity_path, depth, step1_completed = self.set_input_image(prompt, img)
        else:
            # Results of the previous steps are available
            step1_completed = True

        # pre-process inputs
        img = torch.from_numpy(img).to(dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        img = crop_and_resize(img=img, size=self.img_res)
        img = (img * 255.0)[0].permute(1, 2, 0).to(dtype=torch.uint8).numpy()
        depth = torch.from_numpy(depth).to(dtype=torch.float32, device=self.device)[None, None, ...]
        depth = crop_and_resize(img=depth, size=self.img_res)
        depth = depth[0].detach().cpu().permute(1, 2, 0).numpy()[..., 0]
        fg_mask = torch.from_numpy(fg_mask).to(dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        fg_mask = crop_and_resize(img=fg_mask, size=self.img_res)
        fg_mask = (fg_mask * 255.0)[0].permute(1, 2, 0).to(dtype=torch.uint8).numpy()

        job_manager = GradioJobManager()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.close()
            img_path = f.name
        imageio.imwrite(img_path, img)
        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
            f.close()
            depth_path = f.name
        imageio.imwrite(depth_path, depth)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.close()
            fg_mask_path = f.name
        imageio.imwrite(fg_mask_path, fg_mask)

        bg_job = GradioJob(job=self.foreground_remover_client.submit(
            gradio_client.file(img_path), gradio_client.file(fg_mask_path),
            fg_mask_dilation))

        bg_path = None
        bg_img = None
        bg_depth_path = None
        bg_depth = None
        bg_depth_harmonized_path = None
        bg_depth_harmonized = None
        bg_depth_mesh_path = None
        fg_depth_mesh_path = None

        def read_bg_depth_harmonized(jobs, job_manager):
            nonlocal bg_depth_harmonized_path, bg_depth_harmonized, bg_depth_mesh_path, fg_depth_mesh_path
            if self.return_meshes:
                bg_depth_harmonized_path, bg_depth_mesh_path, fg_depth_mesh_path = jobs[0].outputs()[0]
            else:
                bg_depth_harmonized_path = jobs[0].outputs()[0]
            bg_depth_harmonized = np.asarray(imageio.imread(bg_depth_harmonized_path))

        def run_set_foreground(jobs, job_manager):
            nonlocal bg_depth_path, bg_depth
            bg_depth_path = jobs[0].outputs()[0]
            bg_depth = np.asarray(imageio.imread(bg_depth_path))
            set_foreground_job = GradioJob(job=self.diffhandles_client.submit(
                gradio_client.file(depth_path), gradio_client.file(fg_mask_path), gradio_client.file(bg_depth_path),
                gradio_client.file(img_path), gradio_client.file(bg_path),
                api_name="/set_foreground"))
            job_manager.add_job(set_foreground_job)
            job_manager.add_callback(func=read_bg_depth_harmonized, when_jobs_done=[set_foreground_job])

        def run_bg_depth(jobs, job_manager):
            nonlocal bg_path, bg_img
            bg_path = jobs[0].outputs()[0]
            if self.debug_images:
                bg_img = np.asarray(imageio.imread(bg_path))
            bg_depth_job = GradioJob(job=self.depth_estimator_client.submit(
                gradio_client.file(bg_path)))
            job_manager.add_job(bg_depth_job)
            job_manager.add_callback(func=run_set_foreground, when_jobs_done=[bg_depth_job])

        job_manager.add_job(bg_job)
        job_manager.add_callback(func=run_bg_depth, when_jobs_done=[bg_job])

        job_manager.run()

        for temp_path in [img_path, fg_mask_path, bg_path, depth_path, bg_depth_path, bg_depth_harmonized_path]:
            if pathlib.Path(temp_path).is_file():
                pathlib.Path(temp_path).unlink()

        outputs = (
            input_image_identity_path, depth, step1_completed,
            bg_depth_harmonized, bg_img)

        if self.return_meshes:
            outputs = outputs + (bg_depth_mesh_path, fg_depth_mesh_path)

        step2_completed = True

        outputs = outputs + (step2_completed,)

        return outputs

    def preview_edit(
            self, img: npt.NDArray = None, fg_mask: npt.NDArray = None, fg_mask_dilation: int = 3, bg_img: npt.NDArray = None, depth: npt.NDArray = None, bg_depth_harmonized: npt.NDArray = None,
            rot_angle: float = 0.0, rot_axis_x: float = 0.0, rot_axis_y: float = 1.0, rot_axis_z: float = 0.0,
            trans_x: float = 0.0, trans_y: float = 0.0, trans_z: float = 0.0,
            preview_mode = "depth"):

        # gr_input_image, gr_fg_mask, gr_depth, gr_bg_depth_harmonized,

        if any(inp is None for inp in [img, fg_mask]):
            raise ValueError('Some inputs are missing.')

        job_manager = GradioJobManager()

        # print(prompt)
        # print(f'{img.shape} {img.dtype}')
        # print(f'{fg_mask.shape} {fg_mask.dtype}')
        # print(f'{depth.shape} {depth.dtype}')
        # print(f'{bg_depth.shape} {bg_depth.dtype}')

        # pre-process inputs
        img = torch.from_numpy(img).to(dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        fg_mask = torch.from_numpy(fg_mask).to(dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        img = crop_and_resize(img=img, size=self.img_res)
        fg_mask = crop_and_resize(img=fg_mask, size=self.img_res)
        img = (img * 255.0)[0].permute(1, 2, 0).to(dtype=torch.uint8).numpy()
        fg_mask = (fg_mask * 255.0)[0].permute(1, 2, 0).to(dtype=torch.uint8).numpy()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.close()
            img_path = f.name
        imageio.imwrite(img_path, img)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.close()
            fg_mask_path = f.name
        imageio.imwrite(fg_mask_path, fg_mask)

        orig_depth = None
        depth_path = None
        if depth is None:
            depth_job = GradioJob(job=self.depth_estimator_client.submit(
                gradio_client.file(img_path)))

            depth = None
            def read_depth(jobs, job_manager):
                nonlocal depth_path, depth
                depth_path = jobs[0].outputs()[0]
                depth = np.asarray(imageio.imread(depth_path))

            job_manager.add_job(depth_job)
            job_manager.add_callback(func=read_depth, when_jobs_done=[depth_job])
        else:
            orig_depth = np.copy(depth)

        orig_bg_img = None
        orig_bg_depth_harmonized = None
        bg_depth_path = None
        bg_path = None
        if bg_depth_harmonized is None:
            bg_job = GradioJob(job=self.foreground_remover_client.submit(
                gradio_client.file(img_path), gradio_client.file(fg_mask_path),
                fg_mask_dilation))

            bg_depth = None
            def read_bg_depth(jobs, job_manager):
                nonlocal bg_depth_path, bg_depth
                bg_depth_path = jobs[0].outputs()[0]
                bg_depth = np.asarray(imageio.imread(bg_depth_path))

            bg_img = None
            def run_bg_depth(jobs, job_manager):
                nonlocal bg_path, bg_img
                bg_path = jobs[0].outputs()[0]
                if self.debug_images:
                    bg_img = np.asarray(imageio.imread(bg_path))
                bg_depth_job = GradioJob(job=self.depth_estimator_client.submit(
                    gradio_client.file(bg_path)))
                job_manager.add_job(bg_depth_job)
                job_manager.add_callback(func=read_bg_depth, when_jobs_done=[bg_depth_job])

            job_manager.add_job(bg_job)
            job_manager.add_callback(func=run_bg_depth, when_jobs_done=[bg_job])
        else:
            orig_bg_img = np.copy(bg_img)
            orig_bg_depth_harmonized = np.copy(bg_depth_harmonized)

        job_manager.run()

        # delete temporary files
        for temp_path in [img_path, fg_mask_path, bg_path, depth_path, bg_depth_path]:
            if temp_path is not None and pathlib.Path(temp_path).is_file():
                pathlib.Path(temp_path).unlink()

        depth = torch.from_numpy(depth).to(dtype=torch.float32, device=self.device)[None, None, ...]
        depth = crop_and_resize(img=depth, size=self.img_res)
        fg_mask = torch.from_numpy(fg_mask).to(dtype=torch.float32, device=self.device).permute(2, 0, 1)[None, ...] / 255.0
        if fg_mask.shape[1] > 1:
            fg_mask = fg_mask.mean(dim=1, keepdim=True) # average channels
        fg_mask = crop_and_resize(img=fg_mask, size=self.img_res)
        fg_mask = (fg_mask>0.5).to(dtype=torch.float32, device=self.device)
        rot_angle = float(rot_angle)
        rot_axis = torch.tensor([rot_axis_x, rot_axis_y, rot_axis_z], dtype=torch.float32, device=self.device)
        translation = torch.tensor([trans_x, trans_y, trans_z], dtype=torch.float32, device=self.device)

        # infilling bg depth hole
        if bg_depth_harmonized is None:
            print('infilling bg depth hole ...')
            bg_depth = torch.from_numpy(bg_depth).to(dtype=torch.float32, device=self.device)[None, None, ...]
            bg_depth = crop_and_resize(img=bg_depth, size=self.img_res)
            bg_depth_harmonized = solve_laplacian_depth(
                depth[0, 0].cpu().numpy(),
                bg_depth[0, 0].cpu().numpy(),
                scipy.ndimage.binary_dilation(fg_mask[0, 0].cpu().numpy(), iterations=15))
            bg_depth_harmonized = torch.from_numpy(bg_depth_harmonized)[None, None, ...].to(device=depth.device)
        else:
            bg_depth_harmonized = torch.from_numpy(bg_depth_harmonized).to(dtype=torch.float32, device=self.device)[None, None, ...]
            bg_depth_harmonized = crop_and_resize(img=bg_depth_harmonized, size=self.img_res)

        # transforming depth
        print('transforming depth ...')
        with torch.no_grad():
            edited_disparity, correspondences = transform_depth(
                depth=depth, bg_depth=bg_depth_harmonized, fg_mask=fg_mask,
                intrinsics=GuidedStableDiffuser.get_depth_intrinsics(device=depth.device),
                rot_angle=rot_angle, rot_axis=rot_axis, translation=translation,
                use_input_depth_normalization=False)

        print('done')

        if self.debug_images:
            debug_images = torch.cat([
                ((depth - depth.min()) / (depth.max() - depth.min()))[:, [0,0,0], ...],
                fg_mask[:, [0,0,0], ...],
                ((bg_depth_harmonized - bg_depth_harmonized.min()) / (bg_depth_harmonized.max() - bg_depth_harmonized.min()))[:, [0,0,0], ...],
                edited_disparity[:, [0,0,0], ...] / 255.0
            ], dim=3)
            debug_images = (debug_images * 255)[0].permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()
            debug_images = np.concatenate([img, debug_images[:, :debug_images.shape[0]*2], bg_img, debug_images[:, debug_images.shape[0]*2:]], axis=1)

        edited_disparity = (edited_disparity - edited_disparity.min())/(edited_disparity.max() - edited_disparity.min())
        edited_disparity = (edited_disparity[0, 0].detach().cpu().numpy()*255).round().astype("uint8")

        # gr_edited_image, gr_depth, gr_bg_depth_harmonized]

        if self.return_meshes or preview_mode == "rgb":
            intrinsics = GuidedStableDiffuser.get_depth_intrinsics(device=depth.device)

            bg_depth_mesh = depth_to_mesh(depth=bg_depth_harmonized, intrinsics=intrinsics)
            fg_depth_mesh = depth_to_mesh(depth=depth, intrinsics=intrinsics, mask=fg_mask[0, 0]>0.5)

        if self.return_meshes:
            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
                f.close()
                bg_depth_mesh_path = f.name
                self.temp_file_paths.append(bg_depth_mesh_path)
            save_mesh(bg_depth_mesh, bg_depth_mesh_path)

            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
                f.close()
                fg_depth_mesh_path = f.name
                self.temp_file_paths.append(fg_depth_mesh_path)
            save_mesh(fg_depth_mesh, fg_depth_mesh_path)

        # use original depth and bg_depth_harmonized if they were given as input to avoid accumulating numerical errors if the preview is run multiple times
        if orig_depth is not None:
            depth = orig_depth
        else:
            depth = depth[0].detach().cpu().permute(1, 2, 0).numpy()[..., 0]
        if orig_bg_depth_harmonized is not None:
            bg_depth_harmonized = orig_bg_depth_harmonized
        else:
            bg_depth_harmonized = bg_depth_harmonized[0].detach().cpu().permute(1, 2, 0).numpy()[..., 0]
        if orig_bg_img is not None:
            bg_img = orig_bg_img

        if preview_mode == "depth":
            outputs = (edited_disparity, bg_img, depth, bg_depth_harmonized)

        elif preview_mode == "rgb":

            with torch.no_grad():
                camera = Camera(intrinsics=intrinsics)

                renderer = PyTorch3DRenderer(
                    output_names=['flat_vertex_color'],
                    args=PyTorch3DRendererArgs(
                        device=self.device,
                        output_res=(depth.shape[-2], depth.shape[-1]),
                        cull_backfaces=True,
                        blur_radius=0.00001) # need a small blur radius otherwise the renderer may have artifacts if triangle edges align with pixel centers
                )
                
                fg_depth_mesh.verts.copy_(transform_points(
                    points=fg_depth_mesh.verts,
                    rot_angle=torch.tensor(rot_angle, dtype=torch.float32, device=self.device),
                    rot_axis=rot_axis, translation=translation))

                # TEMP: get mask of disoccluded pixels
                renderer.update_scene(scene_elements={
                    'meshes': [fg_depth_mesh],
                    'cameras': [camera]})
                rendered_outputs = renderer.render()
                edited_fg_mask = rendered_outputs['flat_vertex_color'][..., 2] > 0.5
                disoccluded_mask = (fg_mask > 0.5) & ~edited_fg_mask[:, None]

                # get vertex colors for foreground and background meshes
                # by sampling the foreground and background images at the mesh vertices
                im = torch.from_numpy(img).to(dtype=torch.float32, device=self.device).permute(2, 0, 1)[None, ...] / 255.0
                im = crop_and_resize(img=im, size=self.img_res)
                bg_im = torch.from_numpy(bg_img).to(dtype=torch.float32, device=self.device).permute(2, 0, 1)[None, ...] / 255.0
                bg_im = crop_and_resize(img=bg_im, size=self.img_res)
                for mesh, src_img in zip([bg_depth_mesh, fg_depth_mesh], [bg_im, im]):
                    img_coords = mesh.vert_attributes['color'].values[..., :2]
                    vert_colors = torch.nn.functional.grid_sample(
                        input=src_img,
                        grid=img_coords[None, None, ...]*2-1,
                        align_corners=True
                        )[0, :, 0, :].permute(1, 0)
                    mesh.vert_attributes['color'].values.copy_(vert_colors)

                renderer.update_scene(scene_elements={
                    'meshes': [bg_depth_mesh, fg_depth_mesh],
                    'cameras': [camera]})
                rendered_outputs = renderer.render()

                edited_image = rendered_outputs['flat_vertex_color'].permute(0, 3, 1, 2)
                
                # TEMP: use mask of disoccluded pixels as alpha channel
                edited_image[:, [3]] = (~disoccluded_mask).to(dtype=edited_image.dtype)

                edited_image = (edited_image[0].permute(1, 2, 0).detach().cpu().numpy()*255).round().astype("uint8")

            outputs = (edited_image, bg_img, depth, bg_depth_harmonized)

        else:
            raise ValueError(f'Invalid preview_mode: {preview_mode}.')

        if self.return_meshes:
            outputs = outputs + (bg_depth_mesh_path, fg_depth_mesh_path)
        
        if self.debug_images:
            outputs = outputs + (debug_images,)

        return outputs

    def transform_foreground(
            self, prompt: str = None, img: npt.NDArray = None,
            input_image_identity_path: str = None, depth: npt.NDArray = None, fg_mask: npt.NDArray = None, fg_mask_dilation: int = 6,
            bg_depth_harmonized: npt.NDArray = None, bg_img: npt.NDArray = None,
            bg_depth_mesh_path: str = None, fg_depth_mesh_path: str = None,
            rot_angle: float = 0.0, rot_axis_x: float = 0.0, rot_axis_y: float = 1.0, rot_axis_z: float = 0.0,
            trans_x: float = 0.0, trans_y: float = 0.0, trans_z: float = 0.0,
            gr_fg_weight: float = 1.5, gr_bg_weight: float = 1.25):

        inputs_from_previous_steps = [input_image_identity_path, depth, bg_depth_harmonized, bg_img]
        if self.return_meshes:
            inputs_from_previous_steps += [bg_depth_mesh_path, fg_depth_mesh_path]

        if any(x is None for x in inputs_from_previous_steps):
            # Results of the previous steps are not available, compute the previous steps
            if self.return_meshes:
                (input_image_identity_path, depth, step1_completed,
                bg_depth_harmonized, bg_img, bg_depth_mesh_path, fg_depth_mesh_path, step2_completed
                ) = self.set_foreground(prompt, img, input_image_identity_path, depth, fg_mask, fg_mask_dilation)
            else:
                (input_image_identity_path, depth, step1_completed,
                bg_depth_harmonized, bg_img, step2_completed
                ) = self.set_foreground(prompt, img, input_image_identity_path, depth, fg_mask, fg_mask_dilation)
        else:
            # Results of the previous steps are available,
            # don't change the status of the buttons and labels in the previous steps.
            step1_completed = True
            step2_completed = True

        if any(inp is None for inp in [prompt, fg_mask, depth, bg_depth_harmonized, input_image_identity_path]):
            raise ValueError('Some inputs are missing.')

        job_manager = GradioJobManager()

        # pre-process inputs
        fg_mask = torch.from_numpy(fg_mask).to(dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        fg_mask = crop_and_resize(img=fg_mask, size=self.img_res)
        fg_mask = (fg_mask * 255.0)[0].permute(1, 2, 0).to(dtype=torch.uint8).numpy()
        depth = torch.from_numpy(depth).to(dtype=torch.float32, device=self.device)[None, None, ...]
        depth = crop_and_resize(img=depth, size=self.img_res)
        depth = depth[0].detach().cpu().permute(1, 2, 0).numpy()[..., 0]
        bg_depth_harmonized = torch.from_numpy(bg_depth_harmonized).to(dtype=torch.float32, device=self.device)[None, None, ...]
        bg_depth_harmonized = crop_and_resize(img=bg_depth_harmonized, size=self.img_res)
        bg_depth_harmonized = bg_depth_harmonized[0].detach().cpu().permute(1, 2, 0).numpy()[..., 0]
        img = torch.from_numpy(img).to(dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        img = crop_and_resize(img=img, size=self.img_res)
        img = (img * 255.0)[0].permute(1, 2, 0).to(dtype=torch.uint8).numpy()
        bg_img = torch.from_numpy(bg_img).to(dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        bg_img = crop_and_resize(img=bg_img, size=self.img_res)
        bg_img = (bg_img * 255.0)[0].permute(1, 2, 0).to(dtype=torch.uint8).numpy()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.close()
            fg_mask_path = f.name
        imageio.imwrite(fg_mask_path, fg_mask)
        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
            f.close()
            depth_path = f.name
        imageio.imwrite(depth_path, depth)
        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
            f.close()
            bg_depth_harmonized_path = f.name
        imageio.imwrite(bg_depth_harmonized_path, bg_depth_harmonized)

        edited_image_path = None
        edited_image = None
        debug_images = None
        def read_edited_image(jobs, job_manager):
            nonlocal edited_image_path, edited_image, debug_images
            if self.debug_images:
                edited_image_path, debug_images_path = jobs[0].outputs()[0]
                debug_images = np.asarray(imageio.imread(debug_images_path))
            else:
                edited_image_path = jobs[0].outputs()[0]
            edited_image = np.asarray(imageio.imread(edited_image_path))

        transform_foreground_job = GradioJob(job=self.diffhandles_client.submit(
            prompt,
            gradio_client.file(fg_mask_path), gradio_client.file(depth_path),
            gradio_client.file(bg_depth_harmonized_path), gradio_client.file(input_image_identity_path),
            rot_angle, rot_axis_x, rot_axis_y, rot_axis_z,
            trans_x, trans_y, trans_z,
            gr_fg_weight, gr_bg_weight,
            api_name="/transform_foreground"))

        job_manager.add_job(transform_foreground_job)
        job_manager.add_callback(func=read_edited_image, when_jobs_done=[transform_foreground_job])
        job_manager.run()

        # delete temporary files
        for temp_path in [fg_mask_path, depth_path, bg_depth_harmonized_path, edited_image_path]:
            if pathlib.Path(temp_path).is_file():
                pathlib.Path(temp_path).unlink()

        # step 1 outputs
        outputs = (input_image_identity_path, depth, step1_completed)

        # step 2 outputs
        outputs += (bg_depth_harmonized, bg_img)
        if self.return_meshes:
            outputs = outputs + (bg_depth_mesh_path, fg_depth_mesh_path)
        outputs += (step2_completed,)

        # step 3 outputs
        outputs += (edited_image,)
        if self.debug_images:
            debug_images = np.concatenate([img, debug_images[:, :debug_images.shape[0]*2], bg_img, debug_images[:, debug_images.shape[0]*2:]], axis=1)
            outputs = outputs + (debug_images,)

        return outputs

    def read_prompt(self, path):
        
        with open(path, 'r') as f:
            prompt = f.read().splitlines()
        prompt = [prompt for prompt in prompt if len(prompt) > 0]
        if len(prompt) == 0:
            print(f'WARNING: prompt for default inputs {path} is empty. Skipping sample.')
            return None, None, None, None, None, None, None
        else:
            if len(prompt) > 1:
                print(f'WARNING: prompt for default inputs {path} has multiple lines, only using the first line.')
            prompt = prompt[0]

        return prompt

    
    def read_default_edit(self, path):
        dir_path = os.path.dirname(path)
        edit_name = os.path.basename(path)

        input_image_path = os.path.join(dir_path, "input.png")
        
        # load input prompt
        input_prompt_path = os.path.join(dir_path, "prompt.txt")
        input_prompt = self.read_prompt(path=input_prompt_path)

        fg_mask_path = os.path.join(dir_path, "mask.png")

        # load foreground prompt
        fg_prompt_path = os.path.join(dir_path, "fg_prompt.txt")
        fg_prompt = self.read_prompt(path=fg_prompt_path)

        # load transforms
        with open(os.path.join(dir_path, 'transforms.json'), 'r') as f:
            transforms = json.load(f)
        if edit_name not in transforms:
            raise ValueError(f'WARNING: transform {edit_name} not found in {os.path.join(dir_path, "transforms.json")}.')
        transform = transforms[edit_name]
        translation = transform['translation'] if 'translation' in transform else [0.0, 0.0, 0.0]
        rot_axis = transform['rotation_axis'] if 'rotation_axis' in transform else [0.0, 1.0, 0.0]
        rot_angle = transform['rotation_angle'] if 'rotation_angle' in transform else 0.0

        # load other config
        config_path = os.path.join(dir_path, "config.yaml")
        conf = OmegaConf.load(config_path)
        
        return {
            'input_image_path': input_image_path,
            'input_prompt': input_prompt,
            'fg_mask_path': fg_mask_path,
            'fg_prompt': fg_prompt,
            'fg_mask_dilation': conf.fg_removal_dilation,
            'translation': translation,
            'rot_axis': rot_axis,
            'rot_angle': rot_angle,
            'fg_preservation': conf.fg_weight,
            'bg_preservation': conf.bg_weight}
    
    def toggle_visibility(self, current_visibility):
        current_visibility = not current_visibility
        return gr.update(visible=current_visibility), current_visibility
    
    def step_processed_changed(self, step_processed):
        
        if step_processed:
            status_text = "Step processed."
        else:
            status_text = "Inputs changed, step will be re-processed."

        return gr.Markdown(value=status_text)
    
    def build_gradio_app(self):

        default_inputs = self.read_default_edit(path=self.default_edit_path)

        with gr.Blocks() as gr_app:
            with gr.Row():
                gr.HTML(value="""
                        <h1>Diffusion Handles</h1>
                        """
                        )

            with gr.Row():
                
                # Step 2: Upload or Generate an Input Image
                with gr.Column():
                    gr_input_image = gr.Image(label="Input Image", value=default_inputs['input_image_path'])
                    gr.HTML(value="""
                            <h2>Step 1: Upload or Generate an Input Image.</h2>
                            <ol>
                            <li>Enter a text prompt that describes the input image.</li>
                            <li>Upload the input image or generate it from the text prompt.</li>
                            </ol>
                            The text prompt is still needed, even if you upload an image!
                            """
                            # <li>Process the input image. (Processing time ~46 seconds. This will be improved in the future.)</li>
                            )
                    
                    gr_generate_button = gr.Button("Generate Input Image")
                    gr_text_prompt = gr.Textbox(label="Image Prompt", value=default_inputs['input_prompt'])
                    gr_set_input_button = gr.Button("Process Input Image", visible=False)

                    # gr_step1_processed = gr.State(False)
                    
                    gr_step1_processed = gr.Checkbox(label="Step 1 Processed", value=False, visible=False, interactive=False)
                    gr_step1_processed_text = gr.Markdown(value="Inputs changed, step will be re-processed.")

                    # gr_show_step1_outputs = gr.Checkbox(label="Show Intermediate Step 1 Outputs", value=False)
                    # gr.HTML(value="<h3>Intermediate Step 1 Outputs</h3>")
                    gr_step1_show_intermediate = gr.Checkbox(label="Show Intermediate Outputs", value=False)
                    gr_step1_intermediate_row = gr.Row(visible=False)
                    gr_step1_intermediate_row_visible = gr.State(False)
                    with gr_step1_intermediate_row:
                        gr_input_image_identity = gr.File(label="Input Image Identity", type="filepath", interactive=False)
                        gr_depth = HDRImage(label="Depth Image", interactive=False)
                    

                # Step 2: Select Object
                with gr.Column():
                    gr_fg_mask = gr.Image(label="Object Mask", value=default_inputs['fg_mask_path'])
                    gr.HTML(value="""
                            <h2>Step 2: Select Object</h2>
                            <ol>
                            <li>Upload an object mask or generate it from a text prompt.</li>
                            </ol>
                            """
                            # <li>Process the object selection. (Processing time ~22 seconds with Object Peeling, or ~2 seconds with Lama.)</li>
                            )
                    
                    gr_select_object_button = gr.Button("Select Object")
                    gr_object_prompt = gr.Textbox(label="Select Object with a Prompt", value=default_inputs['fg_prompt'])
                    gr_fg_mask_dilation = gr.Number(label="Forground Mask Dilation", precision=0, value=default_inputs['fg_mask_dilation'], minimum=0, maximum=100)
                    gr_set_foreground_button = gr.Button("Process Object Selection", visible=False)

                    # gr_step2_processed = gr.State(False)
                    
                    gr_step2_processed = gr.Checkbox(label="Step 2 Processed", value=False, visible=False, interactive=False)
                    gr_step2_processed_text = gr.Markdown(value="Inputs changed, step will be re-processed.")

                    # gr_show_step2_outputs = gr.Checkbox(label="Show Intermediate Step 2 Outputs", value=False)
                    # gr.HTML(value="<h3>Intermediate Step 2 Outputs</h3>")
                    gr_step2_show_intermediate = gr.Checkbox(label="Show Intermediate Outputs", value=False)
                    gr_step2_intermediate_row = gr.Row(visible=False)
                    gr_step2_intermediate_row_visible = gr.State(False)
                    with gr_step2_intermediate_row:
                        gr_bg_depth_harmonized = HDRImage(label="Harmonized Background Depth", interactive=False)
                        gr_bg_img = gr.Image(label="Background Image", interactive=False)
                        if self.return_meshes:
                            with gr.Row():
                                with gr.Column():
                                    gr_bg_depth_mesh = gr.File(label="Background Mesh", type="filepath", interactive=False)
                                with gr.Column():
                                    gr_fg_depth_mesh = gr.File(label="Foreground Mesh", type="filepath", interactive=False)
                    

                # Step 3: Edit Object
                with gr.Column():
                    gr_edited_image = gr.Image(label="Edited Image", format='png')
                    if self.debug_images:
                        gr_debug_images = gr.Image(label="Debug Images")
                    gr.HTML(value="""
                            <h2>Step 3: Edit Object</h2>
                            <li>Enter the transformation parameters (rotation angle, rotation axis, tranlation - translation is roughly in meters).</li>
                            <li>'Preview Edit' to get a fast preview of the depth of the edited image.</li>
                            <li>'Perform Edit' to get the edited image.</li>
                            Previewing is fast, but performing the edit takes:<br/>
                            ~46s if inputs to step 1 were changed (or running first time) +<br/>
                            ~2s if inputs to steps 1 or 2 were changed (or running first time) +<br/>
                            ~36s for processing the object edit
                            (We are looking to improve processing times in the future.)
                            """
                            )
                    with gr.Row():
                        with gr.Column():
                            gr_preview_edit_button = gr.Button("Preview Edit")
                        with gr.Column():
                            gr_edit_button = gr.Button("Perform Edit")
                    gr_trans_x = gr.Number(label="Translation X (left, right)", value=default_inputs['translation'][0], minimum=-100.0, maximum=100.0)
                    gr_trans_y = gr.Number(label="Translation Y (up, down)", value=default_inputs['translation'][1], minimum=-100.0, maximum=100.0)
                    gr_trans_z = gr.Number(label="Translation Z (backward, forward)", value=default_inputs['translation'][2], minimum=-100.0, maximum=100.0)
                    # gr_rot_angle = gr.Slider(label="Rotation Angle", value=40.0, minimum=-180.0, maximum=180.0, step=1.0)
                    gr_rot_angle = gr.Slider(label="Rotation Angle", value=default_inputs['rot_angle'], minimum=-180.0, maximum=180.0, step=1.0)
                    gr_rot_axis_x = gr.Number(label="Rotation Axis X", value=default_inputs['rot_axis'][0], minimum=-1.0, maximum=1.0)
                    gr_rot_axis_y = gr.Number(label="Rotation Axis Y", value=default_inputs['rot_axis'][1], minimum=-1.0, maximum=1.0)
                    gr_rot_axis_z = gr.Number(label="Rotation Axis Z", value=default_inputs['rot_axis'][2], minimum=-1.0, maximum=1.0)

                    gr_fg_weight = gr.Number(label="Foreground Preservation", value=default_inputs['fg_preservation'], minimum=0.0, maximum=100.0)
                    gr_bg_weight = gr.Number(label="Background Preservation", value=default_inputs['bg_preservation'], minimum=0.0, maximum=100.0)
                    # gr_fg_mask_dilation = gr.Number(label="Forground Mask Dilation", precision=0, value=3, minimum=0, maximum=100)
                    
                    gr_preview_mode = gr.Dropdown(label="Preview Mode", choices=["depth", "rgb"], value="depth")
                   
            gr_step1_show_intermediate.change(
                self.toggle_visibility,
                inputs=[gr_step1_intermediate_row_visible],
                outputs=[gr_step1_intermediate_row, gr_step1_intermediate_row_visible])
            
            gr_step2_show_intermediate.change(
                self.toggle_visibility,
                inputs=[gr_step2_intermediate_row_visible],
                outputs=[gr_step2_intermediate_row, gr_step2_intermediate_row_visible])
            
            step1_inputs = [gr_text_prompt, gr_input_image]
            step1_outputs = [gr_input_image_identity, gr_depth]

            step2_inputs = [gr_fg_mask, gr_fg_mask_dilation]
            step2_outputs = [gr_bg_depth_harmonized, gr_bg_img]
            if self.return_meshes:
                step2_outputs += [gr_bg_depth_mesh, gr_fg_depth_mesh]

            step3_inputs = [gr_rot_angle, gr_rot_axis_x, gr_rot_axis_y, gr_rot_axis_z, gr_trans_x, gr_trans_y, gr_trans_z, gr_fg_weight, gr_bg_weight]
            step3_outputs = [gr_edited_image]
            if self.debug_images:
                step3_outputs += [gr_debug_images]

            for gr_input in step1_inputs:
                gr_input.change(
                    lambda: (None,)*(len(step1_outputs)+len(step2_outputs)) + (False, False),
                    inputs=[],
                    outputs=step1_outputs + step2_outputs + [gr_step1_processed, gr_step2_processed])

            for gr_input in step2_inputs:
                gr_input.change(
                    lambda: (None,)*len(step2_outputs) + (False,),
                    inputs=[],
                    outputs=step2_outputs + [gr_step2_processed])

            gr_step2_processed.change(
                self.step_processed_changed,
                inputs=[gr_step2_processed],
                outputs=[gr_step2_processed_text]
            )

            gr_step1_processed.change(
                self.step_processed_changed,
                inputs=[gr_step1_processed],
                outputs=[gr_step1_processed_text]
            )

            # gr_show_step1_outputs.change(
            #     self.set_intermediate_outputs_visible,
            #     inputs=[gr_show_step1_outputs] + step1_outputs,
            #     outputs=step1_outputs)
            
            # gr_show_step2_outputs.change(
            #     self.set_intermediate_outputs_visible,
            #     inputs=[gr_show_step2_outputs] + step2_outputs,
            #     outputs=step2_outputs)
            
            gr_generate_button.click(
                self.generate_input,
                inputs=[gr_text_prompt],
                outputs=[gr_input_image])

            gr_select_object_button.click(
                self.select_foreground,
                inputs=[gr_input_image, gr_object_prompt],
                outputs=[gr_fg_mask])

            gr_set_input_button.click(
                self.set_input_image,
                inputs=step1_inputs,
                outputs=step1_outputs + [gr_step1_processed])

            gr_set_foreground_button.click(
                self.set_foreground,
                inputs=step1_inputs + step2_inputs,
                outputs=step1_outputs + [gr_step1_processed] + step2_outputs + [gr_step2_processed])

            gr_edit_button.click(
                self.transform_foreground,
                inputs=step1_inputs + step1_outputs + step2_inputs + step2_outputs + step3_inputs,
                outputs=step1_outputs + [gr_step1_processed] + step2_outputs + [gr_step2_processed] + step3_outputs)

            preview_edit_button_outputs = [
                gr_edited_image, gr_bg_img, gr_depth, gr_bg_depth_harmonized]
            if self.return_meshes:
                preview_edit_button_outputs += [gr_bg_depth_mesh, gr_fg_depth_mesh]
            if self.debug_images:
                preview_edit_button_outputs.append(gr_debug_images)
            gr_preview_edit_button.click(
                self.preview_edit,
                inputs=[
                    gr_input_image, gr_fg_mask, gr_fg_mask_dilation, gr_bg_img, gr_depth, gr_bg_depth_harmonized,
                    gr_rot_angle, gr_rot_axis_x, gr_rot_axis_y, gr_rot_axis_z, gr_trans_x, gr_trans_y, gr_trans_z,
                    gr_preview_mode],
                outputs=preview_edit_button_outputs
            )

            # gr_edit_button.click(
            #     self.run_diffhandles_pipeline,
            #     inputs=[
            #         gr_text_prompt, gr_input_image, gr_fg_mask,
            #         gr_rot_angle, gr_rot_axis_x, gr_rot_axis_y, gr_rot_axis_z, gr_trans_x, gr_trans_y, gr_trans_z,
            #         gr_fg_weight, gr_bg_weight, gr_fg_mask_dilation],
            #     outputs=[gr_edited_image, gr_debug_images] if self.debug_images else [gr_edited_image]
            #         )
        return gr_app

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--netpath', type=str, default='/dh')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--text2img_url', type=str, default='http://localhost:6011/text2img')
    parser.add_argument('--foreground_selector_url', type=str, default='http://localhost:6010/foreground_selector')
    parser.add_argument('--foreground_remover_url', type=str, default='http://localhost:6008/foreground_remover')
    parser.add_argument('--depth_estimator_url', type=str, default='http://localhost:6007/depth_estimator')
    parser.add_argument('--diffhandles_url', type=str, default='http://localhost:6006/diffhandles')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--debug_images', action='store_true', default=False)
    parser.add_argument('--return_meshes', action='store_true', default=False)
    parser.add_argument('--default_edit_path', type=str, default='data/toy_cubes/edit_001')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    server = DiffhandlesPipelineWebapp(
        netpath=args.netpath, port=args.port,
        text2img_url=args.text2img_url, foreground_selector_url=args.foreground_selector_url, foreground_remover_url=args.foreground_remover_url,
        depth_estimator_url=args.depth_estimator_url, diffhandles_url=args.diffhandles_url,
        default_edit_path=args.default_edit_path, debug_images=args.debug_images, return_meshes=args.return_meshes, device=args.device)
    server.start()
