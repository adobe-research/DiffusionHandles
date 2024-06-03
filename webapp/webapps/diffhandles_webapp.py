import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import argparse
import tempfile
# import io
import time

import torch
import numpy as np
import numpy.typing as npt
import gradio as gr
from gradio_hdrimage import HDRImage
from omegaconf import OmegaConf

from diffhandles import DiffusionHandles
from diffhandles.depth_transform import depth_to_mesh
from diffhandles.mesh_io import save_mesh

from utils import crop_and_resize
from gradio_webapp import GradioWebapp

class DiffhandlesWebapp(GradioWebapp):

    def __init__(self, netpath: str, port: int, config_path: str = None, debug_images=False, return_meshes=False, device: str = 'cuda:0'):
        super().__init__(netpath=netpath, port=port)

        self.debug_images = debug_images
        self.return_meshes = return_meshes
        self.config_path = config_path
        diff_handles_config = OmegaConf.load(config_path) if config_path is not None else None
        self.diff_handles = DiffusionHandles(conf=diff_handles_config)
        self.diff_handles.to(device)
        self.img_res = 512

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

    def set_input_image(
            self, prompt: str, depth: npt.NDArray, img: npt.NDArray) -> str:

        if any(inp is None for inp in [prompt, depth]):
            raise ValueError('Some inputs are missing.')

        self.delete_old_temp_files()

        # prepare inputs (convert to torch tensors, etc.)
        depth = torch.from_numpy(depth).to(device=self.diff_handles.device, dtype=torch.float32)[None, None, ...]
        depth = crop_and_resize(img=depth, size=self.img_res)

        if img is None:
            null_text_emb, init_noise = None, None
        else:
            img = torch.from_numpy(img).to(device=self.diff_handles.device, dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
            img = crop_and_resize(img=img, size=self.img_res)

            # invert input image if input image is given
            null_text_emb, init_noise = self.diff_handles.invert_input_image(img, depth, prompt)

        # generate input image (using the inversion result if an input image was given)
        null_text_emb, init_noise, activations, latent_image = self.diff_handles.generate_input_image(
            depth=depth, prompt=prompt, null_text_emb=null_text_emb, init_noise=init_noise)

        input_image_identity = {
            'null_text_emb': null_text_emb.detach().cpu().numpy(),
            'init_noise': init_noise.detach().cpu().numpy(),
            'activations1': activations[0].detach().cpu().numpy(),
            'activations2': activations[1].detach().cpu().numpy(),
            'activations3': activations[2].detach().cpu().numpy()
        }
        
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            f.close()
            input_image_identity_path = f.name
            self.temp_file_paths.append(input_image_identity_path)
        np.savez(input_image_identity_path, **input_image_identity)

        return input_image_identity_path

        # input_image_identity_bytes = io.BytesIO()
        # np.savez(input_image_identity_bytes, **input_image_identity)

        # return input_image_identity_bytes.read()

    def set_foreground(self, depth: npt.NDArray, fg_mask: npt.NDArray, bg_depth: npt.NDArray, img: npt.NDArray, bg_img: npt.NDArray) -> npt.NDArray:

        if any(inp is None for inp in [depth, fg_mask, bg_depth]):
            raise ValueError('Some inputs are missing.')

        self.delete_old_temp_files()
        
        # prepare inputs (convert to torch tensors, etc.)
        depth = torch.from_numpy(depth).to(device=self.diff_handles.device, dtype=torch.float32)[None, None, ...]
        depth = crop_and_resize(img=depth, size=self.img_res)

        fg_mask = torch.from_numpy(fg_mask).to(device=self.diff_handles.device, dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        if fg_mask.shape[1] > 1:
            fg_mask = fg_mask.mean(dim=1, keepdim=True) # average channels
        fg_mask = crop_and_resize(img=fg_mask, size=self.img_res)
        fg_mask = (fg_mask>0.5).to(device=self.diff_handles.device, dtype=torch.float32)

        bg_depth = torch.from_numpy(bg_depth).to(device=self.diff_handles.device, dtype=torch.float32)[None, None, ...]
        bg_depth = crop_and_resize(img=bg_depth, size=self.img_res)

        img = torch.from_numpy(img).to(device=self.diff_handles.device, dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        img = crop_and_resize(img=img, size=self.img_res)

        bg_img = torch.from_numpy(bg_img).to(device=self.diff_handles.device, dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        bg_img = crop_and_resize(img=bg_img, size=self.img_res)

        # set foreground
        bg_depth = self.diff_handles.set_foreground(depth=depth, fg_mask=fg_mask, bg_depth=bg_depth)

        if self.return_meshes:
            with torch.no_grad():
                intrinsics = self.diff_handles.diffuser.get_depth_intrinsics(device=depth.device)
                
                bg_depth_mesh = depth_to_mesh(depth=bg_depth, intrinsics=intrinsics)
                fg_depth_mesh = depth_to_mesh(depth=depth, intrinsics=intrinsics, mask=fg_mask[0, 0]>0.5)

                # change color attribute of vertices
                # from image coordinates corresponding to each vertex
                # to the color of the input image at the image coordinates
                for mesh, src_img in zip([bg_depth_mesh, fg_depth_mesh], [bg_img, img]):
                    img_coords = mesh.vert_attributes['color'].values[..., :2]
                    vert_colors = torch.nn.functional.grid_sample(
                        input=src_img,
                        grid=img_coords[None, None, ...]*2-1,
                        align_corners=True
                        )[0, :, 0, :].permute(1, 0)
                    mesh.vert_attributes['color'].values.copy_(vert_colors)
            
            # with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
                f.close()
                bg_depth_mesh_path = f.name
                self.temp_file_paths.append(bg_depth_mesh_path)
            save_mesh(bg_depth_mesh, bg_depth_mesh_path)

            # with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
                f.close()
                fg_depth_mesh_path = f.name
                self.temp_file_paths.append(fg_depth_mesh_path)
            save_mesh(fg_depth_mesh, fg_depth_mesh_path)

        # prepare output (convert to numpy array)
        bg_depth = bg_depth[0].detach().cpu().permute(1, 2, 0).numpy()[..., 0]

        if self.return_meshes:
            return bg_depth, bg_depth_mesh_path, fg_depth_mesh_path
        else:
            return bg_depth

    def transform_foreground(self, prompt: str, fg_mask: npt.NDArray = None,
            depth: npt.NDArray = None, bg_depth_harmonized: npt.NDArray = None,
            input_image_identity_path: str = None,
            rot_angle: float = 0.0, rot_axis_x: float = 0.0, rot_axis_y: float = 1.0, rot_axis_z: float = 0.0,
            trans_x: float = 0.0, trans_y: float = 0.0, trans_z: float = 0.0,
            fg_weight: float = 1.5, bg_weight: float = 1.25) -> npt.NDArray:

        if any(inp is None for inp in [prompt, fg_mask, depth, bg_depth_harmonized, input_image_identity_path]):
            raise ValueError('Some inputs are missing.')

        input_image_identity = np.load(input_image_identity_path)
        null_text_emb = torch.from_numpy(input_image_identity['null_text_emb']).to(device=self.diff_handles.device)
        init_noise = torch.from_numpy(input_image_identity['init_noise']).to(device=self.diff_handles.device)
        activations = [torch.from_numpy(input_image_identity[f'activations{i+1}']).to(device=self.diff_handles.device) for i in range(3)]

        # prepare inputs (convert to torch tensors, etc.)
        fg_mask = torch.from_numpy(fg_mask).to(device=self.diff_handles.device, dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        if fg_mask.shape[1] > 1:
            fg_mask = fg_mask.mean(dim=1, keepdim=True) # average channels
        fg_mask = crop_and_resize(img=fg_mask, size=self.img_res)
        fg_mask = (fg_mask>0.5).to(device=self.diff_handles.device, dtype=torch.float32)

        depth = torch.from_numpy(depth).to(device=self.diff_handles.device, dtype=torch.float32)[None, None, ...]
        depth = crop_and_resize(img=depth, size=self.img_res)

        bg_depth_harmonized = torch.from_numpy(bg_depth_harmonized).to(device=self.diff_handles.device, dtype=torch.float32)[None, None, ...]
        bg_depth_harmonized = crop_and_resize(img=bg_depth_harmonized, size=self.img_res)

        rot_angle = float(rot_angle)
        rot_axis = torch.tensor([rot_axis_x, rot_axis_y, rot_axis_z], dtype=torch.float32, device=self.diff_handles.device)
        translation = torch.tensor([trans_x, trans_y, trans_z], dtype=torch.float32, device=self.diff_handles.device)

        (edited_img, edited_disparity
         ) = self.diff_handles.transform_foreground(
            depth=depth, prompt=prompt,
            fg_mask=fg_mask, bg_depth=bg_depth_harmonized,
            null_text_emb=null_text_emb, init_noise=init_noise,
            activations=activations,
            rot_angle=rot_angle, rot_axis=rot_axis, translation=translation,
            fg_weight=fg_weight, bg_weight=bg_weight,
            use_input_depth_normalization=False)

        edited_img = (edited_img * 255.0)[0].permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()

        if self.debug_images:
            debug_images = torch.cat([
                ((depth - depth.min()) / (depth.max() - depth.min()))[:, [0,0,0], ...],
                fg_mask[:, [0,0,0], ...],
                ((bg_depth_harmonized - bg_depth_harmonized.min()) / (bg_depth_harmonized.max() - bg_depth_harmonized.min()))[:, [0,0,0], ...],
                edited_disparity[:, [0,0,0], ...] / 255.0
            ], dim=3)
            debug_images = (debug_images * 255)[0].permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()
            return edited_img, debug_images
        else:
            return edited_img

    def run_diffhandles(
            self, prompt: str = None, img: npt.NDArray = None, fg_mask: npt.NDArray = None,
            depth: npt.NDArray = None, bg_depth: npt.NDArray = None,
            rot_angle: float = 0.0, rot_axis_x: float = 0.0, rot_axis_y: float = 1.0, rot_axis_z: float = 0.0,
            trans_x: float = 0.0, trans_y: float = 0.0, trans_z: float = 0.0,
            fg_weight: float = 1.5, bg_weight: float = 1.25) -> npt.NDArray:

        # print('run_diffhandles')

        if any(inp is None for inp in [prompt, fg_mask, depth, bg_depth]):
            raise ValueError('Some inputs are missing.')

        # print(prompt)
        # print(f'{img.shape} {img.dtype}')
        # print(f'{fg_mask.shape} {fg_mask.dtype}')
        # print(f'{depth.shape} {depth.dtype}')
        # print(f'{bg_depth.shape} {bg_depth.dtype}')

        # prepare inputs (convert to torch tensors, etc.)
        fg_mask = torch.from_numpy(fg_mask).to(device=self.diff_handles.device, dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
        if fg_mask.shape[1] > 1:
            fg_mask = fg_mask.mean(dim=1, keepdim=True) # average channels
        fg_mask = crop_and_resize(img=fg_mask, size=self.img_res)
        fg_mask = (fg_mask>0.5).to(device=self.diff_handles.device, dtype=torch.float32)

        depth = torch.from_numpy(depth).to(device=self.diff_handles.device, dtype=torch.float32)[None, None, ...]
        depth = crop_and_resize(img=depth, size=self.img_res)

        bg_depth = torch.from_numpy(bg_depth).to(device=self.diff_handles.device, dtype=torch.float32)[None, None, ...]
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

        if img is None:
            null_text_emb, init_noise = None, None
        else:
            img = torch.from_numpy(img).to(device=self.diff_handles.device, dtype=torch.float32).permute(2, 0, 1)[None, ...] / 255.0
            img = crop_and_resize(img=img, size=self.img_res)

            null_text_emb, init_noise = self.diff_handles.invert_input_image(img, depth, prompt)

        null_text_emb, init_noise, activations, latent_image = self.diff_handles.generate_input_image(
            depth=depth, prompt=prompt, null_text_emb=null_text_emb, init_noise=init_noise)

        bg_depth = self.diff_handles.set_foreground(depth=depth, fg_mask=fg_mask, bg_depth=bg_depth)

        # (bg_depth, null_text_emb, init_noise, activations, latent_image
        #  ) = self.diff_handles.set_foreground(
        #     img, depth, prompt, fg_mask, bg_depth)

        (edited_img, edited_disparity
         ) = self.diff_handles.transform_foreground(
            depth=depth, prompt=prompt,
            fg_mask=fg_mask, bg_depth=bg_depth,
            null_text_emb=null_text_emb, init_noise=init_noise,
            activations=activations,
            rot_angle=rot_angle, rot_axis=rot_axis, translation=translation,
            fg_weight=fg_weight, bg_weight=bg_weight,
            use_input_depth_normalization=False)

        edited_img = (edited_img * 255.0)[0].permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()

        if self.debug_images:
            debug_images = torch.cat([
                img,
                ((depth - depth.min()) / (depth.max() - depth.min()))[:, [0,0,0], ...],
                fg_mask[:, [0,0,0], ...],
                ((bg_depth - bg_depth.min()) / (bg_depth.max() - bg_depth.min()))[:, [0,0,0], ...],
                edited_disparity[:, [0,0,0], ...] / 255.0
            ], dim=3)
            debug_images = (debug_images * 255)[0].permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()
            return edited_img, debug_images
        else:
            return edited_img

    def build_gradio_app(self):

        with gr.Blocks() as gr_app:
            with gr.Row():
                gr.HTML(value="""
                        <h2>Step 1: Upload or Generate an input image.</h2>
                        <ol>
                        <li>Enter a text prompt that describes the input image.</li>
                        <li>Enter the depth of the input image.</li>
                        <li>Optionally upload the input image (otherwise it will be generated from the prompt).</li>
                        <li>'Set Input' to set or generate the input image.</li>
                        </ol>
                        The text prompt is still needed, even if you upload an image!
                        The output is an npz file describing the identity of the input image.
                        """
                        )
            with gr.Row():
                with gr.Column():
                    gr_text_prompt = gr.Textbox(label="Text Prompt", value="a sunflower in the garden")
                    gr_input_image = gr.Image(label="Input Image", value="data/sunflower/input.png")
                    gr_depth = HDRImage(label="Depth", value="data/sunflower/depth.exr")
                    gr_set_input_button = gr.Button("Set Input")
                with gr.Column():
                    gr_input_image_identity = gr.File(label="Input Image Identity", type="filepath")

            with gr.Row():
                gr.HTML(value="""
                        <h2>Step 2: Select Object</h2>
                        <ol>
                        <li>Upload a mask of the foreground object.</li>
                        <li>Upload a the depth of the background (without foreground object).</li>
                        <li>'Select Object' to perform selection.</li>
                        </ol>
                        The output is an updated background depth that has been harmonized to match the depth of the input image everywhere except where the foreground object used to be.
                        """
                        )
            with gr.Row():
                with gr.Column():
                    gr_fg_mask = gr.Image(label="Foreground Mask", value="data/sunflower/mask.png")
                    gr_bg_depth = HDRImage(label="Background Depth", value="data/sunflower/bg_depth.exr")
                    gr_bg_image = gr.Image(label="Background Image", value="data/sunflower/bg.png")
                    gr_select_button = gr.Button("Select Object")
                with gr.Column():
                    gr_bg_depth_harmonized = HDRImage(label="Harmonized Background Depth")
                    if self.return_meshes:
                        gr_bg_depth_mesh = gr.File(label="Background Mesh", type="filepath")
                        gr_fg_depth_mesh = gr.File(label="Foreground Mesh", type="filepath")

            with gr.Row():
                gr.HTML(value="""
                        <h2>Step 3: Edit Object</h2>
                        <ol>
                        <li>Upload a mask of the foreground object.</li>
                        <li>'Select Object' to perform selection and view a mask of the selected object.</li>
                        </ol>
                        """
                        )
            with gr.Row():
                with gr.Column():
                    gr_rot_angle = gr.Slider(label="Rotation Angle", value=40.0, minimum=-180.0, maximum=180.0, step=1.0)
                    gr_rot_axis_x = gr.Number(label="Rotation Axis X", value=0.0, minimum=-1.0, maximum=1.0)
                    gr_rot_axis_y = gr.Number(label="Rotation Axis Y", value=1.0, minimum=-1.0, maximum=1.0)
                    gr_rot_axis_z = gr.Number(label="Rotation Axis Z", value=0.0, minimum=-1.0, maximum=1.0)
                    gr_trans_x = gr.Number(label="Translation X", value=0.0, minimum=-100.0, maximum=100.0)
                    gr_trans_y = gr.Number(label="Translation Y", value=0.0, minimum=-100.0, maximum=100.0)
                    gr_trans_z = gr.Number(label="Translation Z", value=0.0, minimum=-100.0, maximum=100.0)
                    gr_fg_weight = gr.Number(label="Foreground Weight", value=1.5, minimum=0.0, maximum=100.0)
                    gr_bg_weight = gr.Number(label="Background Weight", value=1.25, minimum=0.0, maximum=100.0)
                    gr_edit_button = gr.Button("Edit Object")
                with gr.Column():
                    if self.debug_images:
                        gr_debug_images = gr.Image(label="Debug Images")
                    gr_edited_image = gr.Image(label="Edited Image")

            with gr.Row():
                 gr.HTML(value="""
                            <h2>Run Full Pipeline</h2>
                            <ol>
                            <li>Provide all inputs in the left columns of each step (or leave at their defaults).</li>
                            <li>'Run Full Pipeline' to run each step in sequence.</li>
                            </ol>
                            """
                            )
            with gr.Row():
                gr_pipeline_button = gr.Button("Run Full Pipeline")

            gr_set_input_button.click(
                self.set_input_image,
                inputs=[gr_text_prompt, gr_depth, gr_input_image],
                outputs=[gr_input_image_identity])

            gr_select_button.click(
                self.set_foreground,
                inputs=[gr_depth, gr_fg_mask, gr_bg_depth, gr_input_image, gr_bg_image],
                outputs=[gr_bg_depth_harmonized, gr_bg_depth_mesh, gr_fg_depth_mesh] if self.return_meshes else [gr_bg_depth_harmonized])

            gr_edit_button.click(
                self.transform_foreground,
                inputs=[
                    gr_text_prompt, gr_fg_mask, gr_depth, gr_bg_depth_harmonized,
                    gr_input_image_identity,
                    gr_rot_angle, gr_rot_axis_x, gr_rot_axis_y, gr_rot_axis_z, gr_trans_x, gr_trans_y, gr_trans_z,
                    gr_fg_weight, gr_bg_weight],
                outputs=[gr_edited_image, gr_debug_images] if self.debug_images else [gr_edited_image])

            gr_pipeline_button.click(
                self.run_diffhandles,
                inputs=[
                    gr_text_prompt, gr_input_image, gr_fg_mask, gr_depth, gr_bg_depth_harmonized,
                    gr_rot_angle, gr_rot_axis_x, gr_rot_axis_y, gr_rot_axis_z, gr_trans_x, gr_trans_y, gr_trans_z,
                    gr_fg_weight, gr_bg_weight],
                outputs=[gr_edited_image, gr_debug_images] if self.debug_images else [gr_edited_image])

        return gr_app

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--netpath', type=str, default='/diffhandles')
    parser.add_argument('--port', type=int, default=6006)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--debug_images', action='store_true', default=False)
    parser.add_argument('--return_meshes', action='store_true', default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    server = DiffhandlesWebapp(
        netpath=args.netpath, port=args.port, config_path=args.config_path,
        debug_images=args.debug_images, return_meshes=args.return_meshes, device=args.device)
    server.start()
