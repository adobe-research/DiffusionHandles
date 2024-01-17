import torch
import torchvision

import scipy.ndimage

from lang_sam import LangSAM

from diffhandles.zoe_depth_estimator import ZoeDepthEstimator
from diffhandles.stable_null_inverter import StableNullInverter
from diffhandles.stable_diffuser import StableDiffuser
from diffhandles.lama_inpainter import LamaInpainter
from diffhandles.depth_transform import transform_depth
from diffhandles.utils import solve_laplacian_depth

class ImageEditor:

    def __init__(self):

        # Zoe Depth Estimator, for estimating the depth of the input image
        # https://github.com/isl-org/ZoeDepth
        self.depth_estimator_class = ZoeDepthEstimator

        # Language Segment Anything Model, for selecting the foreground object
        # https://github.com/luca-medeiros/lang-segment-anything
        self.foreground_segmenter_class = LangSAM

        # TODO: use a single diffuser model for inversion and inference
        # TODO: even when not using a single diffuser model, make sure versions used for inversion and guided inference match
        self.diffuser_class = StableDiffuser
        self.diffuser = None
        # self.diffuser = None
        # self.inverter = NullInversion(self.diffuser)

        # LaMa, for removing the foreground object from the input image
        # https://github.com/advimman/lama
        self.inpainter_class = LamaInpainter

        self.img_res = 512

        # input image, its depth map, its depth point cloud, and a corresponding prompt
        self.img = None
        self.depth = None
        self.pts = None
        self.prompt = None

        # background image (input image with removed foreground object) its depth map, its depth point cloud, and a corresponding prompt
        self.bg_pts = None
        self.bg_img = None
        self.bg_depth = None
        self.bg_phrase = None

        # foreground object mask and point cloud, and a corresponding prompt
        self.fg_mask = None
        self.fg_phrase = None

        # inverted input image, can be used to reconstruct the input image with the diffuser
        self.inverted_noise = None
        self.inverted_null_text = None

        # intermediate features to use as guidance
        self.attentions = None
        self.activations = None
        self.activations2 = None
        self.activations3 = None

        self.device = torch.device('cpu')

    def to(self, device: torch.device = None):
        # if self.depth_estimator is not None:
        #     self.depth_estimator.to(device=device)

        if self.diffuser is not None:
            self.diffuser.to(device=device)

        # if self.foreground_segmenter is not None:
        #     self.foreground_segmenter.sam.model.to(device=device)
        #     self.foreground_segmenter.device = device

        # if self.inpainter is not None:
        #     self.inpainter.to(device=device)

        self.device = device

    ### High-level functions ###

    def save(self, filename):
        state = {
            "depth_estimator_class": self.depth_estimator_class,
            "foreground_segmenter_class": self.foreground_segmenter_class,
            "diffuser_class": self.diffuser_class,
            "has_diffuser": self.diffuser is not None,
            "inpainter_class": self.inpainter_class,
            "img_res": self.img_res,
            "img": self.img,
            "depth": self.depth,
            "pts": self.pts,
            "prompt": self.prompt,
            "bg_pts": self.bg_pts,
            "bg_img": self.bg_img,
            "bg_depth": self.bg_depth,
            "bg_phrase": self.bg_phrase,
            "fg_mask": self.fg_mask,
            "fg_phrase": self.fg_phrase,
            "inverted_noise": self.inverted_noise,
            "inverted_null_text": self.inverted_null_text,
            "attentions": self.attentions,
            "activations": self.activations,
            "activations2": self.activations2,
            "activations3": self.activations3,
            "device": self.device,
        }
        torch.save(state, filename)
    
    @staticmethod
    def load(filename):
        image_editor = ImageEditor()
        
        state = torch.load(filename)
        
        image_editor.depth_estimator_class = state["depth_estimator_class"]
        image_editor.foreground_segmenter_class = state["foreground_segmenter_class"]
        image_editor.diffuser_class = state["diffuser_class"]
        image_editor.inpainter_class = state["inpainter_class"]
        image_editor.img_res = state["img_res"]
        image_editor.img = state["img"]
        image_editor.depth = state["depth"]
        image_editor.pts = state["pts"]
        image_editor.prompt = state["prompt"]
        image_editor.bg_pts = state["bg_pts"]
        image_editor.bg_img = state["bg_img"]
        image_editor.bg_depth = state["bg_depth"]
        image_editor.bg_phrase = state["bg_phrase"]
        image_editor.fg_mask = state["fg_mask"]
        image_editor.fg_phrase = state["fg_phrase"]
        image_editor.inverted_noise = state["inverted_noise"]
        image_editor.inverted_null_text = state["inverted_null_text"]
        image_editor.attentions = state["attentions"]
        image_editor.activations = state["activations"]
        image_editor.activations2 = state["activations2"]
        image_editor.activations3 = state["activations3"]
        image_editor.device = state["device"]

        # diffuser is not saved to file, so it must be recreated
        if state["has_diffuser"]:
            image_editor.diffuser = image_editor.diffuser_class(custom_unet=True)
            image_editor.diffuser.to(image_editor.device)

        return image_editor

    def edit_image(self, img: torch.Tensor, prompt: str, fg_phrase: str, bg_phrase: str = "", rot_angle: float = None, rot_axis: torch.Tensor = None, translation: torch.Tensor = None):
        self.set_input_image(img=img, prompt=prompt)
        self.select_foreground(fg_phrase=fg_phrase, bg_phrase=bg_phrase)
        return self.transform_foreground(rot_angle=rot_angle, rot_axis=rot_axis, translation=translation)

    def set_input_image(self, img: torch.Tensor, prompt: str = ""):

        # remove information about previous image
        if self.diffuser is not None:
            del self.diffuser
        self.diffuser = None
        self.bg_pts = None
        self.bg_img = None
        self.bg_depth = None
        self.bg_phrase = None
        self.fg_mask = None
        self.fg_phrase = None
        self.attentions = None
        self.activations = None
        self.activations2 = None
        self.activations3 = None

        # check image resolution
        if img.shape[-2:] != (self.img_res, self.img_res):
            raise ValueError(f"Image must be of size {self.img_res}x{self.img_res}.")

        self.img = img
        self.prompt = prompt

        # esimate depth
        # self.depth = np.array(Image.fromarray(self.depth_estimator(img)))
        depth_estimator = ZoeDepthEstimator()
        depth_estimator.to(self.device)
        with torch.no_grad():
            self.depth = depth_estimator.estimate_depth(img=self.img)
        del depth_estimator

        # get point cloud from depth
        self.pts = self.depth_estimator_class.depth_to_points(self.depth)

    def select_foreground(self, fg_phrase: str, bg_phrase: str = ""):

        if self.depth is None:
            raise ValueError("Input image must be set before selecting foreground.")

        self.fg_phrase = fg_phrase
        self.bg_phrase = bg_phrase

        # get foreground object mask
        self.fg_mask = self.foreground_mask(fg_prompt=self.fg_phrase)

        # remove foreground object to get background image
        self.bg_img = self.remove_foreground(img=self.img, foreground_mask=self.fg_mask)

        # estimate depth from background image
        depth_estimator = ZoeDepthEstimator()
        depth_estimator.to(self.device)
        with torch.no_grad():
            initial_bg_depth = depth_estimator.estimate_depth(img=self.bg_img)
        del depth_estimator
        # initial_bg_depth = np.array(Image.fromarray(self.model_zoe_nk.infer_pil(self.bg_img)))

        # infill hole in the depth of the input image (where the foreground object used to be)
        # with the depth of the background image
        self.bg_depth = solve_laplacian_depth(
            self.depth[0, 0].cpu().numpy(),
            initial_bg_depth[0, 0].cpu().numpy(),
            scipy.ndimage.binary_dilation(self.fg_mask[0, 0].cpu().numpy(), iterations=15))
        self.bg_depth = torch.from_numpy(self.bg_depth).to(device=self.device)[None, None]

        # get point cloud from background depth
        self.bg_pts = self.depth_estimator_class.depth_to_points(self.bg_depth)

        # Need to 3d-transform the depth with an identity transform
        # to get the same depth format as in the guided inference pass
        # where transformed depth is used.
        # (The depth is actually converted to disparity, which the stable diffuser expects as input.)
        # TODO: can we directly call `points_to_depth_merged` here to transform `self.pts` directly to disparity
        # and avoid all the overhead of transforming the point cloud? This would also allow us to do this only once
        # when loading the image instead of each time the foreground is transformed.
        depth, target_mask, correspondences = transform_depth(
            pts=self.pts, bg_pts=self.bg_pts, fg_mask=self.fg_mask,
            intrinsics=self.diffuser_class.get_depth_intrinsics(h=self.img_res, w=self.img_res),
            img_res=self.img_res,
            rot_angle=0.0,
            rot_axis=torch.tensor([0.0, 1.0, 0.0]),
            translation=torch.tensor([0.0, 0.0, 0.0]),
            )

        # invert image to get noise and null text that can be used to reproduce the image
        diffuser = self.diffuser_class(custom_unet=False)
        diffuser.to(self.device)
        inverter = StableNullInverter(diffuser)
        _, self.inverted_noise, self.inverted_null_text = inverter.invert(
            target_img=self.img, depth=depth, prompt=self.prompt, num_inner_steps=5 ,verbose=True)
        del inverter
        del diffuser

        # perform first diffusion inference pass to get intermediate features
        # TODO: Can this be done once when loading the image? Are phrases needed here?
        # Can foreground and background features be separated after computing this forward pass, so that phrases are not needed in the forward pass?
        self.diffuser = self.diffuser_class(custom_unet=True)
        self.diffuser.to(self.device)
        with torch.no_grad():
            self.attentions, self.activations, self.activations2, self.activations3 = self.diffuser.initial_inference(
                latents=self.inverted_noise, depth=depth, uncond_embeddings=self.inverted_null_text,
                prompt=self.prompt, phrases=[self.fg_phrase, self.bg_phrase])

        # torchvision.utils.save_image(recon_img, '../../data/test/recon_img.png') # TEMP!

    def transform_foreground(self, rot_angle: float = None, rot_axis: torch.Tensor = None, translation: torch.Tensor = None):

        if self.activations is None:
            raise ValueError("Foreground must be selected before transforming it.")

        # 3d-transform depth
        transformed_depth, target_mask, correspondences = transform_depth(
            pts=self.pts, bg_pts=self.bg_pts, fg_mask=self.fg_mask,
            intrinsics=self.diffuser_class.get_depth_intrinsics(h=self.img_res, w=self.img_res),
            img_res=self.img_res,
            rot_angle=rot_angle,
            rot_axis=rot_axis,
            translation=translation)

        if transformed_depth.shape[-2:] != (self.img_res, self.img_res):
            raise ValueError(f"Transformed depth must be of size {self.img_res}x{self.img_res}.")

        # perform second diffusion inference pass guided by the 3d-transformed features
        with torch.no_grad():
            output_img = self.diffuser.guided_inference(
                latents=self.inverted_noise, depth=transformed_depth, uncond_embeddings=self.inverted_null_text,
                prompt=self.prompt, phrases=[self.fg_phrase, self.bg_phrase],
                attention_maps_orig=self.attentions, activations_orig=self.activations, activations2_orig=self.activations2, activations3_orig=self.activations3,
                correspondences=correspondences)

        return output_img

    ### Lower-level functions ###

    def foreground_mask(self, fg_prompt: str):
        foreground_segmenter = self.foreground_segmenter_class()
        foreground_segmenter.sam.model.to(self.device)
        foreground_segmenter.device = self.device
        # with torch.no_grad():
        masks, boxes, phrases, logits = foreground_segmenter.predict(
            image_pil=torchvision.transforms.functional.to_pil_image(self.img[0]),
            text_prompt=fg_prompt)
        del foreground_segmenter
        mask = masks[0, None, None, :, :].to(device=self.device, dtype=torch.float32)
        return mask

    def remove_foreground(self, img, foreground_mask):
        # dilate the foreground mask
        dilate_amount = 2
        foreground_mask = foreground_mask.cpu().numpy() > 0.5
        foreground_mask = scipy.ndimage.binary_dilation(foreground_mask[0, 0], iterations=dilate_amount)[None, None, ...]
        foreground_mask = torch.from_numpy(foreground_mask).to(device=self.device, dtype=torch.float32)
        
        # inpaint the foreground region to remove the foreground
        inpainter = self.inpainter_class()
        inpainter.to(self.device)
        bg_img = inpainter.inpaint(image=img, mask=foreground_mask)
        del inpainter
        return bg_img
