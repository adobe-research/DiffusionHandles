import torch

import scipy.ndimage

from diffhandles.zoe_depth_estimator import ZoeDepthEstimator
from diffhandles.stable_null_inverter import StableNullInverter
from diffhandles.stable_diffuser import StableDiffuser
from diffhandles.depth_transform import transform_depth, points_to_depth_merged
from diffhandles.utils import solve_laplacian_depth

class DiffusionHandles:

    def __init__(self):

        # TODO: use a single diffuser model for inversion and inference
        # TODO: even when not using a single diffuser model, make sure versions used for inversion and guided inference match
        self.diffuser_class = StableDiffuser
        self.diffuser = None
        # self.diffuser = None
        # self.inverter = NullInversion(self.diffuser)

        self.img_res = 512

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

    def edit_image(
            self, img: torch.Tensor, depth: torch.Tensor, fg_mask:torch.Tensor, bg_depth: torch.Tensor,
            prompt: str, fg_phrase: str, bg_phrase: str,
            rot_angle: float = None, rot_axis: torch.Tensor = None, translation: torch.Tensor = None):
        """
        Edit an image by moving the foreground object.
        Args:
            img: Input image.
            depth: Depth of the input image.
            fg_mask: A mask of the foreground object.
            bg_depth: Depth of the background (with removed foreground object).
            prompt: Full prompt for the input image.
            fg_phrase: The phrase in the prompt describing the foreground object.
            bg_phrase: The phrase in the prompt describing the background object.
            rot_angle: Rotation angle in degrees.
            rot_axis: Rotation axis.
            translation: Translation vector.

        Returns:
            output_img: The edited image.
        """

        inverted_noise, inverted_null_text, bg_depth, attentions, activations, activations2, activations3 = self.select_foreground(
            img=img, depth=depth, fg_mask=fg_mask, bg_depth=bg_depth,
            prompt=prompt, fg_phrase=fg_phrase, bg_phrase=bg_phrase)

        output_img = self.transform_foreground(
            depth=depth, fg_mask=fg_mask, bg_depth=bg_depth,
            prompt=prompt, fg_phrase=fg_phrase, bg_phrase=bg_phrase,
            inverted_null_text=inverted_null_text, inverted_noise=inverted_noise, 
            attentions=attentions, activations=activations, activations2=activations2, activations3=activations3,
            rot_angle=rot_angle, rot_axis=rot_axis, translation=translation)
        
        return output_img

    def select_foreground(
            self, img: torch.Tensor, depth: torch.Tensor, fg_mask:torch.Tensor, bg_depth: torch.Tensor,
            prompt: str, fg_phrase: str, bg_phrase: str, testing: bool = False):
        """
        Select the foreground object in the image. The following steps are performed:
        1) The background depth is updated by infilling the hole in the depth of the input image
        2) The image is inverted to get noise and null text that can be used to reproduce the image
        3) The image is diffused to get intermediate features (starting from the inverted noise and null text)

        Args:
            img: Input image.
            depth: Depth of the input image.
            fg_mask: A mask of the foreground object.
            bg_depth: Depth of the background (with removed foreground object).
            prompt: Full prompt for the input image.
            fg_phrase: The phrase in the prompt describing the foreground object.
            bg_phrase: The phrase in the prompt describing the background object.

        Returns:
            inverted_noise: The noise of the inverted input image.
            inverted_null_text: The null text of the inverted input image.
            bg_depth: An updated background depth that has been adjusted to better match the depth of the input image.
            attentions: Attention maps from the first diffusion inference pass.
            activations: Layer 1 activations from the first diffusion inference pass (from layer 1 of the decoder of the UNet).
            activations2: Layer 2 activations from the first diffusion inference pass (from layer 2 of the decoder of the UNet).
            activations3: Layer 3 activations from the first diffusion inference pass (from layer 3 of the decoder of the UNet).
        """

        # infill hole in the depth of the input image (where the foreground object used to be)
        # with the depth of the background image
        bg_depth = solve_laplacian_depth(
            depth[0, 0].cpu().numpy(),
            bg_depth[0, 0].cpu().numpy(),
            scipy.ndimage.binary_dilation(fg_mask[0, 0].cpu().numpy(), iterations=15))
        bg_depth = torch.from_numpy(bg_depth).to(device=self.device)[None, None]

        # get point clouds from depths
        bg_pts = ZoeDepthEstimator.depth_to_points(bg_depth)
        pts = ZoeDepthEstimator.depth_to_points(depth)

        if testing:
            # Testing depth for inversion that does not require foreground selection
            # (so that inversion can be done only once when loading the input image).
            # Gives only slightly different results. Leaving it out for now as it is less well tested.
            (depth, occluded_pixels, target_mask, transformed_positions_x, transformed_positions_y, orig_visibility_mask, depth_minmax) = points_to_depth_merged(
                points=pts.view(-1, 3),
                mod_ids=torch.zeros(size=(pts.view(-1, 3).shape[0],), device=pts.device, dtype=torch.uint8),
                intrinsics=self.diffuser_class.get_depth_intrinsics(h=self.img_res, w=self.img_res),
                output_size=(self.img_res, self.img_res),
                max_depth_value=pts.view(-1, 3)[:, 2].max(),
                depth_minmax=None
            )
            depth = torch.from_numpy(depth).unsqueeze(dim=0).unsqueeze(dim=0).to(device=img.device, dtype=torch.float32)
        else:
            # Need to 3d-transform the depth with an identity transform
            # to get the same depth format as in the guided inference pass
            # where transformed depth is used.
            # (The depth is actually converted to disparity, which the stable diffuser expects as input.)
            # TODO: can we directly call `points_to_depth_merged` here to transform `self.pts` directly to disparity
            # and avoid all the overhead of transforming the point cloud? This would also allow us to do this only once
            # when loading the image instead of each time the foreground is transformed.
            depth, target_mask, correspondences = transform_depth(
                pts=pts, bg_pts=bg_pts, fg_mask=fg_mask,
                intrinsics=self.diffuser_class.get_depth_intrinsics(h=self.img_res, w=self.img_res),
                img_res=self.img_res,
                rot_angle=0.0,
                rot_axis=torch.tensor([0.0, 1.0, 0.0]),
                translation=torch.tensor([0.0, 0.0, 0.0]),
                )
            depth_minmax = None

        # invert image to get noise and null text that can be used to reproduce the image
        diffuser = self.diffuser_class(custom_unet=False)
        diffuser.to(self.device)
        inverter = StableNullInverter(diffuser)
        _, inverted_noise, inverted_null_text = inverter.invert(
            target_img=img, depth=depth, prompt=prompt, num_inner_steps=5 ,verbose=True)
        del inverter
        del diffuser

        # perform first diffusion inference pass to get intermediate features
        # TODO: Can this be done once when loading the image? Are phrases needed here?
        # Can foreground and background features be separated after computing this forward pass, so that phrases are not needed in the forward pass?
        self.diffuser = self.diffuser_class(custom_unet=True)
        self.diffuser.to(self.device)
        with torch.no_grad():
            attentions, activations, activations2, activations3 = self.diffuser.initial_inference(
                latents=inverted_noise, depth=depth, uncond_embeddings=inverted_null_text,
                prompt=prompt, phrases=[fg_phrase, bg_phrase])

        return inverted_noise, inverted_null_text, bg_depth, attentions, activations, activations2, activations3, depth_minmax
    
    def transform_foreground(
            self, depth: torch.Tensor, fg_mask:torch.Tensor, bg_depth: torch.Tensor,
            prompt: str, fg_phrase: str, bg_phrase: str,
            inverted_null_text: torch.Tensor, inverted_noise: torch.Tensor, 
            attentions: torch.Tensor, activations: torch.Tensor, activations2: torch.Tensor, activations3: torch.Tensor,
            rot_angle: float = None, rot_axis: torch.Tensor = None, translation: torch.Tensor = None,
            testing=False, depth_minmax=None):
        """
        Move the foreground object. The following steps are performed:
        1) The depth of the foreground object and the intermediate features are 3D-transformed
        2) The edited image is generated guided by the 3D-transformed intermediate features

        Args:
            depth: Depth of the input image.
            fg_mask: A mask of the foreground object.
            bg_depth: Depth of the background (with removed foreground object).
            prompt: Full prompt for the input image.
            fg_phrase: The phrase in the prompt describing the foreground object.
            bg_phrase: The phrase in the prompt describing the background object.
            inverted_null_text: The null text of the inverted input image.
            inverted_noise: The noise of the inverted input image.
            attentions: Attention maps from the first diffusion inference pass.
            activations: Layer 1 activations from the first diffusion inference pass (from layer 1 of the decoder of the UNet).
            activations2: Layer 2 activations from the first diffusion inference pass (from layer 2 of the decoder of the UNet).
            activations3: Layer 3 activations from the first diffusion inference pass (from layer 3 of the decoder of the UNet).
            rot_angle: Rotation angle in degrees.
            rot_axis: Rotation axis.
            translation: Translation vector.
        
        Returns:
            output_img: The edited image.
        """
        
        # get point clouds from depths
        bg_pts = ZoeDepthEstimator.depth_to_points(bg_depth)
        pts = ZoeDepthEstimator.depth_to_points(depth)

        
        # 3d-transform depth
        transformed_depth, target_mask, correspondences = transform_depth(
            pts=pts, bg_pts=bg_pts, fg_mask=fg_mask,
            intrinsics=self.diffuser_class.get_depth_intrinsics(h=self.img_res, w=self.img_res),
            img_res=self.img_res,
            rot_angle=rot_angle,
            rot_axis=rot_axis,
            translation=translation,
            depth_minmax=depth_minmax)

        if transformed_depth.shape[-2:] != (self.img_res, self.img_res):
            raise ValueError(f"Transformed depth must be of size {self.img_res}x{self.img_res}.")

        # perform second diffusion inference pass guided by the 3d-transformed features
        with torch.no_grad():
            output_img = self.diffuser.guided_inference(
                latents=inverted_noise, depth=transformed_depth, uncond_embeddings=inverted_null_text,
                prompt=prompt, phrases=[fg_phrase, bg_phrase],
                attention_maps_orig=attentions, activations_orig=activations, activations2_orig=activations2, activations3_orig=activations3,
                correspondences=correspondences)

        return output_img
