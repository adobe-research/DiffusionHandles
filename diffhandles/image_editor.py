import numpy as np
import torch
import torchvision
from PIL import Image
import cv2
import scipy.ndimage

from lang_sam import LangSAM

from diffhandles.zoe_depth_estimator import ZoeDepthEstimator
from diffhandles.null_inversion import NullInversion
from diffhandles.stable_diffuser import StableDiffuser
from diffhandles.lama_inpainter import LamaInpainter
from diffhandles.utils import max_pool_numpy, poisson_solve, transform_point_cloud, solve_laplacian_depth, pack_correspondences

class ImageEditor:

    def __init__(self):

        # Zoe Depth Estimator, for estimating the depth of the input image
        # https://github.com/isl-org/ZoeDepth
        self.depth_estimator = ZoeDepthEstimator()
        
        # Language Segment Anything Model, for selecting the foreground object
        # https://github.com/luca-medeiros/lang-segment-anything
        self.foreground_segmenter = LangSAM()

        # TODO: use a single diffuser model for inversion and inference
        # TODO: even when not using a single diffuser model, make sure versions used for inversion and inference match
        self.diffuser = None
        # self.inverter = NullInversion(self.diffuser)

        # LaMa, for removing the foreground object from the input image
        # https://github.com/advimman/lama
        self.inpainter = LamaInpainter()

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

        self.device = self.depth_estimator.device

    def to(self, device: torch.device = None):
        if self.depth_estimator is not None:
            self.depth_estimator.to(device=device)

        if self.diffuser is not None:
            self.diffuser.to(device=device)

        if self.foreground_segmenter is not None:
            self.foreground_segmenter.sam.model.to(device=device)
            self.foreground_segmenter.device = device
        
        if self.inpainter is not None:
            self.inpainter.to(device=device)

        self.device = device

    ### High-level functions ###

    def edit_image(self, img: torch.Tensor, prompt: str, fg_phrase: str, bg_phrase: str = "", rot_angle: float = None, rot_axis: torch.Tensor = None, translation: torch.Tensor = None):
        self.set_input_image(img=img, prompt=prompt)
        self.select_foreground(fg_phrase=fg_phrase, bg_phrase=bg_phrase)
        return self.transform_foreground(rot_angle=rot_angle, rot_axis=rot_axis, translation=translation)

    def set_input_image(self, img: torch.Tensor, prompt: str = ""):

        # check image resolution
        if img.shape[-2:] != (self.img_res, self.img_res):
            raise ValueError(f"Image must be of size {self.img_res}x{self.img_res}.")

        self.prompt = prompt

        # esimate depth
        # self.depth = np.array(Image.fromarray(self.depth_estimator(img)))
        self.depth = self.depth_estimator(img)

        # get point cloud from depth
        self.pts = self.depth_estimator.depth_to_points(self.depth[None])

        diffuser = StableDiffuser()
        inverter = NullInversion(diffuser)
        _, self.inverted_noise, self.inverted_null_text = inverter.invert(
            target_img=self.img, depth=self.depth, prompt=self.prompt, num_inner_steps=5 ,verbose=True)
        # del diffuser
        # del inverter

    def select_foreground(self, fg_phrase: str, bg_phrase: str = ""):
        self.fg_phrase = fg_phrase
        self.bg_phrase = bg_phrase
        
        # get foreground object mask
        self.fg_mask = self.foreground_mask(fg_prompt=self.fg_phrase)

        # remove foreground object to get background image
        self.bg_img = self.remove_foreground(img=self.img, foreground_mask=self.fg_mask)

        # estimate depth from background image
        initial_bg_depth = np.array(Image.fromarray(self.model_zoe_nk.infer_pil(self.bg_img)))

        # infill hole in the depth of the input image (where the foreground object used to be)
        # with the depth of the background image
        self.bg_depth = solve_laplacian_depth(
            self.depth,
            initial_bg_depth,
            scipy.ndimage.binary_dilation(self.fg_mask, iterations=15))

        # get point cloud from background depth
        self.bg_pts = self.depth_estimator.depth_to_points(self.bg_depth[None])

        # perform first diffusion inference pass to get intermediate features
        # TODO: Can this be done once when loading the image? Are phrases needed here?
        # Can foreground and background features be separated after computing this forward pass, so that phrases are not needed in the forward pass?
        self.diffuser = StableDiffuser(custom_unet=True)
        self.diffuser.to(self.device)
        self.attentions, self.activations, self.activations2, self.activations3 = self.diffuser.initial_inference(
            latents=self.inverted_noise, depth=self.depth, uncond_embeddings=self.inverted_null_text,
            prompt=self.prompt, phrases=[self.fg_phrase, self.bg_phrase])

    def transform_foreground(self, rot_angle: float = None, rot_axis: torch.Tensor = None, translation: torch.Tensor = None):

        # 3d-transform depth and features
        transformed_depth, target_mask, correspondences = self.transform_depth(
            rot_angle=rot_angle, rot_axis=rot_axis, translation=translation)

        if transformed_depth.shape[-2:] != (self.img_res, self.img_res):
            raise ValueError(f"Transformed depth must be of size {self.img_res}x{self.img_res}.")

        # perform second diffusion inference pass guided by the 3d-transformed features
        output_img = self.diffuser.guided_inference(
            latents=self.inverted_noise, depth=transformed_depth, uncond_embeddings=self.inverted_null_text,
            prompt=self.prompt, phrases=[self.fg_phrase, self.bg_phrase],
            attention_maps_orig=self.attentions, activations_orig=self.activations, activations2_orig=self.activations2, activations3_orig=self.activations3,
            correspondences=correspondences)

        return output_img

    ### Lower-level functions ###

    def transform_depth(self, rot_angle: float = None, rot_axis: torch.Tensor = None, translation: torch.Tensor = None):

        pts, mod_ids = transform_point_cloud(
            pts=self.pts,
            axis=np.array(rot_axis),
            angle_degrees=rot_angle,
            x=translation[0].item(),
            y=translation[1].item(),
            z=translation[2].item(),
            mask=self.fg_mask)

        #points = pts.reshape((self.img_res**2, 3))

        #orig_pts = pts


        if isinstance(mod_ids, np.ndarray):
            mod_ids = torch.from_numpy(mod_ids)

        #reproject points to depth map

        reshaped_bg_pts = self.bg_pts.reshape((self.img_res**2, 3))

        reshaped_pts = pts.reshape((self.img_res**2, 3))

        new_mod_ids = np.zeros(len(reshaped_bg_pts) + len(reshaped_pts[mod_ids]), dtype = np.uint8)

        new_mod_ids[np.arange(new_mod_ids.size) > len(reshaped_bg_pts) - 1] = 1

        modded_id_list = np.where(mod_ids)[0]

        idx_to_coord = {}

        for idx in modded_id_list:
            pt = reshaped_pts[idx]
            reshaped_bg_pts = np.vstack((reshaped_bg_pts, pt))
            idx_to_coord[len(reshaped_bg_pts) - 1] = divmod(idx, self.img_res)

        (rendered_depth, occluded_pixels, target_mask,
         transformed_positions_x, transformed_positions_y, orig_visibility_mask
         )  = self.depth_estimator.points_to_depth_merged(
            points=reshaped_bg_pts,
            mod_ids=torch.from_numpy(new_mod_ids),
            output_size=(self.img_res, self.img_res),
            max_depth_value=reshaped_bg_pts[:, 2].max()
        )

        #plot_img(rendered_depth)

        infer_visible_original = np.zeros_like(mod_ids.reshape((self.img_res,self.img_res)), dtype = np.uint8)

        original_idxs = [idx_to_coord[key] for key in np.where(orig_visibility_mask)[0]]

        for idx in original_idxs:
            infer_visible_original[idx] = 1

        original_positions_y, original_positions_x = np.where(infer_visible_original)

        #target_mask_binary = target_mask.to(torch.int)
        # Convert the target mask to uint8
        #target_mask_uint8 = target_mask_binary.detach().cpu().numpy().astype(np.uint8) * 255

        target_mask_uint8 = target_mask.astype(np.uint8)*255

        # Define a kernel for the closing operation (you can adjust the size and shape)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.img_res // 250 , self.img_res // 250))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.img_res // 50 , self.img_res // 50))

        # Perform the closing operation
        #target_mask_cleaned = cv2.morphologyEx(target_mask_uint8, cv2.MORPH_CLOSE, kernel)

        target_mask_cleaned = target_mask_uint8

        #target_mask_cleaned = cv2.medianBlur(target_mask_cleaned, 3)
        #target_mask_cleaned = cv2.medianBlur(target_mask_cleaned, 3)


        # Perform the closing operation
        #target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_OPEN, open_kernel)


        # Perform the closing operation
        target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_CLOSE, kernel)

        # Perform the closing operation
        target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_OPEN, open_kernel)


        # Perform the closing operation
        #target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_CLOSE, kernel)



        # Filter correspondences based on the mask
        filtered_original_x = []
        filtered_original_y = []
        filtered_transformed_x = []
        filtered_transformed_y = []

        for ox, oy, tx, ty in zip(original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y):
            if target_mask_cleaned[ty, tx] == 255:  # if the original point lies within the mask
                filtered_original_x.append(ox)
                filtered_original_y.append(oy)
                filtered_transformed_x.append(tx)
                filtered_transformed_y.append(ty)

        original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y = np.array(filtered_original_x), np.array(filtered_original_y), np.array(filtered_transformed_x), np.array(filtered_transformed_y)

        #save_positions(original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y, save_path + 'positions.npy')

        # correspondences = np.stack((original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y), axis=-1)
        correspondences = pack_correspondences(
            torch.from_numpy(original_positions_x),
            torch.from_numpy(original_positions_y),
            torch.from_numpy(transformed_positions_x),
            torch.from_numpy(transformed_positions_y))
            
        # img_tensor = np.array(cut_img)
        img_tensor = np.array(self.fg_mask) # TODO: check that this works
        ref_mask = (img_tensor[:, :] == 255)
        mask = np.zeros_like(ref_mask, dtype = np.uint8)
        mask[ref_mask.nonzero()] = 255
        mask = max_pool_numpy(mask, 512 // self.img_res)
        occluded_mask = mask

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        occluded_mask = cv2.dilate(occluded_mask.astype(np.uint8), kernel)

        #rendered_depth = rendered_depth.squeeze()
        #bg_img = bg_img[:,:,0]

        #visualize_img(rendered_depth.detach().cpu().numpy(), save_path + 'rendered_depth')

        # visualize_img(target_mask_cleaned, 'clean_target_mask')
        # visualize_img(target_mask_uint8, 'init_target_mask')

        #plot_img(target_mask_cleaned)


        noise_mask = target_mask_uint8.astype(int) - target_mask_cleaned.astype(int)

        final_mask = target_mask_cleaned.astype(int) - target_mask_uint8.astype(int)
        final_mask[final_mask < 0] = 0
        noise_mask[noise_mask < 0] = 0

        #plot_img(final_mask)

        inpaint_mask = final_mask + noise_mask #+ occluded_mask
        inpaint_mask = (inpaint_mask > 0).astype(np.uint8)

        # visualize_img(inpaint_mask, 'inpaint_mask')

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        inpaint_mask_dilated = cv2.dilate(inpaint_mask, kernel)

        lap_inpainted_depth_map = poisson_solve(np.array(rendered_depth), inpaint_mask_dilated)

        #lap_inpainted_depth_map[np.where(target_mask_cleaned == 0)] = 1 - bg_img[np.where(target_mask_cleaned == 0)]

        img = lap_inpainted_depth_map
        img = (img - img.min())/(img.max() - img.min())
        img = (img*255).round().astype("uint8")

        #plot_img(img)

        #visualize_img(img,'fixed_depth_map_denoised')

        img = target_mask_cleaned
        img = (img - img.min())/(img.max() - img.min())
        img = (img*255).round().astype("uint8")

        return lap_inpainted_depth_map, target_mask_cleaned, correspondences

    def foreground_mask(self, fg_prompt: str):
        masks, boxes, phrases, logits = self.foreground_segmenter.predict(
            image_pil=torchvision.transforms.functional.to_pil_image(self.img),
            text_prompt=fg_prompt)
        return masks[0]

    def remove_foreground(self, img, foreground_mask):
        return self.inpainter.inpaint(img=img, mask=foreground_mask)

    def invert_image(self, img):
        pass


