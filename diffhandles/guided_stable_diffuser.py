import math
import inspect
from typing import List
from packaging import version

import torch
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.configuration_utils import FrozenDict
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from diffhandles.model.unet_2d_condition import UNet2DConditionModel # this is the custom UNet that can also return intermediate activations and attentions
from diffhandles.guided_diffuser import GuidedDiffuser
from diffhandles.utils import normalize_attn_torch, unpack_correspondences
from diffhandles.losses import compute_localized_transformed_appearance_loss, compute_background_loss

class GuidedStableDiffuser(GuidedDiffuser):
    def __init__(self, conf):
        super().__init__(conf=conf)

        if self.conf.use_depth:
            model_name = "stabilityai/stable-diffusion-2-depth"
        else:       
            model_name = "stabilityai/stable-diffusion-2-1"

        self.scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", save_activations=True)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")

        self.device = self.unet.device

        # fix deprecated config file for some stable diffusion versions
        is_unet_version_less_0_9_0 = hasattr(self.unet.config, "_diffusers_version") and version.parse(
            version.parse(self.unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(self.unet.config, "sample_size") and self.unet.config.sample_size < 64
        is_stable_diffusion_2_depth = (
            hasattr(self.unet.config, "_name_or_path")
            and self.unet.config._name_or_path == "stabilityai/stable-diffusion-2-depth"
        )
        if (is_unet_version_less_0_9_0 and is_unet_sample_size_less_64) or is_stable_diffusion_2_depth:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate(
                "sample_size<64", "1.0.0", deprecation_message, standard_warn=False
            )
            new_unet_config = dict(self.unet.config)
            new_unet_config["sample_size"] = 64
            self.unet._internal_dict = FrozenDict(new_unet_config)
            self.unet.sample_size = 64

    def to(self, device: torch.device = None):

        device=torch.device(device)
        
        self.unet = self.unet.to(device=device)
        # self.tokenizer = self.tokenizer.to(device=device)
        self.text_encoder = self.text_encoder.to(device=device)
        self.vae = self.vae.to(device=device)

        self.device = device

        return self

    def get_image_shape(self):
        feat_shape = self.get_feature_shape()
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        return (feat_shape[0] * vae_scale_factor, feat_shape[1] * vae_scale_factor, 3)

    def get_feature_shape(self):
        feat_hw = self.unet.sample_size
        if isinstance(feat_hw, int):
            feat_hw = (feat_hw, feat_hw)
        return (feat_hw[0], feat_hw[1], self.unet.config.out_channels)
    
    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.tokenizer(
            [""], padding="max_length", max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return torch.cat([uncond_embeddings, text_embeddings])

    @torch.no_grad()
    def init_depth(self, depth):
        # resize depth map to match the size of the feature image (post vae encoding)
        h, w = self.get_feature_shape()[:2]
        # h, w = 64, 64 # TEMP! (comment in above)
        depth = torch.nn.functional.interpolate(
            depth,
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        
        # normalize depth to [0, 1]
        depth_min = torch.amin(depth, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth, dim=[1, 2, 3], keepdim=True)
        depth = 2.0 * (depth - depth_min) / (depth_max - depth_min) - 1.0

        return depth

    @staticmethod
    def get_depth_intrinsics(device: torch.device = None):
        """
        Return intrinsics suitable for the input depth.
        Intrinsics for a pinhole camera model.
        Assume fov of 55 degrees, a central principal point,
        and that the output coordinates on the image plane for the 55° fov  are in [-1, 1]^2
        """
        # # f = 0.5 * w / np.tan(0.5 * 6.24 * np.pi / 180.0) #car benchmark
        # #f = 0.5 * W / np.tan(0.5 * 7.18 * np.pi / 180.0) #airplane benchmark
        # #f = 0.5 * W / np.tan(0.5 * 14.9 * np.pi / 180.0) #chair, cup, lamp, stool benchmark        
        # #f = 0.5 * W / np.tan(0.5 * 7.23 * np.pi / 180.0) #plant benchmark            
        # f = 0.5 * w / np.tan(0.5 * 55 * np.pi / 180.0)
        # cx = 0.5 * w
        # cy = 0.5 * h

        fov = 55.0
        f = 1.0 / np.tan(0.5 * fov * (np.pi / 180.0))
        cx = 0.0
        cy = 0.0
        return torch.tensor([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
            ], dtype=torch.float32, device=device)

    def initial_inference(self, init_latents: torch.Tensor, depth: torch.Tensor, uncond_embeddings: torch.Tensor, prompt: str): #, phrases: List[str]):

        num_timesteps = 50
        strength = 1.0
        seed = 2773
        
        generator = torch.manual_seed(seed)  # Seed generator to create the inital latent noise - 305 for car, 105 for cup, 155 for lamp
        
        #Set timesteps
        self.scheduler.set_timesteps(num_timesteps, device=self.device)
        timesteps, num_inference_steps = self.get_timesteps(num_timesteps, strength)

        if self.conf.use_depth:
            depth = self.init_depth(depth)
        
        # Encode Prompt
        cond_input = self.tokenizer(
                [prompt],
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )

        cond_embeddings = self.text_encoder(cond_input.input_ids.to(self.device))[0]

        if uncond_embeddings is None:
            uncond_input = self.tokenizer(
                [""],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[[0]]
        if uncond_embeddings.shape[0] == 0:
            uncond_embeddings = uncond_embeddings.expand(len(timesteps), -1, -1, -1)
        
        if init_latents is None:
            # in_channels-1 because depth will be concatentated if depth is used
            num_latent_channels = self.unet.config.in_channels-1 if self.conf.use_depth else self.unet.config.in_channels
            init_latents = torch.zeros(
                size=[1, num_latent_channels, self.unet.config.sample_size, self.unet.config.sample_size],
                device=self.device, dtype=torch.float32)
            noise = randn_tensor(
                shape=[1, num_latent_channels, self.unet.config.sample_size, self.unet.config.sample_size],
                generator=generator, device=self.device, dtype=torch.float32)
            init_latents = self.scheduler.add_noise(init_latents, noise, timesteps[0])

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.0)

        # attention_list = []
        activation_list = [] 
        activation2_list = []
        activation3_list = []

        latents = init_latents
        
        # obj_number = 2
        for t_idx, t in enumerate(tqdm(timesteps)):
            with torch.no_grad():

                #Prepare latent variables
                latent_model_input = latents #torch.cat([latents]) #if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if self.conf.use_depth:
                    latent_model_input = torch.cat([latent_model_input, depth], dim=1)

                # predict the noise residual
                unet_output = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=cond_embeddings,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )
                
                noise_pred = unet_output[0]
                # attn_map_integrated_up = unet_output[1]
                # attn_map_integrated_mid = unet_output[2]
                # attn_map_integrated_down = unet_output[3]
                activations = unet_output[4]
                activations2 = unet_output[5]
                activations3 = unet_output[6]
                
                activation_list.append(activations[0])
                activation2_list.append(activations2[0])
                activation3_list.append(activations3[0])
                # attention_obj_list = []
                            
                # for m in range(obj_number):
                #     attentions = retrieve_attention_maps(attn_map_integrated_down, attn_map_integrated_mid, attn_map_integrated_up, m, object_positions, latents.shape[2])
                #     attention_obj_list.append(attentions)
                # attention_list.append(attention_obj_list)

                latent_model_input = torch.cat([latents]*2) #if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if self.conf.use_depth:
                    latent_model_input = torch.cat([latent_model_input, torch.cat([depth]*2, dim=0)], dim=1)


                text_embeddings = torch.cat([uncond_embeddings[t_idx].expand(*cond_embeddings.shape), cond_embeddings])

                # predict the noise residual
                unet_output = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )
                    
                noise_pred = unet_output[0]

                # perform guidance
                #if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]     
                torch.cuda.empty_cache()
        
        activations = [
            torch.stack(activation_list, dim=0),
            torch.stack(activation2_list, dim=0),
            torch.stack(activation3_list, dim=0)]
        
        return activations, latents, uncond_embeddings, init_latents

    def encode_latent_image(self, image: torch.Tensor) -> torch.Tensor:
        # image = VaeImageProcessor(vae_scale_factor=self.vae.config.scaling_factor).preprocess(image, output_type="pt")
        # latent_image = self.vae.encode(image).latent_dist.mode() * self.vae.config.scaling_factor
        # return latent_image
        # TODO: check that the code above works correctly and is the same thing that the StabelDiffusionDepth2Img Pipeline is doing
        # https://github.com/huggingface/diffusers/blob/v0.27.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_depth2img.py
        raise NotImplementedError
    
    def decode_latent_image(self, latent_image: torch.Tensor) -> torch.Tensor:
        image = self.vae.decode(latent_image / self.vae.config.scaling_factor, return_dict=False)[0]
        image = VaeImageProcessor(vae_scale_factor=self.vae.config.scaling_factor).postprocess(image, output_type="pt")
        return image
    
    def guided_inference(
            self, latents: torch.Tensor, depth: torch.Tensor, uncond_embeddings: torch.Tensor, prompt: str,
            activations_orig: list[torch.Tensor],
            correspondences: torch.Tensor, fg_weight: float = None, bg_weight: float = None, save_denoising_steps: bool = False):
        
        strength = 1.0
        num_timesteps = 50
        seed = 2773

        if fg_weight is None:
            fg_weight = self.conf.fg_weight
        if bg_weight is None:
            bg_weight = self.conf.bg_weight

        generator = torch.manual_seed(seed)

        #Set timesteps
        self.scheduler.set_timesteps(num_timesteps, device=self.device)
        timesteps, num_inference_steps = self.get_timesteps(num_timesteps, strength)
        
        processed_correspondences = self.process_correspondences(correspondences, img_res=depth.shape[-1])
        
        if self.conf.use_depth:
            depth = self.init_depth(depth)
        
        # Encode Prompt
        input_ids = self.tokenizer(
                [prompt],
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )

        cond_embeddings = self.text_encoder(input_ids.input_ids.to(self.device))[0]

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.0)

        loss = torch.tensor(10000)
        # timestep_num = 0
        
        if save_denoising_steps:
            denoising_steps = {
                'opt': [],
                'post-opt': [],
            }
        
        activations_orig, activations2_orig, activations3_orig = activations_orig[0], activations_orig[1], activations_orig[2]
        
        for t_idx, t in enumerate(tqdm(timesteps)):
            iteration = 0
            # attention_map_orig = attention_maps_orig[timestep_num]
            activation_orig = activations_orig[t_idx]
            activation2_orig = activations2_orig[t_idx]
            activation3_orig = activations3_orig[t_idx]
            torch.set_grad_enabled(True)

            activations_size = (activation3_orig.shape[-2], activation3_orig.shape[-1])

            if save_denoising_steps:
                denoising_steps['opt'].append([])
                        
            while iteration < 3 and (t_idx < 38 and t_idx >= 0):
                
                latent_model_input = latents
                latents = latents.requires_grad_(True)

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if self.conf.use_depth:
                    latent_model_input = torch.cat([latent_model_input, depth], dim=1)
                                
                # predict the noise residual
                unet_output = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=cond_embeddings,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )
                    
                noise_pred = unet_output[0]
                # attn_map_integrated_up = unet_output[1]
                # attn_map_integrated_mid = unet_output[2]
                # attn_map_integrated_down = unet_output[3]
                activations = unet_output[4]
                activations2 = unet_output[5]
                activations3 = unet_output[6]

                # loss = 0
                
                if(t_idx >= 0):
                    appearance_loss = 0.0
                    bg_loss = 0.0
                    if(t_idx % 3 == 0):
                        appearance_loss += 7.5 * compute_localized_transformed_appearance_loss(
                            activations=activations3[0], activations_orig=activation3_orig,
                            processed_correspondences=processed_correspondences,
                            attn_layer_low=6, attn_layer_high=7, patch_size=1, activations_size=activations_size)
                        bg_loss += 1.5 * compute_background_loss(
                            activations=activations3[0],
                            activations_orig=activation3_orig,
                            processed_correspondences=processed_correspondences,
                            attn_layer_low=6, attn_layer_high=7, activations_size=activations_size)

                    if(t_idx % 3 == 1):
                        appearance_loss += 5.0 * compute_localized_transformed_appearance_loss(
                            activations=activations2[0], activations_orig=activation2_orig,
                            processed_correspondences=processed_correspondences,
                            attn_layer_low=5, attn_layer_high=6, patch_size=1, activations_size=activations_size)
                        bg_loss += 1.5 * compute_background_loss(
                            activations=activations2[0], activations_orig=activation2_orig,
                            processed_correspondences=processed_correspondences,
                            attn_layer_low=5, attn_layer_high=6, activations_size=activations_size)

                    if(t_idx % 3 == 2):
                        appearance_loss += 5.0 * compute_localized_transformed_appearance_loss(
                            activations=activations2[0], activations_orig=activation2_orig,
                            processed_correspondences=processed_correspondences,
                            attn_layer_low=5, attn_layer_high=6, patch_size=1, activations_size=activations_size)
                        appearance_loss += 7.5 * compute_localized_transformed_appearance_loss(
                            activations=activations3[0], activations_orig=activation3_orig,
                            processed_correspondences=processed_correspondences,
                            attn_layer_low=6, attn_layer_high=7, patch_size=1, activations_size=activations_size)
                        bg_loss += 1.5 * compute_background_loss(
                            activations=activations3[0], activations_orig=activation3_orig,
                            processed_correspondences=processed_correspondences,
                            attn_layer_low=6, attn_layer_high=7, activations_size=activations_size)
                        bg_loss += 1.5 * compute_background_loss(
                            activations=activations2[0], activations_orig=activation2_orig,
                            processed_correspondences=processed_correspondences,
                            attn_layer_low=5, attn_layer_high=6, activations_size=activations_size)
                else:
                    appearance_loss = 0.1 * compute_localized_transformed_appearance_loss(
                        activations=activations3[0], activations_orig=activation3_orig,
                        attn_layer_low=6, attn_layer_high=7, patch_size=1, activations_size=activations_size)

                if(iteration == 0):
                    app_wt = 2.5
                    bg_wt = 1.25
                if(iteration == 1):
                    app_wt = 1.25
                    bg_wt = 2.5   
                if(iteration == 2):
                    app_wt = 1.25
                    bg_wt = 1.25
                if(iteration == 3):
                    app_wt = 2.5
                    bg_wt = 2.5

                # print('bg loss')
                # print(bg_loss)

                # print('appearance loss')
                # print(appearance_loss)

                loss = 30.0 * (fg_weight*app_wt*appearance_loss + bg_weight*bg_wt*bg_loss)

                # loss *= 30

                if(loss == 0):
                    grad_cond = 0
                else:
                    grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]
                            
                var = 0.1 
                latents = latents - 1.0 * grad_cond * var
                iteration += 1
                torch.cuda.empty_cache()

                if save_denoising_steps:
                    with torch.no_grad():
                        # image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                        # image = VaeImageProcessor(vae_scale_factor=self.vae.config.scaling_factor).postprocess(image, output_type="pt")
                        image = self.decode_latent_image(latents)
                        denoising_steps['opt'][-1].append(image.detach().cpu())
                
            torch.set_grad_enabled(False)            
            with torch.no_grad():

                latent_model_input = torch.cat([latents] * 2) #if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if self.conf.use_depth:
                    latent_model_input = torch.cat([latent_model_input, torch.cat([depth]*2, dim=0)], dim=1)

                text_embeddings = torch.cat([uncond_embeddings[t_idx].expand(*cond_embeddings.shape), cond_embeddings])

                # predict the noise residual
                unet_output = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )
                    
                noise_pred = unet_output[0]
                
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                torch.cuda.empty_cache()

                if save_denoising_steps:
                    # image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                    # image = VaeImageProcessor(vae_scale_factor=self.vae.config.scaling_factor).postprocess(image, output_type="pt")
                    image = self.decode_latent_image(latents)
                    denoising_steps['post-opt'].append(image.detach().cpu())
            # timestep_num += 1
                    
        with torch.no_grad():
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image = VaeImageProcessor(vae_scale_factor=self.vae.config.scaling_factor).postprocess(image, output_type="pt")
        
        if save_denoising_steps:
            return image, denoising_steps
        else:
            return image

    def process_correspondences(self, correspondences, img_res):

        original_x, original_y, transformed_x, transformed_y = unpack_correspondences(correspondences)

        # Since np.split creates arrays of shape (N, 1), we'll squeeze them to get back to shape (N,)
        original_x = original_x.squeeze()
        original_y = original_y.squeeze()
        transformed_x = transformed_x.squeeze()
        transformed_y = transformed_y.squeeze()
        
        original_mask = np.zeros((img_res, img_res))
        
        transformed_mask = np.zeros((img_res, img_res))
        
        visible_orig_x = []
        visible_orig_y = []
        visible_trans_x = []
        visible_trans_y = []    
        
        for x, y, tx, ty in zip(original_x, original_y, transformed_x, transformed_y):
            if((tx >= 0 and tx < img_res) and (ty >= 0 and ty < img_res)):
                visible_orig_x.append(x)
                visible_orig_y.append(y)
                visible_trans_x.append(tx)
                visible_trans_y.append(ty)
        
        for x, y in zip(visible_orig_x, visible_orig_y):
            original_mask[y,x] = 1

        for x, y in zip(visible_trans_x, visible_trans_y):
            transformed_mask[y,x] = 1        
        
        original_x, original_y, transformed_x, transformed_y = (
            np.array(visible_orig_x), np.array(visible_orig_y), np.array(visible_trans_x), np.array(visible_trans_y))

        # original_x, original_y, transformed_x, transformed_y = load_positions(scene_dir + 'positions.npy')
        original_x, original_y = original_x // (img_res // 64), original_y // (img_res // 64)
        transformed_x, transformed_y = transformed_x // (img_res // 64), transformed_y // (img_res // 64)

        bg_original_x, bg_original_y, bg_transformed_x, bg_transformed_y = original_x, original_y, transformed_x, transformed_y

        # Create sets for original and transformed pixels
        original_pixels = set(zip(bg_original_x, bg_original_y))
        transformed_pixels = set(zip(bg_transformed_x, bg_transformed_y))

        # Create a set of all pixels in a 64x64 image
        all_pixels = {(x, y) for x in range(64) for y in range(64)}

        # Find pixels not in either of the original or transformed sets
        remaining_pixels = all_pixels - (original_pixels | transformed_pixels)

        # Extract remaining_x and remaining_y
        remaining_x = np.array([x for x, y in remaining_pixels])
        remaining_y = np.array([y for x, y in remaining_pixels])

        remaining_pixels_orig = all_pixels - (original_pixels)

        remaining_x_orig = np.array([x for x, y in remaining_pixels_orig])
        remaining_y_orig = np.array([y for x, y in remaining_pixels_orig])

        remaining_pixels_trans = all_pixels - (transformed_pixels)

        remaining_x_trans = np.array([x for x, y in remaining_pixels_trans])
        remaining_y_trans = np.array([y for x, y in remaining_pixels_trans])

        processed_correspondences = {
            'original_x': original_x,
            'original_y': original_y,
            'transformed_x': transformed_x,
            'transformed_y': transformed_y,
            'remaining_x': remaining_x,
            'remaining_y': remaining_y,
            'remaining_x_orig': remaining_x_orig,
            'remaining_y_orig': remaining_y_orig,
            'remaining_x_trans': remaining_x_trans,
            'remaining_y_trans': remaining_y_trans,
        }

        return processed_correspondences
    
    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

def retrieve_attention_maps(attn_down, attn_mid, attn_up, obj_idx, object_positions, img_dims):
    
    attn_maps = [] 
    
    for i in range(len(attn_down)):
        attn_map = 0

        for attn_map_integrated in attn_down[i]:
            attn_map += attn_map_integrated
    
        attn_map /= len(attn_down[i])
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        
        ca_map_obj = 0
        for object_position in object_positions[obj_idx]:
            ca_map_obj += attn_map[:,:,object_position].reshape(b,H,W)

        ca_map_obj = ca_map_obj.mean(axis = 0)
        ca_map_obj = normalize_attn_torch(ca_map_obj)
        ca_map_obj = ca_map_obj.view(1, 1, H, W)
        #m = torch.nn.Upsample(scale_factor=img_dims / H, mode='nearest')
        #ca_map_obj = m(ca_map_obj)
        ca_map_obj = torch.nn.functional.interpolate(ca_map_obj, (img_dims, img_dims), mode = 'bilinear')
        attn_maps.append(ca_map_obj[0][0])

    attn_map = 0

    for attn_map_integrated in attn_mid:
        attn_map += attn_map_integrated
    
    attn_map /= len(attn_mid)
    b, i, j = attn_map.shape
    H = W = int(math.sqrt(i))

    ca_map_obj = 0
    
    for object_position in object_positions[obj_idx]:
        ca_map_obj += attn_map[:,:,object_position].reshape(b,H,W)

    ca_map_obj = ca_map_obj.mean(axis = 0)
    ca_map_obj = normalize_attn_torch(ca_map_obj)
    ca_map_obj = ca_map_obj.view(1, 1, H, W)
    ca_map_obj = torch.nn.functional.interpolate(ca_map_obj, (img_dims, img_dims), mode = 'bilinear')    
    #m = torch.nn.Upsample(scale_factor=img_dims / H, mode='nearest')
    #ca_map_obj = m(ca_map_obj)

    attn_maps.append(ca_map_obj[0][0])
    
    for i in range(len(attn_up)):
        attn_map = 0

        for attn_map_integrated in attn_up[i]:
            attn_map += attn_map_integrated
    
        attn_map /= len(attn_up[i])
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))

        ca_map_obj = 0
        for object_position in object_positions[obj_idx]:
            ca_map_obj += attn_map[:,:,object_position].reshape(b,H,W)

        ca_map_obj = ca_map_obj.mean(axis = 0)
        ca_map_obj = normalize_attn_torch(ca_map_obj)
        ca_map_obj = ca_map_obj.view(1, 1, H, W)
        ca_map_obj = torch.nn.functional.interpolate(ca_map_obj, (img_dims, img_dims), mode = 'bilinear')        
        #m = torch.nn.Upsample(scale_factor=img_dims / H, mode='nearest')
        #ca_map_obj = m(ca_map_obj)
        attn_maps.append(ca_map_obj[0][0])
        
    return attn_maps

def phrase_to_index(prompt, phrases):
    phrases = [x.strip() for x in phrases]
    prompt_list = prompt.strip('.').split(' ')
    object_positions = []
    for obj in phrases:
        obj_position = []
        for word in obj.split(' '):
            obj_first_index = prompt_list.index(word) + 1
            obj_position.append(obj_first_index)
        object_positions.append(obj_position)

    return object_positions

