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
import scipy.ndimage

from diffhandles.model.unet_2d_condition import UNet2DConditionModel # this is the custom UNet that can also return intermediate activations and attentions
from diffhandles.guided_diffuser import GuidedDiffuser
from diffhandles.utils import normalize_attn_torch, unpack_correspondences
from diffhandles.losses import compute_foreground_loss, compute_background_loss

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

        strength = 1.0
        
        generator = torch.manual_seed(self.conf.seed)  # Seed generator to create the inital latent noise - 305 for car, 105 for cup, 155 for lamp
        
        #Set timesteps
        self.scheduler.set_timesteps(self.conf.num_timesteps, device=self.device)
        timesteps, num_inference_steps = self.get_timesteps(self.conf.num_timesteps, strength)

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

        if fg_weight is None:
            fg_weight = self.conf.fg_weight
        if bg_weight is None:
            bg_weight = self.conf.bg_weight

        with torch.no_grad():
        
            generator = torch.manual_seed(self.conf.seed)

            #Set timesteps
            self.scheduler.set_timesteps(self.conf.num_timesteps, device=self.device)
            timesteps, num_inference_steps = self.get_timesteps(self.conf.num_timesteps, strength)
            
            processed_correspondences = self.process_correspondences(correspondences, img_res=depth.shape[-1], bg_erosion=self.conf.bg_erosion)
            
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

            if save_denoising_steps:
                denoising_steps = {
                    'opt': [],
                    'post-opt': [],
                }
            
            # create guidance weight schedule
            fg_weight *= 30
            bg_weight *= 30
            denoising_weight_schedule = []
            if self.conf.guidance_schedule_type == "constant":
                fg_weight_falloff = np.linspace(fg_weight, fg_weight, self.conf.guidance_max_step)
                bg_weight_falloff = np.linspace(bg_weight, bg_weight, self.conf.guidance_max_step)
            elif self.conf.guidance_schedule_type == "linear":
                fg_weight_falloff = np.linspace(fg_weight, 0.0, self.conf.guidance_max_step)
                bg_weight_falloff = np.linspace(bg_weight, 0.0, self.conf.guidance_max_step)
            elif self.conf.guidance_schedule_type == "quadratic":
                fg_weight_falloff = np.linspace(np.sqrt(fg_weight), 0.0, self.conf.guidance_max_step)**2
                bg_weight_falloff = np.linspace(np.sqrt(bg_weight), 0.0, self.conf.guidance_max_step)**2
            else:
                raise ValueError(f"Unknown guidance schedule type: {self.conf.guidance_schedule_type}")
            for t_idx in range(self.conf.guidance_max_step):
                if t_idx % 3 == 0:
                    fg_weights = [0.0, 0.0, 7.5]
                    bg_weights = [0.0, 0.0, 1.5]
                elif t_idx % 3 == 1:
                    fg_weights = [0.0, 5.0, 0.0]
                    bg_weights = [0.0, 1.5, 0.0]
                elif t_idx % 3 == 2:
                    fg_weights = [0.0, 5.0, 7.5]
                    bg_weights = [0.0, 1.5, 1.5]
                denoising_weight_schedule.append((
                    t_idx,
                    (np.array(fg_weights)*fg_weight_falloff[t_idx]).tolist(),
                    (np.array(bg_weights)*bg_weight_falloff[t_idx]).tolist()))
            denoising_weight_schedule.append((self.conf.guidance_max_step, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
            optimization_weight_schedule = [
                (0, [2.5, 2.5, 2.5], [1.25, 1.25, 1.25]),
                (1, [1.25, 1.25, 1.25], [2.5, 2.5, 2.5]),
                (2, [1.25, 1.25, 1.25], [1.25, 1.25, 1.25]),
                (3, [2.5, 2.5, 2.5], [2.5, 2.5, 2.5]),
            ]
            guidance_weight_schedule = StepGuidanceWeightSchedule(
                denoising_steps=denoising_weight_schedule,
                optimization_steps=optimization_weight_schedule)
        
        # latents = latents.requires_grad_(True)

        for t_idx, t in enumerate(tqdm(timesteps)):
            
            torch.set_grad_enabled(True)
            # with torch.enable_grad():
            latents = latents.requires_grad_(True)

            activations_size = (activations_orig[2][t_idx].shape[-2], activations_orig[2][t_idx].shape[-1])

            if save_denoising_steps:
                denoising_steps['opt'].append([])
                        
            # opt = torch.optim.Adam(params=[latents], lr=0.01)
            # opt = torch.optim.SGD(params=[latents], lr=0.1)

            iteration = 0
            while iteration < self.conf.num_optsteps and t_idx < self.conf.guidance_max_step:

                # latents = latents.detach().requires_grad_(True)
                
                # latents = latents.requires_grad_(True)
                latent_model_input = latents

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
                activations = [unet_output[4], unet_output[5], unet_output[6]]

                fgw, bgw = guidance_weight_schedule(t_idx, iteration)

                loss = 0.0
                for act_idx in range(len(activations_orig)):
                    if fgw != 0.0:
                        loss += fgw[act_idx] * compute_foreground_loss(
                            activations=activations[act_idx][0], activations_orig=activations_orig[act_idx][t_idx],
                            processed_correspondences=processed_correspondences,
                            patch_size=self.conf.fg_patch_size, activations_size=activations_size)
                    if bgw != 0.0:
                        loss += bgw[act_idx] * compute_background_loss(
                            activations=activations[act_idx][0], activations_orig=activations_orig[act_idx][t_idx],
                            processed_correspondences=processed_correspondences,
                            patch_size=self.conf.bg_patch_size, activations_size=activations_size, loss_type=self.conf.bg_loss_type)

                if(loss == 0):
                    grad_cond = 0
                else:
                    grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]
                latents = latents - grad_cond * 0.1

                # if loss != 0:
                #     opt.zero_grad()
                #     loss.backward()
                #     opt.step()
                
                iteration += 1
                # torch.cuda.empty_cache()

            if save_denoising_steps:
                with torch.no_grad():
                    image = self.decode_latent_image(latents.detach())
                    denoising_steps['opt'][-1].append(image.cpu())
                
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
                    image = self.decode_latent_image(latents.detach())
                    denoising_steps['opt'][-1].append(image.cpu())
                    
        with torch.no_grad():
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image = VaeImageProcessor(vae_scale_factor=self.vae.config.scaling_factor).postprocess(image, output_type="pt")
        
        if save_denoising_steps:
            return image, denoising_steps
        else:
            return image

    def process_correspondences(self, correspondences, img_res, bg_erosion=0):

        original_x, original_y, transformed_x, transformed_y = unpack_correspondences(correspondences)

        # Since np.split creates arrays of shape (N, 1), we'll squeeze them to get back to shape (N,)
        original_x = original_x.squeeze()
        original_y = original_y.squeeze()
        transformed_x = transformed_x.squeeze()
        transformed_y = transformed_y.squeeze()
        
        # bg_mask_orig = np.zeros((img_res, img_res))
        
        # bg_mask_trans = np.zeros((img_res, img_res))
        
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
        
        # for x, y in zip(visible_orig_x, visible_orig_y):
        #     bg_mask_orig[y,x] = 1

        # for x, y in zip(visible_trans_x, visible_trans_y):
        #     bg_mask_trans[y,x] = 1        
        
        original_x, original_y, transformed_x, transformed_y = (
            np.array(visible_orig_x, dtype=np.int64), np.array(visible_orig_y, dtype=np.int64),
            np.array(visible_trans_x, dtype=np.int64), np.array(visible_trans_y, dtype=np.int64))

        original_x, original_y = original_x // (img_res // 64), original_y // (img_res // 64)
        transformed_x, transformed_y = transformed_x // (img_res // 64), transformed_y // (img_res // 64)

        bg_mask_orig = np.ones(shape=[64, 64], dtype=np.bool_)
        if len(original_x) > 0:
            bg_mask_orig[original_y, original_x] = False

        bg_mask_trans = np.ones(shape=[64, 64], dtype=np.bool_)
        if len(transformed_x) > 0:
            bg_mask_trans[transformed_y, transformed_x] = False

        if bg_erosion > 0:
            bg_mask_orig = scipy.ndimage.binary_erosion(bg_mask_orig, iterations=bg_erosion)
            bg_mask_trans = scipy.ndimage.binary_erosion(bg_mask_trans, iterations=bg_erosion)

        bg_y, bg_x = np.nonzero(bg_mask_orig & bg_mask_trans)
        bg_y_orig, bg_x_orig = np.nonzero(bg_mask_orig)
        bg_y_trans, bg_x_trans = np.nonzero(bg_mask_trans)

        
        
        # # Create sets for original and transformed pixels
        # original_pixels = set(zip(original_x, original_y))
        # transformed_pixels = set(zip(transformed_x, transformed_y))

        # # Create a set of all pixels in a 64x64 image
        # all_pixels = {(x, y) for x in range(64) for y in range(64)}

        # # Find pixels not in either of the original or transformed sets
        # bg_pixels = all_pixels - (original_pixels | transformed_pixels)

        # # Extract background_x and background_y
        # bg_x = np.array([x for x, y in bg_pixels])
        # bg_y = np.array([y for x, y in bg_pixels])

        # bg_pixels_orig = all_pixels - (original_pixels)

        # bg_x_orig = np.array([x for x, y in bg_pixels_orig])
        # bg_y_orig = np.array([y for x, y in bg_pixels_orig])

        # bg_pixels_trans = all_pixels - (transformed_pixels)

        # bg_x_trans = np.array([x for x, y in bg_pixels_trans])
        # bg_y_trans = np.array([y for x, y in bg_pixels_trans])

        processed_correspondences = {
            'original_x': original_x,
            'original_y': original_y,
            'transformed_x': transformed_x,
            'transformed_y': transformed_y,
            'background_x': bg_x,
            'background_y': bg_y,
            'background_x_orig': bg_x_orig,
            'background_y_orig': bg_y_orig,
            'background_x_trans': bg_x_trans,
            'background_y_trans': bg_y_trans,
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

class GuidanceWeightSchedule:

    def __init__(self):
        pass

    def __call__(self, denoising_step: int, optimization_step: int):
        fg_weights = [1.0]*3
        bg_weights = [1.0]*3
        return fg_weights, bg_weights

class StepGuidanceWeightSchedule(GuidanceWeightSchedule):

    def __init__(
            self,
            denoising_steps: list[(int, list[float], list[float])],
            optimization_steps: list[(int, list[float], list[float])]):

        super().__init__()
        
        if not all(len(fg_weights) == len(bg_weights) for _, fg_weights, bg_weights in denoising_steps):
            raise ValueError("Number of foreground and background weights do not match.")
        if not all(len(fg_weights) == len(bg_weights) for _, fg_weights, bg_weights in optimization_steps):
            raise ValueError("Number of foreground and background weights do not match.")
        if len(denoising_steps[0][1]) != len(optimization_steps[0][1]):
            raise ValueError("Number of denoising and optimization weights do not match.")

        self.denoising_steps = sorted(denoising_steps, key=lambda step: step[0])
        self.optimization_steps = sorted(optimization_steps, key=lambda step: step[0])

    def __call__(self, denoising_step: int, optimization_step: int):
        
        denoising_fg_weights = None
        denoising_bg_weights = None
        optimization_fg_weights = None
        optimization_bg_weights = None
        
        for step, fg_weights, bg_weights in reversed(self.denoising_steps):
            if denoising_step >= step:
                denoising_fg_weights = fg_weights
                denoising_bg_weights = bg_weights
                break
        for step, fg_weights, bg_weights in reversed(self.optimization_steps):
            if optimization_step >= step:
                optimization_fg_weights = fg_weights
                optimization_bg_weights = bg_weights
                break

        if any(weight is None for weight in [denoising_fg_weights, denoising_bg_weights, optimization_fg_weights, optimization_bg_weights]):
            raise ValueError(f"Could not find weights for denoising step {denoising_step} and optimization step {optimization_step}.")

        fg_weights = [dw * ow for dw, ow in zip(denoising_fg_weights, optimization_fg_weights)]
        bg_weights = [dw * ow for dw, ow in zip(denoising_bg_weights, optimization_bg_weights)]
        
        return fg_weights, bg_weights
    