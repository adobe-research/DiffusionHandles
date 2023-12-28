import torch
import requests
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline
import torch
import torch.nn.functional as F
import numpy as np
import math
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers import StableDiffusionDepth2ImgPipeline
from diffusers.image_processor import VaeImageProcessor
from my_model import unet_2d_condition
import json
from PIL import Image, ImageOps
from batch_edit_gen_utils import Pharse2idx, retrieve_attention_maps, compute_localized_transformed_appearance_loss, preprocess, prepare_extra_step_kwargs, prepare_latents, get_timesteps, compute_background_loss, load_data
import os
from tqdm import tqdm
import requests
from torch.optim.adam import Adam
import torch.nn.functional as nnf
from typing import Optional, Union, Tuple, List, Callable, Dict
from diffusers import DDIMScheduler
import old_code.ptp_utils as ptp_utils
import pickle

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image




@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    depth_mask = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image'
):
    batch_size = len(prompt)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, latents, depth_mask, context, t, guidance_scale, low_resource=False)
        
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent



def run_and_display(prompts, latent=None, depth_mask = None, run_baseline=False, generator=None, uncond_embeddings=None, verbose=True):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = text2image_ldm_stable(ldm_stable, prompts, latent=latent, depth_mask = depth_mask, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings)
    if verbose:
        ptp_utils.view_images(images)
    return images, x_t

class NullInversion:
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        latents = self.model.scheduler.scale_model_input(latents, t)
        latents = torch.cat([latents, self.depth_mask[0].view(1, 1, self.depth_mask.shape[2], self.depth_mask.shape[3])], dim=1)
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        latents_input = self.model.scheduler.scale_model_input(latents_input, t)
        latents_input = torch.cat([latents_input, self.depth_mask], dim=1)

        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list
    
    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        #ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
        
    
    def __init__(self, model, depth_mask):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None
        self.depth_mask = depth_mask


def inverted_inference(device, name, unet, vae, tokenizer, text_encoder, scheduler, latents, depth_mask, uncond_embeddings, prompt, phrases):
    # Get Object Positions
    object_positions = Pharse2idx(prompt, phrases)

    # Encode Classifier Embeddings
    uncond_input = tokenizer(
        [""] * 1, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    
    #uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    # Encode Prompt
    input_ids = tokenizer(
            [prompt] * 1,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

    cond_embeddings = text_encoder(input_ids.input_ids.to(device))[0]
    #text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
    #text_embeddings = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), cond_embeddings])
      
    generator = torch.manual_seed(2773)  # Seed generator to create the inital latent noise - 305 for car, 105 for cup, 155 for lamp
    

    strength = 1.0
    
    #Preprocess image
    #image = preprocess(image)

    #Set timesteps
    scheduler.set_timesteps(50, device=device)
    timesteps, num_inference_steps = get_timesteps(scheduler, 50, strength, device)

    extra_step_kwargs = prepare_extra_step_kwargs(generator, 0.0, scheduler)

    attention_list = []
    activation_list = [] 
    activation2_list = []
    activation3_list = []
    
    obj_number = 2
    depth_iter = True
    timestep_num = 0
    for index, t in enumerate(tqdm(timesteps)):
        if(timestep_num > 51):
            depth_iter = False
        with torch.no_grad():
            #print(index)
            #latent_timestep = timesteps[index:index+1]
            #Prepare latent variables
            latent_model_input = latents #torch.cat([latents]) #if do_classifier_free_guidance else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            if(depth_iter):
                latent_model_input = torch.cat([latent_model_input, depth_mask[0].view(1, 1, depth_mask.shape[2], depth_mask.shape[3])], dim=1)

            # predict the noise residual
            unet_output = unet(
                latent_model_input,
                t,
                encoder_hidden_states=cond_embeddings,
                cross_attention_kwargs=None,
                return_dict=False,
            )
            
            noise_pred = unet_output[0]
            attn_map_integrated_up = unet_output[1]
            attn_map_integrated_mid = unet_output[2]
            attn_map_integrated_down = unet_output[3]
            activations = unet_output[4]
            activations2 = unet_output[5]
            activations3 = unet_output[6]
            
            activation_list.append(activations[0])
            activation2_list.append(activations2[0])
            activation3_list.append(activations3[0])
            attention_obj_list = []
                        
            for m in range(obj_number):
                attentions = retrieve_attention_maps(attn_map_integrated_down, attn_map_integrated_mid, attn_map_integrated_up, m, object_positions, latents.shape[2])
                attention_obj_list.append(attentions)
            attention_list.append(attention_obj_list)

            latent_model_input = torch.cat([latents]*2) #if do_classifier_free_guidance else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            if(depth_iter):
                latent_model_input = torch.cat([latent_model_input, depth_mask], dim=1)


            text_embeddings = torch.cat([uncond_embeddings[index].expand(*cond_embeddings.shape), cond_embeddings])

            # predict the noise residual
            unet_output = unet(
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
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]     
            torch.cuda.empty_cache()
            timestep_num += 1
    
    with torch.no_grad():
        image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        image = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor).postprocess(image, output_type="pil")
        image[0].save('output' + name + '.png')
    return attention_list, activation_list, activation2_list, activation3_list



def simple_inference(device, name, unet, vae, tokenizer, text_encoder, scheduler, image, depth_mask, prompt, phrases):
    # Get Object Positions
    object_positions = Pharse2idx(prompt, phrases)

    # Encode Classifier Embeddings
    uncond_input = tokenizer(
        [""] * 1, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    # Encode Prompt
    input_ids = tokenizer(
            [prompt] * 1,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

    cond_embeddings = text_encoder(input_ids.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
      
    generator = torch.manual_seed(2773)  # Seed generator to create the inital latent noise - 305 for car, 105 for cup, 155 for lamp
    

    strength = 1.0
    
    #Preprocess image
    image = preprocess(image)

    #Set timesteps
    scheduler.set_timesteps(50, device=device)
    timesteps, num_inference_steps = get_timesteps(scheduler, 50, strength, device)
    latent_timestep = timesteps[:1].repeat(1)

    #Prepare latent variables
    latents = prepare_latents(scheduler, vae, 
        image, latent_timestep, 1, 1, uncond_embeddings.dtype, device, generator
    )

    extra_step_kwargs = prepare_extra_step_kwargs(generator, 0.0, scheduler)

    attention_list = []
    activation_list = [] 
    activation2_list = []
    activation3_list = []
    
    obj_number = 2
    depth_iter = True
    timestep_num = 0
    for index, t in enumerate(tqdm(timesteps)):
        if(timestep_num > 51):
            depth_iter = False
        with torch.no_grad():

            latent_model_input = latents #torch.cat([latents]) #if do_classifier_free_guidance else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            if(depth_iter):
                latent_model_input = torch.cat([latent_model_input, depth_mask[0].view(1, 1, depth_mask.shape[2], depth_mask.shape[3])], dim=1)

            # predict the noise residual
            unet_output = unet(
                latent_model_input,
                t,
                encoder_hidden_states=cond_embeddings,
                cross_attention_kwargs=None,
                return_dict=False,
            )
            
            noise_pred = unet_output[0]
            attn_map_integrated_up = unet_output[1]
            attn_map_integrated_mid = unet_output[2]
            attn_map_integrated_down = unet_output[3]
            activations = unet_output[4]
            activations2 = unet_output[5]
            activations3 = unet_output[6]
            
            activation_list.append(activations[0])
            activation2_list.append(activations2[0])
            activation3_list.append(activations3[0])
            attention_obj_list = []
                        
            for m in range(obj_number):
                attentions = retrieve_attention_maps(attn_map_integrated_down, attn_map_integrated_mid, attn_map_integrated_up, m, object_positions, latents.shape[2])
                attention_obj_list.append(attentions)
            attention_list.append(attention_obj_list)

            latent_model_input = torch.cat([latents]*2) #if do_classifier_free_guidance else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            if(depth_iter):
                latent_model_input = torch.cat([latent_model_input, depth_mask], dim=1)

            # predict the noise residual
            unet_output = unet(
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
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]     
            torch.cuda.empty_cache()
            timestep_num += 1
    
    with torch.no_grad():
        image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        image = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor).postprocess(image, output_type="pil")
        image[0].save('output' + name + '.png')
    return attention_list, activation_list, activation2_list, activation3_list

@torch.no_grad()
def depth_estimate_inference(device, unet, vae, tokenizer, text_encoder, scheduler, latents, depth_mask, uncond_embeddings, prompt, phrases, attention_maps_orig, activations_orig, activations2_orig, activations3_orig):

    # Get Object Positions
    object_positions = Pharse2idx(prompt, phrases)

    negative_prompt = "bad, deformed, ugly, bad anotomy"
    


    # Encode Classifier Embeddings
    uncond_input = tokenizer(
        [""] * 1, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    #uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    # Encode Prompt
    input_ids = tokenizer(
            [prompt] * 1,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

    cond_embeddings = text_encoder(input_ids.input_ids.to(device))[0]
    generator = torch.manual_seed(2773)  
    strength = 1.0

    #Set timesteps
    scheduler.set_timesteps(50, device=device)
    timesteps, num_inference_steps = get_timesteps(scheduler, 50, strength, device)

    extra_step_kwargs = prepare_extra_step_kwargs(generator, 0.0, scheduler)

    loss = torch.tensor(10000)
    timestep_num = 0
    
    obj_number = 2
    
    depth_iter = True
    
    for index, t in enumerate(tqdm(timesteps)):
        iteration = 0
        attention_map_orig = attention_maps_orig[timestep_num]
        activation_orig = activations_orig[timestep_num]
        activation2_orig = activations2_orig[timestep_num]
        activation3_orig = activations3_orig[timestep_num]
        torch.set_grad_enabled(True)
                    
        while iteration < 3 and (index < 38 and index >= 0):
            
            latent_model_input = latents
            latents = latents.requires_grad_(True)

            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = torch.cat([latent_model_input, depth_mask[0].view(1,1,depth_mask.shape[2], depth_mask.shape[3])], dim=1)
                            
            # predict the noise residual
            unet_output = unet(
                latent_model_input,
                t,
                encoder_hidden_states=cond_embeddings,
                cross_attention_kwargs=None,
                return_dict=False,
            )
                
            noise_pred = unet_output[0]
            attn_map_integrated_up = unet_output[1]
            attn_map_integrated_mid = unet_output[2]
            attn_map_integrated_down = unet_output[3]
            activations = unet_output[4]
            activations2 = unet_output[5]
            activations3 = unet_output[6]

            loss = 0
            
            for m in range(obj_number):
                attention_maps = retrieve_attention_maps(attn_map_integrated_down, attn_map_integrated_mid, attn_map_integrated_up, m, object_positions, latents.shape[2])
                attention_map_obj_orig = attention_map_orig[m]                
                if(m == 0):
                    if(timestep_num >= 0):
                        appearance_loss = 0.0
                        bg_loss = 0.0
                        if(timestep_num % 3 == 0):
                            appearance_loss += 7.5*compute_localized_transformed_appearance_loss(attention_maps, activations3[0], attention_map_obj_orig, activation3_orig, 6, 7, 1)
                            bg_loss += 1.5*compute_background_loss(attention_maps, activations3[0], attention_map_obj_orig, activation3_orig, 6, 7)

                        if(timestep_num % 3 == 1):
                            appearance_loss += 5.0*compute_localized_transformed_appearance_loss(attention_maps, activations2[0], attention_map_obj_orig, activation2_orig, 5 , 6, 1)
                            bg_loss += 1.5*compute_background_loss(attention_maps, activations2[0], attention_map_obj_orig, activation2_orig, 5, 6)

                        if(timestep_num % 3 == 2):
                            appearance_loss += 5.0*compute_localized_transformed_appearance_loss(attention_maps, activations2[0], attention_map_obj_orig, activation2_orig, 5 , 6, 1)                             
                            appearance_loss += 7.5*compute_localized_transformed_appearance_loss(attention_maps, activations3[0], attention_map_obj_orig, activation3_orig, 6, 7, 1)
                            bg_loss += 1.5*compute_background_loss(attention_maps, activations3[0], attention_map_obj_orig, activation3_orig, 6, 7)
                            bg_loss += 1.5*compute_background_loss(attention_maps, activations2[0], attention_map_obj_orig, activation2_orig, 5, 6)
                    else:
                        appearance_loss = 0.1*compute_localized_transformed_appearance_loss(attention_maps, activations3[0], attention_map_obj_orig, activation3_orig, 6, 7, 1)

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

                    print('bg loss')
                    print(bg_loss)

                    print('appearance loss')
                    print(appearance_loss)
                                        
                                        
                    loss += 1.5*app_wt*appearance_loss + 1.25*bg_wt*bg_loss 

            loss *= 30        

            if(loss == 0):
                grad_cond = 0
            else:
                grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]
                        
            var = 0.1 
            latents = latents - 1.0 * grad_cond * var
            iteration += 1
            torch.cuda.empty_cache() 
            
        prev_t = t

        torch.set_grad_enabled(False)            
        with torch.no_grad():

            latent_model_input = torch.cat([latents] * 2) #if do_classifier_free_guidance else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            if(depth_iter):
                latent_model_input = torch.cat([latent_model_input, depth_mask], dim=1)

            text_embeddings = torch.cat([uncond_embeddings[index].expand(*cond_embeddings.shape), cond_embeddings])

            # predict the noise residual
            unet_output = unet(
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
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            torch.cuda.empty_cache()
        timestep_num += 1
                
    with torch.no_grad():
        image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        image = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor).postprocess(image, output_type="pil")
        image[0].save('output_depth_estimator.png')
        return image[0]
    return

def normalize_depth_map(depth_map, device):
    width, height = depth_map.size
    depth_map = np.asarray(depth_map)
    depth_map = torch.from_numpy(np.array(depth_map))
    depth_map = depth_map.to(device, torch.float32)
    depth_map = depth_map.view(1, depth_map.shape[0], depth_map.shape[1])
    #print(depth_map.shape)
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(64, 64),
        mode="bicubic",
        align_corners=False,
    )

    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = 2.0 *( (depth_map - depth_min) / (depth_max - depth_min) ) - 1.0
    depth_map = depth_map.to(torch.float32)
    depth_map = torch.cat([depth_map] * 2) 
    return depth_map    


def compile_images(image_sets):
    # Determine the width and height of the first image in the first set to get the size
    # Assuming all images are the same size for simplicity
    image_width, image_height = image_sets[0][0].size
    
    # Calculate the maximum number of images in any set to determine the canvas width
    max_images_in_set = max(len(image_set) for image_set in image_sets)
    canvas_width = max_images_in_set * image_width
    # The canvas height is simply the height of one image times the number of sets
    canvas_height = image_height * len(image_sets)
    
    # Create a new image with the calculated canvas size
    compiled_image = Image.new('RGB', (canvas_width, canvas_height))

    # Paste each image set onto the canvas
    y_offset = 0
    for image_set in image_sets:
        x_offset = 0
        for img in image_set:
            compiled_image.paste(img, (x_offset, y_offset))
            x_offset += image_width
        y_offset += image_height

    return compiled_image

NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ''
LOW_RESOURCE = False 
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

img_sets = [] 


def run_batch_from_file(json_file):

    with open(json_file, 'r') as file:
        scene_data = json.load(file)

    

    for scene in scene_data['scene_data']:
        curr_set = []
        curr_set.append(Image.open(scene['scene_dir'] + 'curr_image.png'))
        for i in range(len(scene["transform_dirs"])):
            transformed_img = main(scene["scene_dir"], scene["transform_dirs"][i], scene["init_transform_dir"], scene["examples"])
            curr_set.append(transformed_img)
        img_sets.append(curr_set)
    
    compiled_results = compile_images(img_sets)
    compiled_results.save('compiled_results.png')
    

    
def main(scene_dir, transform_dir, init_transform_dir, examples):

    load_data(scene_dir, transform_dir)

    first_inversion = True
    find_internals = True


    if(os.path.exists(scene_dir + 'inversion_data.pt')):
        first_inversion = False

    if(os.path.exists(scene_dir + 'internal_reps.pkl')):
        find_internals = False


    if(first_inversion):

        ldm_stable = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth", use_auth_token=MY_TOKEN, scheduler=scheduler).to(device)
        try:
            ldm_stable.disable_xformers_memory_efficient_attention()
        except AttributeError:
            print("Attribute disable_xformers_memory_efficient_attention() is missing")
        tokenizer = ldm_stable.tokenizer

        depth_mask = Image.open(scene_dir + init_transform_dir + 'transformed_depth_map.png')
        depth_mask = ImageOps.grayscale(depth_mask)
        depth_mask = normalize_depth_map(depth_mask, device)

        null_inversion = NullInversion(ldm_stable, depth_mask)

        image_path = scene_dir + 'curr_image.png'
        (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, examples['prompt'], num_inner_steps=5 ,verbose=True)

        torch.save({'x_t': x_t, 'uncond_embeddings': uncond_embeddings}, scene_dir + 'inversion_data.pt')

        del ldm_stable

    else:

        inversion_data = torch.load(scene_dir + 'inversion_data.pt')

        x_t = inversion_data['x_t']
        uncond_embeddings = inversion_data['uncond_embeddings']

    unet = unet_2d_condition.UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-depth", subfolder="unet")    

    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-depth", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-depth", subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-depth", subfolder="vae")
    unet.to(device, torch.float32)
    text_encoder.to(device, torch.float32)
    vae.to(device, torch.float32)

    depth_mask = Image.open(scene_dir + init_transform_dir + 'transformed_depth_map.png')
    depth_mask = ImageOps.grayscale(depth_mask)
    depth_mask = normalize_depth_map(depth_mask, device)

    if(find_internals):
        attentions, activations, activations2, activations3 = inverted_inference(device, 'orig', unet, vae, tokenizer, text_encoder, scheduler, x_t, depth_mask, uncond_embeddings, examples['prompt'], examples['phrases'])
        with open(scene_dir + 'internal_reps.pkl', 'wb') as file:
            pickle.dump([attentions, activations, activations2, activations3], file)
    else:
        with open(scene_dir + 'internal_reps.pkl', 'rb') as file:
            loaded_lists = pickle.load(file)
        attentions, activations, activations2, activations3 = loaded_lists

    depth_mask = Image.open(scene_dir + transform_dir + 'transformed_depth_map.png')
    depth_mask = ImageOps.grayscale(depth_mask)
    depth_mask = normalize_depth_map(depth_mask, device)

    output_img = depth_estimate_inference(device, unet, vae, tokenizer, text_encoder, scheduler, x_t, depth_mask, uncond_embeddings, examples['prompt'], examples['phrases'], attentions, activations, activations2, activations3)
    output_img.save(scene_dir + transform_dir + 'output.png')
    return output_img


if __name__ == "__main__":
    json_file = 'id_paths.json' # Path to your JSON file containing path triplets
    run_batch_from_file(json_file)

