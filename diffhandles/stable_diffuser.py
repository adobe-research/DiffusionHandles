import inspect
from typing import List

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffhandles.my_model.unet_2d_condition import UNet2DConditionModel as CustomUNet2DConditionModel
from tqdm import tqdm

from diffhandles.diffuser import Diffuser
from diffhandles.utils import normalize_depth

class StableDiffuser(Diffuser):
    def __init__(self, custom_unet=False):
        super().__init__()

        self.scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        if custom_unet:
            self.unet = CustomUNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-depth", subfolder="unet")
        else:
            self.unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-depth", subfolder="unet")
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-depth", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-depth", subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-depth", subfolder="vae")

        self.device = self.unet.device

    def to(self, device: torch.device = None, dtype: torch.dtype = None):
        self.unet = self.unet.to(device=device, dtype=dtype)
        self.tokenizer = self.tokenizer.to(device=device, dtype=dtype)
        self.text_encoder = self.text_encoder.to(device=device, dtype=dtype)
        self.vae = self.vae.to(device=device, dtype=dtype)

        self.device = device

        return self
    
    def initial_inference(self, latents: torch.Tensor, depth: torch.Tensor, uncond_embeddings: torch.Tensor, prompt: str, phrases: List[str]):

        depth = normalize_depth(depth)
        
        # Get Object Positions
        object_positions = phrase_to_index(prompt, phrases)

        # Encode Classifier Embeddings
        uncond_input = self.tokenizer(
            [""] * 1, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        )
        
        #uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

        # Encode Prompt
        input_ids = self.tokenizer(
                [prompt] * 1,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )

        cond_embeddings = self.text_encoder(input_ids.input_ids.to(self.device))[0]
        #text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
        #text_embeddings = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), cond_embeddings])
        
        generator = torch.manual_seed(2773)  # Seed generator to create the inital latent noise - 305 for car, 105 for cup, 155 for lamp
        

        strength = 1.0
        
        #Preprocess image
        #image = preprocess(image)

        #Set timesteps
        self.scheduler.set_timesteps(50, device=self.device)
        timesteps, num_inference_steps = self.get_timesteps(50, strength)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.0)

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
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if(depth_iter):
                    latent_model_input = torch.cat([latent_model_input, depth[0].view(1, 1, depth.shape[2], depth.shape[3])], dim=1)

                # predict the noise residual
                unet_output = self.unet(
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
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if(depth_iter):
                    latent_model_input = torch.cat([latent_model_input, depth], dim=1)


                text_embeddings = torch.cat([uncond_embeddings[index].expand(*cond_embeddings.shape), cond_embeddings])

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
                timestep_num += 1
        
        # with torch.no_grad():
        #     image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        #     image = VaeImageProcessor(vae_scale_factor=self.vae.config.scaling_factor).postprocess(image, output_type="pil")
        #     image[0].save('output' + name + '.png')
        return attention_list, activation_list, activation2_list, activation3_list

    def guided_inference(
            self, latents: torch.Tensor, depth: torch.Tensor, uncond_embeddings: torch.Tensor, prompt: str, phrases: List[str],
            attention_maps_orig: torch.Tensor, activations_orig: torch.Tensor, activations2_orig: torch.Tensor, activations3_orig: torch.Tensor):

        depth_mask = normalize_depth(depth_mask)
        
        # Get Object Positions
        object_positions = phrase_to_index(prompt, phrases)

        negative_prompt = "bad, deformed, ugly, bad anotomy"
        
        # Encode Classifier Embeddings
        uncond_input = self.tokenizer(
            [""] * 1, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        )
        #uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

        # Encode Prompt
        input_ids = self.tokenizer(
                [prompt] * 1,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )

        cond_embeddings = self.text_encoder(input_ids.input_ids.to(self.device))[0]
        generator = torch.manual_seed(2773)  
        strength = 1.0

        #Set timesteps
        self.scheduler.set_timesteps(50, device=self.device)
        timesteps, num_inference_steps = self.get_timesteps(50, strength)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.0)

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

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, depth_mask[0].view(1,1,depth_mask.shape[2], depth_mask.shape[3])], dim=1)
                                
                # predict the noise residual
                unet_output = self.unet(
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
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if(depth_iter):
                    latent_model_input = torch.cat([latent_model_input, depth_mask], dim=1)

                text_embeddings = torch.cat([uncond_embeddings[index].expand(*cond_embeddings.shape), cond_embeddings])

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
            timestep_num += 1
                    
        # with torch.no_grad():
        #     image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        #     image = VaeImageProcessor(vae_scale_factor=self.vae.config.scaling_factor).postprocess(image, output_type="pil")
        #     image[0].save('output_depth_estimator.png')
        #     return image[0]
        return

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

def phrase_to_index(prompt, phrases):
    phrases = [x.strip() for x in phrases.split(';')]
    prompt_list = prompt.strip('.').split(' ')
    object_positions = []
    for obj in phrases:
        obj_position = []
        for word in obj.split(' '):
            obj_first_index = prompt_list.index(word) + 1
            obj_position.append(obj_first_index)
        object_positions.append(obj_position)

    return object_positions

