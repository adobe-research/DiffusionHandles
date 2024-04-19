from typing import Union

import numpy as np
import torch
from tqdm import tqdm

from diffhandles.null_inverter import NullInverter
from diffhandles.guided_stable_diffuser import GuidedStableDiffuser

class StableNullInverter(NullInverter):

    def __init__(self, model: GuidedStableDiffuser, num_ddim_steps: int = 50, guidance_scale: float = 7.5):

        super().__init__(model=model)
        
        self.num_ddim_steps = num_ddim_steps
        self.guidance_scale = guidance_scale

        self.model.scheduler.set_timesteps(self.num_ddim_steps)
        
    def to(self, device: torch.device):
        self.model.to(device)
        return self

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
    
    def get_noise_pred_single(self, latents, t, context, depth=None):
        latents = self.model.scheduler.scale_model_input(latents, t)
        
        if self.model.conf.use_depth:
            latents = torch.cat([latents, depth[0].view(1, 1, depth.shape[2], depth.shape[3])], dim=1)
        
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        
        return noise_pred

    def get_noise_pred(self, latents, t, context, depth=None, is_forward=True):
        latents_input = torch.cat([latents] * 2)
        latents_input = self.model.scheduler.scale_model_input(latents_input, t)
        
        if self.model.conf.use_depth:
            latents_input = torch.cat([latents_input, torch.cat([depth]*2, dim=0)], dim=1)

        guidance_scale = 1 if is_forward else self.guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents):
        # undo scaling by magic number
        latents = 1 / 0.18215 * latents.detach()
        
        # decode with VAE
        image = self.model.vae.decode(latents)['sample']
        
        # value range from [-1, 1] to [0, 1]
        image = (image + 1) / 2

        # if return_type == 'np':
        #     image = (image / 2 + 0.5).clamp(0, 1)
        #     image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        #     image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            # TODO: only accept torch tensors (conversion has to be done beforehand)
            # if type(image) is Image:
            #     image = np.array(image)
            # if type(image) is torch.Tensor and image.dim() == 4:
            #     latents = image
            # else:
                # image = torch.from_numpy(image).float() / 127.5 - 1
                # image = image.permute(2, 0, 1).unsqueeze(0).to(device)
            
            # value range from [0, 1] to [-1, 1]
            image = image * 2 - 1

            # encode with VAE
            latents = self.model.vae.encode(image)['latent_dist'].mean
            
            # scale by magic number
            latents = latents * 0.18215

        return latents

    @torch.no_grad()
    def ddim_loop(self, latent, context, depth):
        uncond_embeddings, cond_embeddings = context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latents=latent, t=t, context=cond_embeddings, depth=depth)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image, context, depth):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent=latent, context=context, depth=depth)
        return image_rec, ddim_latents

    def null_optimization(self, latents, context, depth, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * self.num_ddim_steps)
        for i in range(self.num_ddim_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = torch.optim.Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latents=latent_cur, t=t, context=cond_embeddings, depth=depth)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latents=latent_cur, t=t, context=uncond_embeddings, depth=depth)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = torch.nn.functional.mse_loss(latents_prev_rec, latent_prev)
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
                latent_cur = self.get_noise_pred(latents=latent_cur, t=t, context=context, depth=depth, is_forward=False)
        bar.close()
        return torch.stack(uncond_embeddings_list, dim=0)
    
    def invert(self, target_img: torch.Tensor, depth: torch.Tensor, prompt: str, num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        
        depth = self.model.init_depth(depth)
        context = self.model.init_prompt(prompt)

        if verbose:
            print("DDIM inversion...")
        recon_img, ddim_latents = self.ddim_inversion(image=target_img, context=context, depth=depth)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(
            latents=ddim_latents, context=context, depth=depth, num_inner_steps=num_inner_steps, epsilon=early_stop_epsilon)
        return (target_img, recon_img), ddim_latents[-1], uncond_embeddings
