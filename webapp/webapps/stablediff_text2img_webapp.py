import argparse
import pathlib

import torch
import numpy as np
from omegaconf import OmegaConf
import numpy.typing as npt

from diffhandles.guided_stable_diffuser import GuidedStableDiffuser
from text2img_webapp import Text2imgWebapp

from diffusers import DiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler

from utils import crop_and_resize

class StablediffText2imgWebapp(Text2imgWebapp):
    def __init__(self, netpath: str, port: int, config_path: str = None, device: str = 'cuda:0'):
        super().__init__(netpath=netpath, port=port)

        if config_path is None:
            config_path = f'{pathlib.Path(__file__).parent.resolve()}/../../diffhandles/config/default.yaml'

        conf = OmegaConf.load(config_path)
        conf.guided_diffuser.use_depth = False

        # TODO: this does not work yet, examine why non-depth-conditioned diffuser does not work
        # -> seems to be related to the scheduler or scheduler settings, which do not work for stable-diffusion-2-1,
        # but do seem to work for stable-diffusion-2-depth
        # self.diffuser = GuidedStableDiffuser(conf=conf.guided_diffuser)
        # self.diffuser.to(device)

        self.diffuser = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
        # self.diffuser.scheduler = DDIMScheduler(
        #     beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        # self.diffuser.scheduler = DPMSolverMultistepScheduler.from_config(self.diffuser.scheduler.config)
        # self.diffuser.scheduler = DDIMScheduler(self.diffuser.scheduler.config)
        self.diffuser = self.diffuser.to(device)

    def generate_image(self, prompt: str = None) -> npt.NDArray:

        print('run_text2img')

        if prompt is None:
            raise ValueError('Some inputs are missing.')

        print(prompt)

        # TODO: this does not work yet, examine why non-depth-conditioned diffuser does not work
        # with torch.no_grad():
        #     activations, activations2, activations3, latent_image, inverted_null_text, inverted_noise = self.diffuser.initial_inference(
        #         init_latents=None, depth=None, uncond_embeddings=None, prompt=prompt)
        #     generated_image = self.diffuser.decode_latent_image(latent_image)

        # # generated_image = crop_and_resize(img=generated_image, size=self.img_res)
        # generated_image = (generated_image * 255.0)[0].permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()

        generated_image = self.diffuser(prompt, num_inference_steps=50).images[0]
        generated_image = np.array(generated_image)

        return generated_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--netpath', type=str, default='/text2img')
    parser.add_argument('--port', type=int, default=6011)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--config_path', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    server = StablediffText2imgWebapp(
        netpath=args.netpath, port=args.port, config_path=args.config_path, device=args.device)
    server.start()
