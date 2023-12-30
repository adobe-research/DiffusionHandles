import os
import io
import zipfile
from dataclasses import dataclass
import urllib.request
import yaml

import torch
from omegaconf import OmegaConf

from diffhandles.inpainter import Inpainter
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.refinement import refine_predict

@dataclass
class LamaRefinerConfig:
    gpu_ids = 0, # the GPU ids of the machine to use. If only single GPU, use: "0,"
    modulo = 8
    n_iters = 15 # number of iterations of refinement for each scale
    lr = 0.002 # learning rate
    min_side = 512 # all sides of image on all scales should be >= min_side / sqrt(2)
    max_scales = 3 # max number of downscaling scales for the image-mask pyramid
    px_budget = 1800000 # pixels budget. Any image will be resized to satisfy height*width <= px_budget


class LamaInpainter(Inpainter):
    def __init__(self, refine=False, refiner_config=LamaRefinerConfig()):
        super().__init__()

        self.refine = refine
        self.refiner_config = refiner_config

        # download pre-trained model to local cache
        if not (
            os.path.exists(os.path.expanduser('~/.cache/lama/config.yaml')) and
            os.path.exists(os.path.expanduser('~/.cache/lama/models/best.ckpt'))):

            os.makedirs(os.path.expanduser('~/.cache/lama'), exist_ok=True)
            with urllib.request.urlopen('https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip') as remote_file:
                with zipfile.ZipFile(io.BytesIO(remote_file.read())) as zfile:
                    zfile.extractall(os.path.expanduser('~/.cache/lama'))
        model_path = os.path.expanduser('~/.cache/lama/big-lama')

        # load pre-trained model
        model_config_path = os.path.join(model_path, 'config.yaml')
        checkpoint_path = os.path.join(model_path, 'models', 'best.ckpt')
        with open(model_config_path, 'r') as f:
            model_config = OmegaConf.create(yaml.safe_load(f))
        model_config.training_model.predict_only = True
        model_config.visualizer.kind = 'noop'
        self.model = load_checkpoint(model_config, checkpoint_path, strict=False, map_location='cpu')
        self.model.freeze()

        self.pad_to_modulo = 8

    def to(self, device: torch.device):
        if self.refine and self.model.device != device:
            raise ValueError('Cannot change device of LamaInpainter when refining.')
        else:
            self.model.to(device)
    
    def inpaint(self, image, mask):
        image = image.squeeze(0)
        mask = mask.squeeze(0)
        batch_size = image.shape[0]

        original_img_size = None
        if self.pad_to_modulo is not None and self.pad_to_modulo > 1:
            pad_to_modulo(img=image, mod=self.pad_to_modulo)
            pad_to_modulo(img=mask, mod=self.pad_to_modulo)
            original_img_size = image.shape[-2:]

        for b in range(batch_size):
            # arrange inputs into a dict (required by the model)
            batch = {
                'image': image[b:b+1],
                'mask': mask[b:b+1],
            }
            if original_img_size is not None:
                batch['unpad_to_size'] = original_img_size
            
            # run inpainter
            if self.refine:
                # run inpainter with refinement

                inpainted_image = refine_predict(batch, self.model, **self.refiner_config)
            else:
                # run inpainter without refinement

                with torch.no_grad():
                    # binarize mask
                    batch['mask'] = (batch['mask'] > 0) * 1
                    
                    # run inpainter
                    inpainted_image = self.model(batch)['inpainted']

                    # unpad image back to original size
                    if original_img_size is not None:
                        orig_height, orig_width = original_img_size
                        inpainted_image = inpainted_image[:orig_height, :orig_width]

            inpainted_image = torch.clamp(inpainted_image, 0, 1)

        return inpainted_image


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod

def pad_to_modulo(img, mod):
    batch_size, channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return torch.nn.functional.pad(img, pad=(0, out_width - width, 0, out_height - height), mode='reflect')
