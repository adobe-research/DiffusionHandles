from dataclasses import dataclass

import torch
import torchvision
import gradio_client.client

def crop_and_resize(img: torch.Tensor, size: int) -> torch.Tensor:
    if img.shape[-2] != img.shape[-1]:
        img = torchvision.transforms.functional.center_crop(img, min(img.shape[-2], img.shape[-1]))
    img = torchvision.transforms.functional.resize(img, size=(size, size), antialias=True)
    return img

@dataclass
class GradioJob:
    job: gradio_client.client.Job = None
    time: float = 0.0
