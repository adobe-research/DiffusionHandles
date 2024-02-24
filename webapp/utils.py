import torch
import torchvision

def crop_and_resize(img: torch.Tensor, size: int) -> torch.Tensor:
    if img.shape[-2] != img.shape[-1]:
        img = torchvision.transforms.functional.center_crop(img, min(img.shape[-2], img.shape[-1]))
    img = torchvision.transforms.functional.resize(img, size=(size, size), antialias=True)
    return img
