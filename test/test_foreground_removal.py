import os
import glob

import torch
import torchvision
import scipy
from PIL import Image

from lama_inpainter import LamaInpainter


def test_foreground_removal():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_img_dir = '../../data/test/m4/fg_removal_inputs/'

    input_image_paths = glob.glob(f'{input_img_dir}/*.png', recursive=False)

    img_res = 512

    inpainter = LamaInpainter()
    inpainter.to(device)

    for input_image_path in input_image_paths:
        foreground_mask_path = os.path.splitext(input_image_path)[0] + '_mask.png'

        if input_image_path.endswith('_bg.png') or input_image_path.endswith('_mask.png'):
            continue

        img = load_image(input_image_path).unsqueeze(dim=0)
        img = crop_and_resize(img=img, size=img_res)
        img = img.to(device)
        
        # segment the foreground object using SAM
        fg_mask = load_image(foreground_mask_path).unsqueeze(dim=0)
        if fg_mask.shape[1] > 1:
            fg_mask = fg_mask.mean(dim=1, keepdim=True) # average channels
        fg_mask = crop_and_resize(img=fg_mask, size=img_res)
        fg_mask = (fg_mask>0.5).to(device=device, dtype=torch.float32)

        # inpaint the foreground region to get a background image without the foreground object
        fg_mask_dilated = fg_mask.cpu().numpy() > 0.5
        fg_mask_dilated = scipy.ndimage.binary_dilation(fg_mask_dilated[0, 0], iterations=2)[None, None, ...]
        fg_mask_dilated = torch.from_numpy(fg_mask_dilated).to(device=device, dtype=torch.float32)
        bg_img = inpainter.inpaint(image=img, mask=fg_mask_dilated)

        save_image(bg_img[0].detach().cpu(), f'{os.path.splitext(input_image_path)[0]}_bg.png')

def crop_and_resize(img: torch.Tensor, size: int) -> torch.Tensor:
    if img.shape[-2] != img.shape[-1]:
        img = torchvision.transforms.functional.center_crop(img, min(img.shape[-2], img.shape[-1]))
    img = torchvision.transforms.functional.resize(img, size=(size, size), antialias=True)
    return img

def load_image(path: str) -> torch.Tensor:
    img = Image.open(path)
    img = img.convert('RGB')
    img = torchvision.transforms.functional.pil_to_tensor(img)
    img = img / 255.0
    return img

def save_image(img: torch.Tensor, path: str):
    img = torchvision.transforms.functional.to_pil_image(img)
    img.save(path)

if __name__ == '__main__':
    test_foreground_removal()
