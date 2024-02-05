import os
import glob

import torch
import torchvision
from PIL import Image

from lang_sam import LangSAM


def test_foreground_selection():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_img_dir = '../../data/test/m4/fg_removal_inputs/'

    input_image_paths = glob.glob(f'{input_img_dir}/*.png', recursive=False)

    img_res = 512

    segmenter = LangSAM()
    segmenter.sam.model.to(device)
    segmenter.device = device

    for input_image_path in input_image_paths:
        fg_phrase_path = os.path.splitext(input_image_path)[0] + '_fg_phrase.txt'

        if input_image_path.endswith('_mask.png'):
            continue

        img = load_image(input_image_path).unsqueeze(dim=0)
        img = crop_and_resize(img=img, size=img_res)
        img = img.to(device)
        
        # segment the foreground object using SAM
        with open(fg_phrase_path, 'r') as f:
            fg_phrase = [phrase.strip() for phrase in f.read().splitlines() if phrase.strip() != '']
        if len(fg_phrase) != 1:
            print(f'Invalid phrase at {fg_phrase_path}.')
            continue

        masks, boxes, phrases, logits = segmenter.predict(
            image_pil=torchvision.transforms.functional.to_pil_image(img[0]),
            text_prompt=fg_phrase)
        del segmenter
        fg_mask = masks[0, None, None, :, :].to(device=device, dtype=torch.float32)
        
        save_image(fg_mask[0].detach().cpu(), f'{os.path.splitext(input_image_path)[0]}_mask.png')

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
    test_foreground_selection()
