import os
import glob

import torch
import torchvision
from lang_sam import LangSAM

from utils import crop_and_resize, load_image, save_image


def estimate_foreground(input_image_paths, foreground_prompt_paths, output_paths, img_res=512):

    if len(input_image_paths) != len(foreground_prompt_paths) or len(input_image_paths) != len(output_paths):
        raise ValueError('Length of input_image_paths, foreground_prompt_paths, and output_paths should be the same.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    segmenter = LangSAM()
    segmenter.sam.model.to(device)
    segmenter.device = device

    for input_image_path, fg_prompt_path, output_path in zip(input_image_paths, foreground_prompt_paths, output_paths):

        img = load_image(input_image_path).unsqueeze(dim=0)
        img = crop_and_resize(img=img, size=img_res)
        img = img.to(device)
        
        # segment the foreground object using SAM
        with open(fg_prompt_path, 'r') as f:
            fg_prompt = [prompt.strip() for prompt in f.read().splitlines() if prompt.strip() != '']
        if len(fg_prompt) != 1:
            print(f'Invalid prompt at {fg_prompt_path}.')
            continue
        fg_prompt = fg_prompt[0]

        masks, boxes, prompts, logits = segmenter.predict(
            image_pil=torchvision.transforms.functional.to_pil_image(img[0]),
            text_prompt=fg_prompt)
        # del segmenter
        fg_mask = masks[0, None, None, :, :].to(device=device, dtype=torch.float32)
        
        save_image(fg_mask[0], output_path)

if __name__ == '__main__':

    input_img_dir = 'data'

    input_image_paths = glob.glob(f'{input_img_dir}/*/input.png', recursive=False)
    foreground_prompt_paths = [os.path.join(os.path.dirname(path), 'fg_prompt.txt') for path in input_image_paths]
    output_paths = [os.path.join(os.path.dirname(path), 'mask.png') for path in input_image_paths]
    
    # only estimate foregrounds that have not yet been estimated
    missing_inds = [sample_idx for sample_idx, path in enumerate(output_paths) if not os.path.exists(path)]
    input_image_paths = [input_image_paths[i] for i in missing_inds]
    foreground_prompt_paths = [foreground_prompt_paths[i] for i in missing_inds]
    output_paths = [output_paths[i] for i in missing_inds]
            
    estimate_foreground(input_image_paths=input_image_paths, foreground_prompt_paths=foreground_prompt_paths, output_paths=output_paths)
