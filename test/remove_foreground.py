import os
import glob

import torch
import scipy

from saicinpainting import LamaInpainter
from utils import crop_and_resize, load_image, save_image


def remove_foreground(input_image_paths, foreground_mask_paths, output_paths, img_res=512, dilation=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if len(input_image_paths) != len(foreground_mask_paths) or len(input_image_paths) != len(output_paths):
        raise ValueError('Length of input_image_paths, foreground_mask_paths, and output_paths should be the same.')
    
    inpainter = LamaInpainter()
    inpainter.to(device)

    for input_image_path, foreground_mask_path, output_path in zip(input_image_paths, foreground_mask_paths, output_paths):

        # load input image
        img = load_image(input_image_path).unsqueeze(dim=0)
        img = crop_and_resize(img=img, size=img_res)
        img = img.to(device)
        
        # load the foreground mask
        fg_mask = load_image(foreground_mask_path).unsqueeze(dim=0)
        if fg_mask.shape[1] > 1:
            fg_mask = fg_mask.mean(dim=1, keepdim=True) # average channels
        fg_mask = crop_and_resize(img=fg_mask, size=img_res)
        fg_mask = (fg_mask>0.5).to(device=device, dtype=torch.float32)

        # inpaint the foreground region to get a background image without the foreground object
        if dilation >= 0:
            fg_mask = fg_mask.cpu().numpy() > 0.5
            fg_mask = scipy.ndimage.binary_dilation(fg_mask[0, 0], iterations=dilation)[None, None, ...]
            fg_mask = torch.from_numpy(fg_mask).to(device=device, dtype=torch.float32)
        bg_img = inpainter.inpaint(image=img, mask=fg_mask)

        # save the background image
        save_image(bg_img[0].detach().cpu(), output_path)

if __name__ == '__main__':

    input_img_dir = 'data'

    input_image_paths = glob.glob(f'{input_img_dir}/*/input.png', recursive=False)
    foreground_mask_paths = [os.path.join(os.path.dirname(path), 'mask.png') for path in input_image_paths]
    output_paths = [os.path.join(os.path.dirname(path), 'bg.png') for path in input_image_paths]

    remove_foreground(input_image_paths=input_image_paths, foreground_mask_paths=foreground_mask_paths, output_paths=output_paths)
