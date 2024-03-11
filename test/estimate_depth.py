import os
import glob

import torch
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

from utils import load_image, crop_and_resize, save_depth


def estimate_depth(input_image_paths, output_paths, img_res=512):

    if len(input_image_paths) != len(output_paths):
        raise ValueError('Length of input_image_paths and output_paths should be the same.')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conf = get_config("zoedepth_nk", "infer")
    depth_estimator = build_model(conf)
    depth_estimator.to(device)

    for input_image_path, output_path in zip(input_image_paths, output_paths):

        img = load_image(input_image_path).unsqueeze(dim=0)
        if img_res is not None:
            img = crop_and_resize(img=img, size=img_res)
        img = img.to(device)
        
        with torch.no_grad():
            depth = depth_estimator.infer(img)

        save_depth(depth[0], output_path)

if __name__ == '__main__':

    input_img_dir = 'data'

    input_image_paths = glob.glob(f'{input_img_dir}/*/input.png', recursive=False)
    bg_image_paths = glob.glob(f'{input_img_dir}/*/bg.png', recursive=False)
    output_paths = [
        os.path.join(os.path.dirname(path), 'depth.exr') for path in input_image_paths] + [
        os.path.join(os.path.dirname(path), 'bg_depth.exr') for path in bg_image_paths]
    input_image_paths = input_image_paths + bg_image_paths

    estimate_depth(input_image_paths=input_image_paths, output_paths=output_paths)
