import os
import glob

import torch
import torchvision
from PIL import Image

from zoe_depth_estimator import ZoeDepthEstimator


def test_depth_estimation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_img_dir = '../../data/test/m4/composites/'

    input_image_paths = glob.glob(f'{input_img_dir}/*.png', recursive=False)

    depth_estimator = ZoeDepthEstimator()
    depth_estimator.to(device)

    for input_image_path in input_image_paths:
        if input_image_path.endswith('_depth.png'):
            continue

        img = load_image(input_image_path).unsqueeze(dim=0)
        img = img.to(device)
        
        with torch.no_grad():
            depth = depth_estimator.estimate_depth(img=img)

        save_image((depth/depth.max())[0, 0].detach().cpu(), f'{os.path.splitext(input_image_path)[0]}_depth.png')

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
    test_depth_estimation()
