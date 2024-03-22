from os.path import join, exists, basename
from os import makedirs
import json
from collections import OrderedDict
import argparse

import torch
from omegaconf import OmegaConf
from diffhandles import DiffusionHandles

from remove_foreground import remove_foreground
from estimate_depth import estimate_depth
from generate_results_webpage import generate_results_webpage
from utils import crop_and_resize, load_image, load_depth, save_image


def test_diffusion_handles(test_set_path:str, input_dir:str, output_dir:str, skip_existing:bool = False, config_path:str = None, device:str = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # # TEMP!
    # # need to:
    # # export CUBLAS_WORKSPACE_CONFIG=:4096:8
    # torch.use_deterministic_algorithms(True)
    # torch.manual_seed(76523524)
    # import random
    # random.seed(634566435)
    # import numpy as np
    # np.random.seed(54396745)
    # # TEMP!

    # load the test set info
    with open(test_set_path, 'r') as f:
        dataset_names = json.load(f, object_pairs_hook=OrderedDict)

    # check samples for completeness
    incomplete_samples = []
    foreground_removal_samples = []
    depth_estimation_samples = []
    for sample_name in dataset_names.keys():
        required_fnames = ['input.png', 'mask.png', 'prompt.txt', 'transforms.json']
        if any(not exists(join(input_dir, sample_name, fname)) for fname in required_fnames):
            print(f"Skipping sample {sample_name}, since it is missing one of the required input files ({required_fnames}).")
            incomplete_samples.append(sample_name)
            continue
        if not exists(join(input_dir, sample_name, 'bg_depth.exr')) and not exists(join(input_dir, sample_name, 'bg.png')):
            foreground_removal_samples.append(sample_name)
        if not exists(join(input_dir, sample_name, 'depth.exr')) or not exists(join(input_dir, sample_name, 'bg_depth.exr')):
            depth_estimation_samples.append(sample_name)

    # remove incomplete samples
    for sample_name in incomplete_samples:
        del dataset_names[sample_name]

    # estimate missing background images (with removed foreground object)
    if len(foreground_removal_samples) > 0:
        print(f"Estimating background images for {len(foreground_removal_samples)} samples ...")
        remove_foreground(
            input_image_paths=[join(input_dir, sample_name, 'input.png') for sample_name in foreground_removal_samples],
            foreground_mask_paths=[join(input_dir, sample_name, 'mask.png') for sample_name in foreground_removal_samples],
            output_paths=[join(input_dir, sample_name, 'bg.png') for sample_name in foreground_removal_samples])

    # estimate missing depths
    if len(depth_estimation_samples) > 0:
        print(f"Estimating depth and background depth for {len(depth_estimation_samples)} samples ...")
        estimate_depth(
            input_image_paths=[
                join(input_dir, sample_name, 'input.png') for sample_name in depth_estimation_samples]+[
                join(input_dir, sample_name, 'bg.png') for sample_name in depth_estimation_samples],
            output_paths=[
                join(input_dir, sample_name, 'depth.exr') for sample_name in depth_estimation_samples]+[
                join(input_dir, sample_name, 'bg_depth.exr') for sample_name in depth_estimation_samples]
            )

    diff_handles = DiffusionHandles(conf_path=config_path)
    diff_handles.to(device)

    # save config to output directory
    makedirs(output_dir, exist_ok=True)
    with open(join(output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=diff_handles.conf, f=f)

    # iterate over test set samples
    print(f"Transforming for {len(dataset_names)} samples ...")
    for sample_idx, (sample_name, transform_names) in enumerate(dataset_names.items()):

        # load prompt
        with open(join(input_dir, sample_name, 'prompt.txt'), 'r') as f:
            prompt = f.read().splitlines()
        prompt = [prompt for prompt in prompt if len(prompt) > 0]
        if len(prompt) == 0:
            print(f'WARNING: prompt for sample {sample_name} is empty. Skipping sample.')
            continue
        if len(prompt) > 1:
            print(f'WARNING: prompt for sample {sample_name} has multiple lines, only using the first line.')
        prompt = prompt[0]

        # load transforms
        with open(join(input_dir, sample_name, 'transforms.json'), 'r') as f:
            transforms = json.load(f, object_pairs_hook=OrderedDict)

        # if all outputs already exist, skip sample
        if skip_existing:
            transform_exists = OrderedDict()
            for transform_name, transform in transforms.items():
                transform_path = join(output_dir, sample_name, f'{transform_name}.png')
                transform_exists[transform_name] = exists(transform_path)
            if all(transform_exists.values()):
                print(f'Skipping sample {sample_name}, since all outputs already exist.')
                continue

        print(f"[{sample_idx+1}/{len(dataset_names)}] Transforming sample {sample_name} with {len(transforms)} transforms ...")

        makedirs(join(output_dir, sample_name), exist_ok=True)

        # save prompt to output directory
        with open(join(output_dir, sample_name, 'prompt.txt'), 'w') as f:
            f.write(f'{prompt}\n')

        # save transforms to output directory
        with open(join(output_dir, sample_name, 'transforms.json'), 'w') as f:
            json.dump(transforms, f, indent=4)

        # load inputs for the sample
        img, fg_mask, depth, bg_depth = load_diffhandles_inputs(
            sample_dir=join(input_dir, sample_name), img_res=diff_handles.img_res, device=device)

        # save inputs for visualization to results directory
        save_image(img[0], join(output_dir, sample_name, 'input.png'))
        save_image(fg_mask[0], join(output_dir, sample_name, 'mask.png'))
        save_image((depth/depth.max())[0], join(output_dir, sample_name, 'depth.png'))
        save_image((bg_depth/bg_depth.max())[0], join(output_dir, sample_name, 'bg_depth.png'))
        if exists(join(input_dir, sample_name, 'bg.png')):
            bg_img = load_image(join(input_dir, sample_name, 'bg.png'))[None, ...]
            bg_img = crop_and_resize(img=bg_img, size=diff_handles.img_res)
            save_image(bg_img[0], join(output_dir, sample_name, 'bg.png'))

        # set the foreground object to get inverted null text, noise, and intermediate activations to use as guidance
        bg_depth, inverted_null_text, inverted_noise, activations, activations2, activations3, latent_image = diff_handles.set_foreground(
            img=img, depth=depth, prompt=prompt, fg_mask=fg_mask, bg_depth=bg_depth)

        # save image reconstructed from inversion
        with torch.no_grad():
            latent_image = 1 / 0.18215 * latent_image.detach()
            recon_image = diff_handles.diffuser.vae.decode(latent_image)['sample']
            recon_image = (recon_image + 1) / 2
        save_image(recon_image.clamp(min=0, max=1)[0], join(output_dir, sample_name, 'recon.png'))

        # iterate over the transformations
        for transform_name in transform_names:

            if transform_name not in transforms:
                print(f'WARNING: Transform {transform_name} not found for sample {sample_name}. Skipping.')
                continue

            if skip_existing and transform_exists[transform_name]:
                print(f'Skipping transform {transform_name} of sample {sample_name}, since its output already exists.')
                continue

            transform = transforms[transform_name]

            # get transformation parameters
            translation = torch.tensor(transform['translation'], dtype=torch.float32) if 'translation' in transform else None
            rot_axis = torch.tensor(transform['rotation_axis'], dtype=torch.float32) if 'rotation_axis' in transform else None
            rot_angle = float(transform['rotation_angle']) if 'rotation_angle' in transform else None

            # transform the foreground object
            edited_img, raw_edited_depth, edited_disparity = diff_handles.transform_foreground(
                depth=depth, prompt=prompt,
                fg_mask=fg_mask, bg_depth=bg_depth,
                inverted_null_text=inverted_null_text, inverted_noise=inverted_noise,
                activations=activations, activations2=activations2, activations3=activations3,
                rot_angle=rot_angle, rot_axis=rot_axis, translation=translation,
                use_input_depth_normalization=False)

            # save the edited depth
            save_image((edited_disparity/edited_disparity.max())[0], join(output_dir, sample_name, f'{transform_name}_disparity.png'))
            save_image((raw_edited_depth/raw_edited_depth.max())[0], join(output_dir, sample_name, f'{transform_name}_depth_raw.png'))

            # save the edited image
            save_image(edited_img[0], join(output_dir, sample_name, f'{transform_name}.png'))

    # save sample names to result directory
    with open(join(output_dir, basename(test_set_path)), 'w') as f:
        json.dump(dataset_names, f, indent=4)

def load_diffhandles_inputs(sample_dir, img_res, device):

    # load the input image
    img = load_image(join(sample_dir, 'input.png'))[None, ...]
    if img.shape[-2:] != (img_res, img_res):
        print(f"WARNING: Resizing and cropping image from {img.shape[-2]}x{img.shape[-1]} to {img_res}x{img_res}.")
        img = crop_and_resize(img=img, size=img_res)
    img = img.to(device)

    # load the foreground mask
    fg_mask = load_image(join(sample_dir, 'mask.png'))[None, ...]
    if fg_mask.shape[1] > 1:
        fg_mask = fg_mask.mean(dim=1, keepdim=True) # average channels
    fg_mask = crop_and_resize(img=fg_mask, size=img_res)
    fg_mask = (fg_mask>0.5).to(device=device, dtype=torch.float32)

    # load the input image depth
    depth = load_depth(join(sample_dir, 'depth.exr'))[None, ...]
    depth = crop_and_resize(img=depth, size=img_res)
    depth = depth.to(device=device, dtype=torch.float32)

    # load the background depth
    bg_depth = load_depth(join(sample_dir, 'bg_depth.exr'))[None, ...]
    bg_depth = crop_and_resize(img=bg_depth, size=img_res)
    bg_depth = bg_depth.to(device=device, dtype=torch.float32)

    return img, fg_mask, depth, bg_depth

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set_path', type=str, default='data/photogen/photogen.json')
    parser.add_argument('--input_dir', type=str, default='data/photogen')
    parser.add_argument('--output_dir', type=str, default='results/photogen')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    test_diffusion_handles(test_set_path=args.test_set_path, input_dir=args.input_dir, output_dir=args.output_dir, skip_existing=args.skip_existing, config_path=args.config_path, device=args.device)
    generate_results_webpage(test_set_path=join(args.output_dir, basename(args.test_set_path)), website_path=join(args.output_dir, 'summary.html'), relative_image_dir='.')
