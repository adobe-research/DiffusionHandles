import torch
import torchvision
from PIL import Image
import scipy

from diffhandles import DiffusionHandles
from zoe_depth_estimator import ZoeDepthEstimator
from lama_inpainter import LamaInpainter


def test_diffusion_handles():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # sunflower stop-motion rotation
    # rot_angles = [30.0, 55.0, 60.0]
    # rot_axis = torch.tensor([0.0, 1.0, 0.0])
    # translations = [torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])]
    # input_img_path = 'data/sunflower.png'
    # prompt = "a sunflower in the garden"
    # foreground_mask_path = None
    # edited_img_path_template = 'results/sunflower'
    # save_input_image = True
    # save_input_mask = True
    # save_depth = True
    # save_bg_image = True
    # save_recon_image = True

    # # 1000028042_low_cropped
    # rot_angles = [0.0]
    # rot_axis = torch.tensor([0.0, 1.0, 0.0])
    # translations = [torch.tensor([0.0, 0.0, 0.0])]
    # # translations = [torch.tensor([-0.5, 0.0, 0.0])]
    # input_img_path = '../../data/test/m4/1000028042_low_cropped.jpg'
    # # config_path = 'config/ex4.yaml'
    # config_path = 'config/ex4_test.yaml' # TEMP!! (comment in above)
    # prompt = "a colorful couch with stools in an office"
    # foreground_mask_path = '../../data/test/m4/1000028042_low_cropped_mask.png'
    # edited_img_path_template = 'results/1000028042_low_cropped'
    # save_input_image = True
    # save_input_mask = True
    # save_depth = True
    # save_bg_image = True
    # save_recon_image = True
    # load_intermediate_results = False
    # save_intermediate_results = True

    # # pexels-matheus-bertelli-17109108_low
    # rot_angles = [0.0]
    # rot_axis = torch.tensor([0.0, 1.0, 0.0])
    # # translations = [torch.tensor([0.0, 0.0, 0.0])]
    # translations = [torch.tensor([0.5, 0.0, 0.0])]
    # input_img_path = '../../data/test/m4/pexels-matheus-bertelli-17109108_low.jpg'
    # config_path = 'config/ex2.yaml'
    # prompt = "a spacious lobby"
    # foreground_mask_path = '../../data/test/m4/pexels-matheus-bertelli-17109108_low_mask.png'
    # edited_img_path_template = 'results/pexels-matheus-bertelli-17109108_low'
    # save_input_image = True
    # save_input_mask = True
    # save_depth = True
    # save_bg_image = True
    # save_recon_image = True
    # load_intermediate_results = False
    # save_intermediate_results = False

    # # ex2
    # rot_angles = [0.0]
    # rot_axis = torch.tensor([0.0, 1.0, 0.0])
    # # translations = [torch.tensor([0.0, 0.0, 0.0])]
    # translations = [torch.tensor([0.5, 0.0, 0.0])]
    # input_img_path = '../../data/test/m4/ex2_input.jpg'
    # config_path = 'config/ex2.yaml'
    # prompt = "a spacious lobby"
    # foreground_mask_path = '../../data/test/m4/ex2_fg_mask.png'
    # edited_img_path_template = 'results/ex2'
    # save_input_image = True
    # save_input_mask = True
    # save_depth = True
    # save_bg_image = True
    # save_recon_image = True
    # load_intermediate_results = False
    # save_intermediate_results = True

    # # ex1
    # rot_angles = [0.0]
    # rot_axis = torch.tensor([0.0, 1.0, 0.0])
    # # translations = [torch.tensor([0.0, 0.0, 0.0])]
    # translations = [torch.tensor([0.0, -0.4, 0.0])]
    # input_img_path = '../../data/test/m4/ex1_input.jpg'
    # prompt = "tables in a restaurant"
    # foreground_mask_path = '../../data/test/m4/ex1_fg_mask.png'
    # config_path = 'config/ex1.yaml'
    # edited_img_path_template = 'results/ex1'
    # save_input_image = True
    # save_input_mask = True
    # save_depth = True
    # save_bg_image = True
    # save_recon_image = True
    # load_intermediate_results = False
    # save_intermediate_results = True

    # # ex3
    # rot_angles = [0.0]
    # rot_axis = torch.tensor([0.0, 1.0, 0.0])
    # translations = [torch.tensor([-0.3, 0.0, 0.0])]
    # # translations = [torch.tensor([0.0, -0.4, 0.0])]
    # input_img_path = '../../data/test/m4/ex3_input.jpg'
    # prompt = "a corner of a room with a table and plants"
    # foreground_mask_path = '../../data/test/m4/ex3_fg_mask1.png'
    # config_path = 'config/ex3.yaml'
    # edited_img_path_template = 'results/ex3'
    # save_input_image = True
    # save_input_mask = True
    # save_depth = True
    # save_bg_image = True
    # save_recon_image = True
    # load_intermediate_results = False
    # save_intermediate_results = True

    # ex2_edit4
    rot_angles = [0.0]
    rot_axis = torch.tensor([0.0, 1.0, 0.0])
    # translations = [torch.tensor([0.0, 0.0, 0.0])]
    translations = [torch.tensor([0.4,-0.15,-0.5])]
    input_img_path = '../../data/m4/inputs/ex2_edit4/curr_image.png'
    config_path = 'config/ex2.yaml'
    prompt = "a spacious lobby"
    foreground_mask_path = '../../data/m4/inputs/ex2_edit4/curr_mask.png'
    edited_img_path_template = 'results/ex2_edit4'
    save_input_image = True
    save_input_mask = True
    save_depth = True
    save_bg_image = True
    save_recon_image = True
    load_intermediate_results = False
    save_intermediate_results = True

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
    

    diff_handles = DiffusionHandles(conf_path=config_path)
    diff_handles.to(device)

    if not load_intermediate_results:

        img = load_image(input_img_path).unsqueeze(dim=0)
        img = img.to(device)
        
        # check image resolution
        if img.shape[-2:] != (diff_handles.img_res, diff_handles.img_res):
            print(f"WARNING: Resizing and cropping image from {img.shape[-2]}x{img.shape[-1]} to {diff_handles.img_res}x{diff_handles.img_res}.")
            img = crop_and_resize(img=img, size=diff_handles.img_res)

        if save_input_image:
            save_image(img[0].detach().cpu(), f'{edited_img_path_template}_input.png')
        
        # load the foreground mask
        fg_mask = load_image(foreground_mask_path).unsqueeze(dim=0)
        if fg_mask.shape[1] > 1:
            fg_mask = fg_mask.mean(dim=1, keepdim=True) # average channels
        fg_mask = crop_and_resize(img=fg_mask, size=diff_handles.img_res)
        fg_mask = (fg_mask>0.5).to(device=device, dtype=torch.float32)

        if save_input_mask:
            save_image(fg_mask[0, 0].detach().cpu(), f'{edited_img_path_template}_mask.png')

        # inpaint the foreground region to get a background image without the foreground object
        fg_mask_dilated = fg_mask.cpu().numpy() > 0.5
        fg_mask_dilated = scipy.ndimage.binary_dilation(fg_mask_dilated[0, 0], iterations=2)[None, None, ...]
        fg_mask_dilated = torch.from_numpy(fg_mask_dilated).to(device=device, dtype=torch.float32)
        inpainter = LamaInpainter()
        inpainter.to(device)
        bg_img = inpainter.inpaint(image=img, mask=fg_mask_dilated)
        del inpainter

        if save_bg_image:
            save_image(bg_img[0].detach().cpu(), f'{edited_img_path_template}_bg.png')

        # estimate depth of the input image and the background image
        depth_estimator = ZoeDepthEstimator()
        depth_estimator.to(device)
        with torch.no_grad():
            depth = depth_estimator.estimate_depth(img=img)
            bg_depth = depth_estimator.estimate_depth(img=bg_img)
        del depth_estimator

        if save_depth:
            save_image((depth/depth.max())[0, 0].detach().cpu(), f'{edited_img_path_template}_depth.png')
            save_image((bg_depth/bg_depth.max())[0, 0].detach().cpu(), f'{edited_img_path_template}_bg_depth.png')

        # # set the input image to get inverted null text and noise
        # # (this requires using the same depth preprocessing we use for the later steps, so best not do it with a separate function)
        # inverted_null_text, inverted_noise, activations, activations2, activations3, latent_image = diff_handles.set_input_image(
        #     img=img, depth=depth, prompt=prompt)

        # # TEMP !!
        # if dilate_amount > 0:
        #     fg_mask = fg_mask.cpu().numpy() > 0.5
        #     fg_mask = scipy.ndimage.binary_dilation(fg_mask[0, 0], iterations=dilate_amount)[None, None, ...]
        #     fg_mask = torch.from_numpy(fg_mask).to(device=device, dtype=torch.float32)
        # # TEMP !!
        
        # # select the foreground object
        # bg_depth = diff_handles.select_foreground(
        #     depth=depth, fg_mask=fg_mask, bg_depth=bg_depth)

        bg_depth, inverted_null_text, inverted_noise, activations, activations2, activations3, latent_image = diff_handles.set_foreground(
            img=img, depth=depth, prompt=prompt, fg_mask=fg_mask, bg_depth=bg_depth)

        if save_recon_image:
            with torch.no_grad():
                latent_image = 1 / 0.18215 * latent_image.detach()
                recon_image = diff_handles.diffuser.vae.decode(latent_image)['sample']
                recon_image = (recon_image + 1) / 2
            save_image(recon_image[0].clamp(min=0, max=1).detach().cpu(), f'{edited_img_path_template}_recon_image.png')

        if save_intermediate_results:
            torch.save({
                'depth': depth,
                'prompt': prompt,
                'fg_mask': fg_mask,
                'bg_depth': bg_depth,
                'inverted_null_text': inverted_null_text,
                'inverted_noise': inverted_noise,
                'activations': activations,
                'activations2': activations2,
                'activations3': activations3,
            }, f'{edited_img_path_template}_intermediate_results.pt')
    else:
        intermediate_results = torch.load(f'{edited_img_path_template}_intermediate_results.pt')
        depth = intermediate_results['depth']
        prompt = intermediate_results['prompt']
        fg_mask = intermediate_results['fg_mask']
        bg_depth = intermediate_results['bg_depth']
        inverted_null_text = intermediate_results['inverted_null_text']
        inverted_noise = intermediate_results['inverted_noise']
        activations = intermediate_results['activations']
        activations2 = intermediate_results['activations2']
        activations3 = intermediate_results['activations3']
    
    for edit_idx, rot_angle, translation in enumerate(zip(rot_angles, translations)):

        # transform the foreground object
        edited_img, raw_edited_depth = diff_handles.transform_foreground(
            depth=depth, prompt=prompt,
            fg_mask=fg_mask, bg_depth=bg_depth,
            inverted_null_text=inverted_null_text, inverted_noise=inverted_noise, 
            activations=activations, activations2=activations2, activations3=activations3,
            rot_angle=rot_angle, rot_axis=rot_axis, translation=translation,
            use_input_depth_normalization=False)

        if save_depth:
            save_image((raw_edited_depth/raw_edited_depth.max())[0, 0].detach().cpu(), f'{edited_img_path_template}_edit_{edit_idx:03d}_raw_edited_depth.png')

        # save the edited image
        edited_img_path = f'{edited_img_path_template}_edit_{edit_idx:03d}.png'
        save_image(edited_img.detach().cpu().squeeze(dim=0), edited_img_path)

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
    test_diffusion_handles()
