import torch
import torchvision
from PIL import Image
import scipy
from lang_sam import LangSAM

from diffhandles import DiffusionHandles
from diffhandles.zoe_depth_estimator import ZoeDepthEstimator
from diffhandles.lama_inpainter import LamaInpainter


def test_diffusion_handles():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # sunflower stop-motion rotation
    rot_angles = [30.0, 55.0, 60.0]
    rot_axis = torch.tensor([0.0, 1.0, 0.0])
    translation = torch.tensor([0.0, 0.0, 0.0])
    input_img_path = 'data/sunflower.png'
    edited_img_path_template = 'results/sunflower'
    prompt = "a sunflower in the garden"
    fg_phrase = "sunflower"
    bg_phrase = "garden"

    diff_handles = DiffusionHandles()
    diff_handles.to(device)

    img = load_image(input_img_path).unsqueeze(dim=0)
    img = img.to(device)
    
    # check image resolution
    if img.shape[-2:] != (diff_handles.img_res, diff_handles.img_res):
        raise ValueError(f"Image must be of size {diff_handles.img_res}x{diff_handles.img_res}.")

    # segment the foreground object using SAM
    segmenter = LangSAM()
    segmenter.sam.model.to(device)
    segmenter.device = device
    masks, boxes, phrases, logits = segmenter.predict(
        image_pil=torchvision.transforms.functional.to_pil_image(img[0]),
        text_prompt=fg_phrase)
    del segmenter
    fg_mask = masks[0, None, None, :, :].to(device=device, dtype=torch.float32)

    # inpaint the foreground region to get a background image without the foreground object
    dilate_amount = 2
    fg_mask_dilated = fg_mask.cpu().numpy() > 0.5
    fg_mask_dilated = scipy.ndimage.binary_dilation(fg_mask_dilated[0, 0], iterations=dilate_amount)[None, None, ...]
    fg_mask_dilated = torch.from_numpy(fg_mask_dilated).to(device=device, dtype=torch.float32)
    inpainter = LamaInpainter()
    inpainter.to(device)
    bg_img = inpainter.inpaint(image=img, mask=fg_mask_dilated)
    del inpainter

    # estimate depth of the input image and the background image
    depth_estimator = ZoeDepthEstimator()
    depth_estimator.to(device)
    with torch.no_grad():
        depth = depth_estimator.estimate_depth(img=img)
        bg_depth = depth_estimator.estimate_depth(img=bg_img)
    del depth_estimator

    # select the foreground object
    inverted_noise, inverted_null_text, bg_depth, attentions, activations, activations2, activations3 = diff_handles.select_foreground(
        img=img, depth=depth, fg_mask=fg_mask, bg_depth=bg_depth,
        prompt=prompt, fg_phrase=fg_phrase, bg_phrase=bg_phrase)

    for rot_angle in rot_angles:

        # transform the foreground object
        edited_img = diff_handles.transform_foreground(
            depth=depth, fg_mask=fg_mask, bg_depth=bg_depth,
            prompt=prompt, fg_phrase=fg_phrase, bg_phrase=bg_phrase,
            inverted_null_text=inverted_null_text, inverted_noise=inverted_noise, 
            attentions=attentions, activations=activations, activations2=activations2, activations3=activations3,
            rot_angle=rot_angle, rot_axis=rot_axis, translation=translation)

        # save the edited image
        edited_img_path = f'{edited_img_path_template}_rotated_{rot_angle:.0f}.png'
        save_image(edited_img.detach().cpu().squeeze(dim=0), edited_img_path)

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
