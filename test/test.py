import torch
import torchvision
from PIL import Image
from diffhandles import ImageEditor


def test():
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

    img_editor = ImageEditor()
    img_editor.to(device)

    input_img = load_image(input_img_path).unsqueeze(dim=0)
    input_img = input_img.to(device)
    img_editor.set_input_image(img=input_img, prompt=prompt)
    # img_editor.save("../../data/test/temp.pt") # TEMP
    # img_editor = ImageEditor.load("../../data/test/temp.pt") # TEMP
    img_editor.select_foreground(fg_phrase=fg_phrase, bg_phrase=bg_phrase)
    # img_editor.save("../../data/test/temp2.pt") # TEMP 
    # img_editor = ImageEditor.load("../../data/test/temp2.pt") # TEMP
    for rot_angle in rot_angles:
        edited_img = img_editor.transform_foreground(
            rot_angle=rot_angle, rot_axis=rot_axis, translation=translation)
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
    test()
