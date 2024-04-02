import torch
import torchvision
import imageio.v3 as imageio
import imageio.plugins as imageio_plugins

imageio_plugins.freeimage.download() # to load exr files

def load_image(path: str) -> torch.Tensor:
    # img = Image.open(path)
    # img = img.convert('RGB')
    # img = torchvision.transforms.functional.pil_to_tensor(img)

    img = torch.from_numpy(imageio.imread(path))
    if img.dim() == 2:
        img = img[..., None]
    img = img.to(dtype=torch.float32)
    img = img.permute(2, 0, 1)
    img = img / 255.0
    return img

def save_image(img: torch.Tensor, path: str):
    # img = torchvision.transforms.functional.to_pil_image(img)
    # img.save(path)

    img = img.detach().cpu()
    img = img * 255.0
    img = img.permute(1, 2, 0)
    img = img.to(dtype=torch.uint8)
    if img.shape[-1] == 1:
        img = img[..., 0]
    imageio.imwrite(path, img.numpy())
    
def load_depth(path: str) -> torch.Tensor:
    # depth = Image.open(path)
    # depth = torchvision.transforms.functional.pil_to_tensor(depth)[None,...]

    depth = torch.from_numpy(imageio.imread(path))
    if depth.dim() == 2:
        depth = depth[..., None]
    depth = depth.to(dtype=torch.float32)
    depth = depth.permute(2, 0, 1)
    return depth

def save_depth(depth: torch.Tensor, path: str):
    # depth = torchvision.transforms.functional.to_pil_image(depth, mode='F')
    # depth.save(path)

    depth = depth.detach().cpu()
    depth = depth.permute(1, 2, 0)
    depth = depth.to(dtype=torch.float32)
    depth = depth[..., 0]
    imageio.imwrite(path, depth.numpy())

def crop_and_resize(img: torch.Tensor, size: int) -> torch.Tensor:
    if img.shape[-2] != img.shape[-1]:
        img = torchvision.transforms.functional.center_crop(img, min(img.shape[-2], img.shape[-1]))
    img = torchvision.transforms.functional.resize(img, size=(size, size), antialias=True)
    return img
