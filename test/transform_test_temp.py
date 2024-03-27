import scipy
import torch

from diffhandles.guided_stable_diffuser import GuidedStableDiffuser
from diffhandles.depth_transform import transform_depth, transform_depth_new
from diffhandles.utils import solve_laplacian_depth

from test_diffusion_handles import load_diffhandles_inputs
    
sample_dir = 'data/photogen/sunflower'
translation = [0.0, 0.0, 0.0]
rot_axis = [0.0, 1.0, 0.0]
rot_angle = 40

img_res = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img, fg_mask, depth, bg_depth = load_diffhandles_inputs(
    sample_dir=sample_dir, img_res=img_res, device=device)

# infill hole in the depth of the input image (where the foreground object used to be)
# with the depth of the background image
print('infilling depth ...')
bg_depth = solve_laplacian_depth(
    depth[0, 0].cpu().numpy(),
    bg_depth[0, 0].cpu().numpy(),
    scipy.ndimage.binary_dilation(fg_mask[0, 0].cpu().numpy(), iterations=15))
bg_depth = torch.from_numpy(bg_depth).to(device=device)[None, None]

# edit the 3D points
print('editing points ...')
with torch.no_grad():
    edited_disparity_orig, correspondences_orig = transform_depth(
        depth=depth, bg_depth=bg_depth, fg_mask=fg_mask,
        intrinsics=GuidedStableDiffuser.get_depth_intrinsics(device=depth.device),
        rot_angle=float(rot_angle),
        rot_axis=torch.tensor(rot_axis, dtype=torch.float32, device=device),
        translation=torch.tensor(translation, dtype=torch.float32, device=device),
        use_input_depth_normalization=False)

with torch.no_grad():
    edited_disparity_new, correspondences_new = transform_depth_new(
        depth=depth, bg_depth=bg_depth, fg_mask=fg_mask,
        intrinsics=GuidedStableDiffuser.get_depth_intrinsics(device=depth.device),
        rot_angle=float(rot_angle),
        rot_axis=torch.tensor(rot_axis, dtype=torch.float32, device=device),
        translation=torch.tensor(translation, dtype=torch.float32, device=device),
        use_input_depth_normalization=False)

import imageio
# print(f'min {edited_disparity.min()} max {edited_disparity.max()}')
imageio.imwrite('results/temp_depth_0.png', (((edited_disparity -  edited_disparity.min()) / (edited_disparity.max() - edited_disparity.min()))[0, ..., [0,0,0]]*255).to(dtype=torch.uint8).cpu().numpy())
# imageio.imwrite('results/temp_depth_1.png', (((edited_depth[1] -  edited_depth[1].min()) / (edited_depth[1].max() - edited_depth[1].min()))[..., None][..., [0,0,0]]*255).to(dtype=torch.uint8).cpu().numpy())

print('done')
