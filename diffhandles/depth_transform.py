import os

import numpy as np
import torch

from depth_edit_zoe import depth_edit_transform, visualize_img, plot_img

curr_dir = '../data/test/a_sunflower_in_the_garden_sunflower/'

x, y, z = 0.0, 0.0, 0.0

axis = np.array([0.0, 1.0, 0.0]) 

angle = 55

repo = "isl-org/ZoeDepth"
model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)

transformed_depth, target_mask, correspondences = depth_edit_transform(curr_dir, axis, angle, x, y, z, model_zoe_nk)

# plot_img(transformed_depth)

transform_dir = 'transform_' + str(x) + '_' + str(y) + '_' + str(z) + '_' + str(angle) + '/'
transform_dir = curr_dir + transform_dir

if not os.path.exists(os.path.dirname(transform_dir)):
    os.makedirs(os.path.dirname(transform_dir))

# visualize_img(transformed_depth, transform_dir + 'transformed_depth')
# visualize_img(target_mask, transform_dir + 'target_mask')
np.save(transform_dir + 'positions.npy', correspondences)