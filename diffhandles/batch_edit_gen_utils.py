import torch
import torch.nn.functional as F
import math
from PIL import Image, ImageDraw, ImageFont
import logging
import os
import numpy as np
import cv2
import inspect
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import PIL_INTERPOLATION, deprecate, logging
import PIL

img_dim = 512

scene_dir = None
original_x = None
original_y = None
transformed_x = None
transformed_y = None
remaining_x = None
remaining_y = None
remaining_x_orig = None
remaining_y_orig = None
remaining_x_trans = None 
remaining_y_trans = None

def load_data(s_dir, t_dir):
    global scene_dir
    global original_x, original_y, transformed_x, transformed_y, remaining_x, remaining_y, remaining_x_orig, remaining_y_orig, remaining_x_trans, remaining_y_trans

    scene_dir = s_dir + t_dir

    original_x, original_y, transformed_x, transformed_y = load_positions(scene_dir + 'positions.npy')
    original_x, original_y = original_x // (img_dim // 64), original_y // (img_dim // 64)
    transformed_x, transformed_y = transformed_x // (img_dim // 64), transformed_y // (img_dim // 64)

    bg_original_x, bg_original_y, bg_transformed_x, bg_transformed_y = original_x, original_y, transformed_x, transformed_y

    # Create sets for original and transformed pixels
    original_pixels = set(zip(bg_original_x, bg_original_y))
    transformed_pixels = set(zip(bg_transformed_x, bg_transformed_y))

    # Create a set of all pixels in a 64x64 image
    all_pixels = {(x, y) for x in range(64) for y in range(64)}

    # Find pixels not in either of the original or transformed sets
    remaining_pixels = all_pixels - (original_pixels | transformed_pixels)

    # Extract remaining_x and remaining_y
    remaining_x = np.array([x for x, y in remaining_pixels])
    remaining_y = np.array([y for x, y in remaining_pixels])

    remaining_pixels_orig = all_pixels - (original_pixels)

    remaining_x_orig = np.array([x for x, y in remaining_pixels_orig])
    remaining_y_orig = np.array([y for x, y in remaining_pixels_orig])

    remaining_pixels_trans = all_pixels - (transformed_pixels)

    remaining_x_trans = np.array([x for x, y in remaining_pixels_trans])
    remaining_y_trans = np.array([y for x, y in remaining_pixels_trans])
    return 


def prepare_extra_step_kwargs(generator, eta, scheduler):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs

def prepare_latents(scheduler, vae, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
    if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        raise ValueError(
            f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
        )

    image = image.to(device=device, dtype=dtype)

    batch_size = batch_size * num_images_per_prompt

    if image.shape[1] == 4:
        init_latents = image

    else:
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        elif isinstance(generator, list):
            init_latents = [
                vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = vae.encode(image).latent_dist.sample(generator)

        init_latents = vae.config.scaling_factor * init_latents

    if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
        # expand init_latents for batch_size
        deprecation_message = (
            f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
            " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
            " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
            " your script to pass as many initial images as text prompts to suppress this warning."
        )
        #deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
        additional_image_per_prompt = batch_size // init_latents.shape[0]
        init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
    elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
        raise ValueError(
            f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
        )
    else:
        init_latents = torch.cat([init_latents], dim=0)

    shape = init_latents.shape
    noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    # get latents
    init_latents = scheduler.add_noise(init_latents, noise, timestep)
    latents = init_latents

    return latents


def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start * scheduler.order :]

    return timesteps, num_inference_steps - t_start
        
def preprocess(image):
    #deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"
    #deprecate("preprocess", "1.0.0", deprecation_message, standard_warn=False)
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image



def visualize_img(img, name):
    upscale_ratio = 512 / img.shape[1]
    img = img.repeat(upscale_ratio, axis = 0).repeat(upscale_ratio, axis = 1)
    img = (img - img.min())/(img.max() - img.min())
    img = (img*255).round().astype("uint8")
    img = Image.fromarray(img)
    img.save('./example_output/' + name + '_map.png')
    return 

def load_positions(filename):
    # Load the array using NumPy's load function
    positions = np.load(filename)
    
    # Split the array into original and transformed positions
    original_x, original_y, transformed_x, transformed_y = np.split(positions, 4, axis=-1)

    # Since np.split creates arrays of shape (N, 1), we'll squeeze them to get back to shape (N,)
    original_x = original_x.squeeze()
    original_y = original_y.squeeze()
    transformed_x = transformed_x.squeeze()
    transformed_y = transformed_y.squeeze()
    
    original_mask = np.zeros((img_dim,img_dim))
    
    transformed_mask = np.zeros((img_dim,img_dim))
    
    visible_orig_x = []
    visible_orig_y = []
    visible_trans_x = []
    visible_trans_y = []    
    
    for x, y, tx, ty in zip(original_x, original_y, transformed_x, transformed_y):
        if((tx >= 0 and tx < img_dim) and (ty >= 0 and ty < img_dim)):
            visible_orig_x.append(x)
            visible_orig_y.append(y)
            visible_trans_x.append(tx)
            visible_trans_y.append(ty)
    
    for x, y in zip(visible_orig_x, visible_orig_y):
        original_mask[y,x] = 1

    for x, y in zip(visible_trans_x, visible_trans_y):
        transformed_mask[y,x] = 1        
        
    # visualize_img(original_mask, 'original_mask')
    # visualize_img(transformed_mask,'transform_mask')
    
    return np.array(visible_orig_x), np.array(visible_orig_y), np.array(visible_trans_x), np.array(visible_trans_y) 


def has_good_coverage(original_x, original_y, block_x, block_y, scale_factor, coverage_threshold):
    # Count how many pixels in the block are in the original set
    coverage_count = np.sum((original_x // scale_factor == block_x) & (original_y // scale_factor == block_y))
    # Calculate the percentage of coverage
    coverage = coverage_count / (scale_factor * scale_factor)
    # Check if the coverage is above the threshold
    return coverage >= coverage_threshold

def downsample_coordinates(original_x, original_y, transformed_x, transformed_y, img_dim, downsampled_dim, coverage_threshold=0.75):
    scale_factor = img_dim // downsampled_dim
    
    # Downsample coordinates by dividing by the scale factor
    downsampled_original_x = original_x // scale_factor
    downsampled_original_y = original_y // scale_factor
    downsampled_transformed_x = transformed_x // scale_factor
    downsampled_transformed_y = transformed_y // scale_factor
    
    # Check each downsampled pixel for good coverage
    valid_downsampled_original_x = []
    valid_downsampled_original_y = []
    valid_downsampled_transformed_x = []
    valid_downsampled_transformed_y = []
    
    for i in range(len(downsampled_original_x)):
        if has_good_coverage(original_x, original_y, downsampled_original_x[i], downsampled_original_y[i], scale_factor, coverage_threshold):
            valid_downsampled_original_x.append(downsampled_original_x[i])
            valid_downsampled_original_y.append(downsampled_original_y[i])
            valid_downsampled_transformed_x.append(downsampled_transformed_x[i])
            valid_downsampled_transformed_y.append(downsampled_transformed_y[i])
    
    return (np.array(valid_downsampled_original_x), np.array(valid_downsampled_original_y),
            np.array(valid_downsampled_transformed_x), np.array(valid_downsampled_transformed_y))


def compute_attention_map_3dtransform(attention_map):
    
    orig_shape = attention_map.shape
    
    #original_x, original_y, transformed_x, transformed_y = load_positions('positions.npy')    

    attention_map = F.interpolate(attention_map.view(1, 1, attention_map.shape[0], attention_map.shape[1]), (512, 512), mode = 'bilinear')
    
    attention_map = attention_map[0][0]
    
    output_size = attention_map.shape

    # Create a new attention map
    new_attention_map = torch.zeros_like(attention_map)
    
    mod_orig_attention_map = torch.zeros_like(attention_map)

    for x, y in zip(original_x, original_y):
        mod_orig_attention_map[y, x] = attention_map[y, x]    
    
    # Transfer attention values for visible transformed pixels
    for orig_x, orig_y, trans_x, trans_y in zip(original_x, original_y, transformed_x, transformed_y):
        if 0 <= trans_x < output_size[1] and 0 <= trans_y < output_size[0]:  
            new_attention_map[trans_y, trans_x] = attention_map[orig_y, orig_x]
    new_attention_map = F.interpolate(new_attention_map.view(1, 1, new_attention_map.shape[0], new_attention_map.shape[1]), orig_shape, mode = 'bilinear')
    mod_orig_attention_map = F.interpolate(mod_orig_attention_map.view(1, 1, mod_orig_attention_map.shape[0], mod_orig_attention_map.shape[1]), orig_shape, mode = 'bilinear')
    
    new_attention_map = new_attention_map[0][0]
    mod_orig_attention_map = mod_orig_attention_map[0][0]
    # visualize_img(new_attention_map.detach().cpu().numpy(), 'new3d-trans-attention')
    # visualize_img(mod_orig_attention_map.detach().cpu().numpy(), 'mod-orig3d-attention')    
    return mod_orig_attention_map, new_attention_map


def blur_attn_map(attn_map):
    # Convert tensor to numpy
    numpy_array = attn_map.cpu().numpy().squeeze()

    # Apply Gaussian blur using OpenCV
    #blurred_array = cv2.GaussianBlur(numpy_array, (3,3), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Adjust the size as needed    
    
    attn_map = numpy_array

    # Perform the closing operation
    #attn_map = cv2.morphologyEx(attn_map, cv2.MORPH_CLOSE, kernel)

    # Perform the closing operation
    #attn_map = cv2.morphologyEx(attn_map, cv2.MORPH_OPEN, kernel)

    # (Optional) Perform additional dilation if needed
    #attn_map = cv2.GaussianBlur(attn_map, (3,3), 0)
    
    #blurred_array = attn_map
    
    #attn_map = cv2.erode(attn_map, kernel)

    # Convert back to tensor
    blurred_tensor = torch.tensor(attn_map).to("cuda")
    
    return blurred_tensor

def compute_size(attn_map):
    size = (1.0 / (attn_map.shape[0]**2)) * (attn_map.sum())
    return size



def normalize_attn_torch(attn_map):
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    attn_map = 10*(attn_map - 0.5)
    attn_map = torch.sigmoid(attn_map)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    return attn_map

def retrieve_attention_maps(attn_down, attn_mid, attn_up, obj_idx, object_positions, img_dims):
    
    attn_maps = [] 
    
    for i in range(len(attn_down)):
        attn_map = 0

        for attn_map_integrated in attn_down[i]:
            attn_map += attn_map_integrated
    
        attn_map /= len(attn_down[i])
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        
        ca_map_obj = 0
        for object_position in object_positions[obj_idx]:
            ca_map_obj += attn_map[:,:,object_position].reshape(b,H,W)

        ca_map_obj = ca_map_obj.mean(axis = 0)
        ca_map_obj = normalize_attn_torch(ca_map_obj)
        ca_map_obj = ca_map_obj.view(1, 1, H, W)
        #m = torch.nn.Upsample(scale_factor=img_dims / H, mode='nearest')
        #ca_map_obj = m(ca_map_obj)
        ca_map_obj = F.interpolate(ca_map_obj, (img_dims, img_dims), mode = 'bilinear')
        attn_maps.append(ca_map_obj[0][0])

    attn_map = 0

    for attn_map_integrated in attn_mid:
        attn_map += attn_map_integrated
    
    attn_map /= len(attn_mid)
    b, i, j = attn_map.shape
    H = W = int(math.sqrt(i))

    ca_map_obj = 0
    
    for object_position in object_positions[obj_idx]:
        ca_map_obj += attn_map[:,:,object_position].reshape(b,H,W)

    ca_map_obj = ca_map_obj.mean(axis = 0)
    ca_map_obj = normalize_attn_torch(ca_map_obj)
    ca_map_obj = ca_map_obj.view(1, 1, H, W)
    ca_map_obj = F.interpolate(ca_map_obj, (img_dims, img_dims), mode = 'bilinear')    
    #m = torch.nn.Upsample(scale_factor=img_dims / H, mode='nearest')
    #ca_map_obj = m(ca_map_obj)

    attn_maps.append(ca_map_obj[0][0])
    
    for i in range(len(attn_up)):
        attn_map = 0

        for attn_map_integrated in attn_up[i]:
            attn_map += attn_map_integrated
    
        attn_map /= len(attn_up[i])
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))

        ca_map_obj = 0
        for object_position in object_positions[obj_idx]:
            ca_map_obj += attn_map[:,:,object_position].reshape(b,H,W)

        ca_map_obj = ca_map_obj.mean(axis = 0)
        ca_map_obj = normalize_attn_torch(ca_map_obj)
        ca_map_obj = ca_map_obj.view(1, 1, H, W)
        ca_map_obj = F.interpolate(ca_map_obj, (img_dims, img_dims), mode = 'bilinear')        
        #m = torch.nn.Upsample(scale_factor=img_dims / H, mode='nearest')
        #ca_map_obj = m(ca_map_obj)
        attn_maps.append(ca_map_obj[0][0])
        
    return attn_maps

def translate_img(img, tx, ty):
    """
    Translate the input image by tx and ty pixels.

    Args:
    img : (torch.Tensor) input image tensor of shape (B, C, H, W)
    tx : (float) translation along the x axis
    ty : (float) translation along the y axis

    Returns:
    (torch.Tensor): Translated image of shape (B, C, H, W)
    """
    B, C, H, W = img.shape
    # Normalise the translations to be between -1 and 1
    tx_n = 2*tx/W
    ty_n = 2*ty/H

    # Create an identity affine transformation matrix
    theta = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float)
    theta = theta.repeat(B, 1, 1)

    # Apply the translations
    theta[:, 0, 2] = tx_n
    theta[:, 1, 2] = ty_n

    # Move the tensors to the same device as the input image
    if img.is_cuda:
        theta = theta.cuda()

    # Create a grid for the affine transformation
    grid = F.affine_grid(theta, size=(B, C, H, W))

    # Sample the input image at the grid points, with zero padding for out-of-bounds pixels
    img_t = F.grid_sample(img, grid, padding_mode='zeros')

    return img_t


def compute_gathered_loss_alt(attn_map, attn_map_gen, act_orig, act, patch_size=1):
    
    mask = torch.zeros_like(attn_map)
    trans_mask = torch.zeros_like(attn_map)
    mask[original_y, original_x] = 1
    attn_map = attn_map * mask    

    # Initialize the transformed attention map with zeros
    trans_attn_map = torch.zeros_like(attn_map)
    trans_attn_map[transformed_y, transformed_x] = attn_map[original_y, original_x]

    attn_map = blur_attn_map(attn_map)
    trans_attn_map = blur_attn_map(trans_attn_map)

    attn_map = mask 

    trans_mask[transformed_y, transformed_x] = mask[original_y, original_x]

    trans_attn_map = trans_mask
    
    
    # visualize_img(attn_map.detach().cpu().numpy(), 'tricky_attn')
    # visualize_img(trans_attn_map.detach().cpu().numpy(), 'trans_tricky_attn')
    
    #attn_map = blur_attn_map(attn_map)

    # Weight the activations
    weighted_act_orig = torch.mul(attn_map, act_orig) / torch.sum(attn_map)
    weighted_act = torch.mul(trans_attn_map, act) / torch.sum(trans_attn_map)

    
    # Compute spatially weighted mean difference
    diff = torch.abs(weighted_act[:, transformed_y, transformed_x] - weighted_act_orig[:, original_y , original_x]) #/ torch.sum(attn_map)
    
    per_pixel_diff = diff.mean(dim=1) 
    per_pixel_diff = per_pixel_diff.mean()
    
    kernel_size = patch_size
    padding = kernel_size // 2  # Set padding to ensure output size equals input size

    # Create average pooling layer
    pooling = torch.nn.AvgPool2d(kernel_size, stride=1, padding=padding)
    
    weighted_act_orig = attn_map * act_orig
    weighted_act = trans_attn_map * act    
 

    # Apply average pooling and multiply by kernel size to get the sum
    weighted_sums = pooling(weighted_act.unsqueeze(0)) * (kernel_size * kernel_size)
    weighted_sums_orig = pooling(weighted_act_orig.unsqueeze(0)) * (kernel_size * kernel_size)

    # Apply average pooling and multiply by kernel size to get the sum for attention maps
    weight_sums = pooling(trans_attn_map.unsqueeze(0).unsqueeze(0)) * (kernel_size * kernel_size)
    weight_sums_orig = pooling(attn_map.unsqueeze(0).unsqueeze(0)) * (kernel_size * kernel_size)
    
    # Avoid zero division using a small constant
    EPS = 1e-10
    weight_sums += EPS
    weight_sums_orig += EPS
    
    # Step 4: Divide to get the weighted average
    weighted_average = weighted_sums / weight_sums    
    weighted_average_orig = weighted_sums_orig / weight_sums_orig        
        
    weighted_average = weighted_average[0]
    weighted_average_orig = weighted_average_orig[0]
 
    # Compute spatially weighted mean difference
    diff = torch.abs(weighted_average[:, transformed_y, transformed_x] - weighted_average_orig[:, original_y , original_x]) #/ torch.sum(attn_map)
    #spatial_loss = torch.sum(diff * attn_map.unsqueeze(0)) / torch.sum(attn_map)
    
    per_patch_diff = diff.mean(dim = 1) #sum(dim=1).sum(dim=1) / (len(original_x))
    per_patch_diff = 1*per_patch_diff.mean()

    
    weighted_act_orig = torch.mul(attn_map, act_orig) / torch.sum(attn_map)
    weighted_act = torch.mul(trans_attn_map, act) / torch.sum(trans_attn_map)
    
    # Global mean difference
    global_weighted_act_orig = torch.sum(weighted_act_orig, dim=(1,2)) #/ attn_map.sum()
    global_weighted_act = torch.sum(weighted_act, dim=(1,2)) #/ trans_attn_map.sum()
    global_mean_diff = torch.abs(global_weighted_act_orig - global_weighted_act).mean()

    # Here you can return the individual loss components or a weighted sum of them.
    # I'll return the individual components for clarity.
    return per_pixel_diff, per_patch_diff, global_mean_diff    
    
def compute_localized_transformed_appearance_loss(attn_maps, activations, attn_maps_orig, activations_orig, attn_layer_low, attn_layer_high, patch_size):
    loss = 0
    for i in range(attn_layer_low,attn_layer_high):
        attn_map_orig = attn_maps_orig[i]
        attn_map = attn_maps[i]
        attn_map_orig = attn_map_orig.detach()  
        attn_map = attn_map.detach()
                        
        activations_orig_map = F.interpolate(activations_orig.view(1, activations_orig.shape[0], activations_orig.shape[1], activations_orig.shape[2]), (attn_map_orig.shape[0], attn_map_orig.shape[1]), mode = 'bilinear')
        activations_map = F.interpolate(activations.view(1, activations.shape[0], activations.shape[1], activations.shape[2]), (attn_map.shape[0], attn_map.shape[1]), mode = 'bilinear')
        pixel_loss, patch_loss, mean_loss = compute_gathered_loss_alt(attn_map_orig, attn_map, activations_orig_map[0], activations_map[0], patch_size)
        
        loss += 1.0*patch_loss

    loss /= (attn_layer_high - attn_layer_low) #len(attn_maps_orig)
    return loss



def compute_background_loss(attn_maps, activations, attn_maps_orig, activations_orig, attn_layer_low, attn_layer_high):
    loss = 0
 
    for i in range(attn_layer_low,attn_layer_high):
        attn_map_orig = attn_maps_orig[i]
        attn_map = attn_maps[i]
        # visualize_img(attn_map_orig.detach().cpu().numpy(), 'attn_orig_' + str(i))
        # visualize_img(attn_map.detach().cpu().numpy(), 'attn_' + str(i))        
        attn_map_orig = attn_map_orig.detach()  
        attn_map = attn_map.detach()

        mask = torch.zeros_like(attn_map)
        mask_orig = torch.zeros_like(attn_map)
        mask_trans = torch.zeros_like(attn_map)
        mask_full = torch.zeros_like(attn_map)        
        #print(mask.shape)
        mask[remaining_y, remaining_x] = 1
        mask_orig[remaining_y_orig, remaining_x_orig] = 1
        mask_trans[remaining_y_trans, remaining_x_trans] = 1
        mask_full[remaining_y_orig, remaining_x_orig] = 1
        mask_full[original_y, original_x] = 1
        mask_full[remaining_y_trans, remaining_x_trans] = 1
        # visualize_img(mask.detach().cpu().numpy(), 'bg_mask_basic_')
        # visualize_img(mask_orig.detach().cpu().numpy(), 'bg_mask_orig_')
        # visualize_img(mask_trans.detach().cpu().numpy(), 'bg_mask_trans_')        
        # visualize_img(mask_full.detach().cpu().numpy(), 'bg_mask_full_')                

        activations_orig_map = F.interpolate(activations_orig.view(1, activations_orig.shape[0], activations_orig.shape[1], activations_orig.shape[2]), (attn_map_orig.shape[0], attn_map_orig.shape[1]), mode = 'bilinear')
        appearance_orig = torch.mul(mask, activations_orig_map[0]) / mask.sum()
        activations_map = F.interpolate(activations.view(1, activations.shape[0], activations.shape[1], activations.shape[2]), (attn_map.shape[0], attn_map.shape[1]), mode = 'bilinear')

        appearance = torch.mul(mask, activations_map[0]) / mask.sum()

        app_orig_mean = torch.mul(mask_orig, activations_orig_map[0]) / mask_orig.sum()
        app_mean = torch.mul(mask_trans, activations_map[0]) / mask_trans.sum()
        # visualize_img(torch.mean(appearance, dim = 0).detach().cpu().numpy(), 'act_vis_' + str(i))
        # visualize_img(torch.mean(appearance_orig, dim = 0).detach().cpu().numpy(), 'act_vis_orig_' + str(i))
        app_orig_mean = torch.sum(app_orig_mean, dim = [1,2])
        app_mean = torch.sum(app_mean, dim = [1,2])

        app_loss_mean = torch.abs(app_orig_mean - app_mean)

        app_loss_per_pixel = torch.sum(torch.abs(appearance - appearance_orig), dim = [1,2])
        loss += 0.0*torch.mean(app_loss_per_pixel) + 1.0*torch.mean(app_loss_mean)
    return loss


    
def Pharse2idx(prompt, phrases):
    phrases = [x.strip() for x in phrases.split(';')]
    prompt_list = prompt.strip('.').split(' ')
    object_positions = []
    for obj in phrases:
        obj_position = []
        for word in obj.split(' '):
            obj_first_index = prompt_list.index(word) + 1
            obj_position.append(obj_first_index)
        object_positions.append(obj_position)

    return object_positions

