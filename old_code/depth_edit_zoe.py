import numpy as np
import torch
import math
from PIL import Image
import cv2
import trimesh
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import torch.nn.functional as F
from scipy import ndimage
from scipy.sparse import linalg, diags, lil_matrix
from scipy.ndimage import convolve
import typing
from tuls_utils import bgr_to_gray, depth_to_points, read_image
from old_code.zoe_depth_estimatation import get_points, zoe_points_to_depth_merged, get_aligned_pts, get_aligned_pts_true_depth, get_aligned_pts_syn_depth
import matplotlib.pyplot as plt

img_dim = 512

def visualize_img(img, name):
    #upscale_ratio = 512 / img.shape[1]
    #img = img.repeat(upscale_ratio, axis = 0).repeat(upscale_ratio, axis = 1)
    img = (img - img.min())/(img.max() - img.min())
    img = (img*255).round().astype("uint8")
    img = Image.fromarray(img)
    img.save(name + '_map.png')
    return

def max_pool_numpy(mask, kernel_size):
    # Convert numpy mask to tensor
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)  # Convert to [1, 1, H, W] shape

    # Max pooling
    pooled_tensor = F.max_pool2d(mask_tensor, kernel_size)

    # Convert tensor back to numpy
    pooled_mask = pooled_tensor.squeeze().numpy()

    return pooled_mask


def laplacian(image):
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return convolve(image, laplacian_kernel, mode='constant')


def poisson_solve(input_image, mask):
    # Get the indices of the unknown pixels
    unknown_pixels = np.where(mask)

    #print(unknown_pixels)
    
    # Compute the number of unknown pixels
    num_unknowns = len(unknown_pixels[0])

    # Generate an index map for the unknown pixels
    index_map = -np.ones_like(input_image, dtype=int)
    index_map[unknown_pixels] = np.arange(num_unknowns)

    # Compute the Laplacian of the input image
    lap = laplacian(input_image)

    # Generate the system matrix
    A = lil_matrix((num_unknowns, num_unknowns))

    # Generate the right hand side of the linear system
    b = np.zeros(num_unknowns)

    for index, (y, x) in enumerate(zip(*unknown_pixels)):
        A[index, index] = 4

        if y > 0 and mask[y-1, x]:  # Check if the north neighbor exists and is in the masked region
            A[index, index_map[y-1, x]] = -1
        elif y > 0:
            b[index] += input_image[y-1, x]

        if y < input_image.shape[0] - 1 and mask[y+1, x]:  # Check if the south neighbor exists and is in the masked region
            A[index, index_map[y+1, x]] = -1
        elif y < input_image.shape[0] - 1:
            b[index] += input_image[y+1, x]

        if x > 0 and mask[y, x-1]:  # Check if the east neighbor exists and is in the masked region
            A[index, index_map[y, x-1]] = -1
        elif x > 0:
            b[index] += input_image[y, x-1]

        if x < input_image.shape[1] - 1 and mask[y, x+1]:  # Check if the west neighbor exists and is in the masked region
            A[index, index_map[y, x+1]] = -1
        elif x < input_image.shape[1] - 1:
            b[index] += input_image[y, x+1]

    # Solve the linear system
    solution = linalg.spsolve(A.tocsr(), b)

    # Generate the output image
    output_image = input_image.copy()
    output_image[unknown_pixels] = solution

    return output_image


def visualize_point_cloud(data_path):
    img = read_image(data_path + 'curr_depth.png')
    img = bgr_to_gray(img)
    img = 1 - img

    bg_img = read_image(data_path + 'curr_bg_depth.png')
    bg_img = bgr_to_gray(bg_img)
    bg_img = 1 - bg_img

    cut_img = Image.open(data_path + 'curr_mask.png')

    pts = depth_to_points(img, 70.0)

    bg_pts = depth_to_points(bg_img, 70.0)

    reshaped_orig_pts = pts.reshape((img_dim**2, 3))

    reshaped_bg_pts = bg_pts.reshape((img_dim**2, 3))
    
    return reshaped_orig_pts, reshaped_bg_pts

def stack_positions(original_x, original_y, transformed_x, transformed_y):
    # Stack the positions into a single array
    positions = np.stack((original_x, original_y, transformed_x, transformed_y), axis=-1)
    
    # Save the array using NumPy's save function
    return positions

def save_positions(original_x, original_y, transformed_x, transformed_y):
    # Stack the positions into a single array
    positions = np.stack((original_x, original_y, transformed_x, transformed_y), axis=-1)
    
    # Save the array using NumPy's save function
    np.save(filename, positions)

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


def transform_point_cloud(points, axis, angle_degrees, x, y, z, mask):
    """
    Rotate point cloud around the centroid of points selected by the mask.
    
    Parameters:
    - points: numpy array of shape (512, 512, 3)
    - axis: rotation axis, numpy array of shape (3,)
    - angle_degrees: rotation angle in degrees
    - mask: boolean array of shape (512, 512) indicating which pixels to consider for the centroid
    
    Returns:
    - rotated_points: numpy array of shape (512, 512, 3)
    """
    #cut_img = Image.open('car-cut.png')
    cut_img = mask
    #cut_img = Image.open('cup-table-cut (2).png')    
    img_tensor = np.array(cut_img)
    ref_mask = (img_tensor[:, :] == 255)
    mask = np.zeros_like(ref_mask, dtype = np.uint8) 
    mask[ref_mask.nonzero()] = 255 
    mask = max_pool_numpy(mask, 512 // img_dim)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Adjust the size as needed
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # Adjust the size as needed    
    #mask = cv2.erode(mask, kernel)
    #mask = cv2.dilate(mask, kernel)
    
    #visualize_img(mask, 'curr_mask')
    mask = (mask[:,:] != 0)

    modified_indices = mask.flatten()  # Flattened version of the mask to match the reshaped points
    
    # Convert angle from degrees to radians
    angle = np.radians(angle_degrees)
    
    # Ensure axis is a unit vector
    axis = axis / np.linalg.norm(axis)
    
    # Compute the centroid of the masked points
    masked_points = points[mask]

    trimesh_pc = trimesh.points.PointCloud(vertices=masked_points)
    trimesh_pc.export("point_cloud_obj.glb")


    centroid = np.mean(masked_points, axis=0)
    
    # Translate points to place centroid at the origin
    translated_points = points - centroid
    
    # Flatten the translated points
    flattened_points = translated_points.reshape(-1, 3)
    
    # Use the Rodriguez rotation formula
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    
    term1 = flattened_points * cos_theta
    term2 = np.cross(axis, flattened_points) * sin_theta
    term3 = axis * np.dot(flattened_points, axis)[:, np.newaxis] * (1 - cos_theta)
    
    rotated_points_flattened = term1 + term2 + term3

    trimesh_pc = trimesh.points.PointCloud(vertices=rotated_points_flattened + centroid + np.array([x,y,z]))
    trimesh_pc.export("modified_point_cloud_obj.glb")
    
    # Reshape the points back to 512x512x3 and translate back to the original position
    rotated_points = rotated_points_flattened.reshape(512, 512, 3) + centroid + np.array([x, y, z])#+ np.array([ 0.0, 0.0, -0.175]) #+ np.array([ 0.5, 0.15, 1.0])#+ np.array([0, 0, -0.1])#+ np.array([-0.035, 0.01, 0.15])
    
    return rotated_points, modified_indices

# def modify_point_cloud(pts, x, y, z, mask):
#     #cut_img = Image.open('car-cut.png')
#     cut_img = mask
#     #cut_img = Image.open('cup-table-cut (2).png')    
#     img_tensor = np.array(cut_img)
#     ref_mask = (img_tensor[:, :] == 255)
#     mask = np.zeros_like(ref_mask, dtype = np.uint8) 
#     mask[ref_mask.nonzero()] = 255 
#     mask = max_pool_numpy(mask, 512 // img_dim)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # Adjust the size as needed
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # Adjust the size as needed    
#     #mask = cv2.erode(mask, kernel)
#     #mask = cv2.dilate(mask, kernel)
    
#     #visualize_img(mask, 'curr_mask')
#     mask = (mask[:,:] != 0)
        
#     pts[mask] += np.array([x,y,z])
    
#     modified_indices = mask.flatten()  # Flattened version of the mask to match the reshaped points
    
#     return pts, modified_indices  
    
def points_to_depth(points: torch.Tensor, fltFov: float, mod_ids: torch.Tensor, output_size=(512, 512), max_depth_value=float('inf')):
    if isinstance(points, np.ndarray):
        return points_to_depth(torch.FloatTensor(points), fltFov, mod_ids, output_size, max_depth_value)
    
    assert(points.ndim == 2 and points.shape[1] == 3 and points.dtype in [torch.float16, torch.float32, torch.float64])
    assert(fltFov > 0.0)

    #print(output_size)
    #fltFocal = 0.5 * max(output_size) * math.tan(math.radians(90.0) - (0.5 * math.radians(fltFov)))
    fltFocal = 0.5 * max(output_size) / np.tan(0.5 * 55 * np.pi / 180.0)

    u_0, v_0 = output_size[1] // 2, output_size[0] // 2

    tenDepth = torch.full(output_size, max_depth_value, dtype=points.dtype, device=points.device)
    target_mask = torch.full(output_size, False, dtype=torch.bool, device=points.device)
    modified_depth_mask = torch.full(output_size, False, dtype=torch.bool, device=points.device)
    # Initialize visibility mask for original positions with False
    original_visibility_mask = torch.full_like(mod_ids, False, dtype=torch.bool, device=mod_ids.device)
    # Keep track of which original point set the depth at each pixel
    depth_set_by = torch.full(output_size, -1, dtype=torch.int64, device=mod_ids.device)  # -1 indicates no point has set the depth yet
       
    # Project the points back to screen space
    tenScreenX = (points[:, 0] * fltFocal / points[:, 2] + u_0).type(torch.int)
    tenScreenY = (points[:, 1] * fltFocal / points[:, 2] + v_0).type(torch.int)

    mask_valid = (tenScreenX >= 0) & (tenScreenX < output_size[1]) & (tenScreenY >= 0) & (tenScreenY < output_size[0])
    
    # Use the valid coordinates to map points to the depth map, and choose the smallest depth for overlapping points

    for i in range(points.size(0)):
        if mask_valid[i]:
            x, y = tenScreenX[i], tenScreenY[i]
            if mod_ids[i]:  # Check if this point was modified
                if points[i, 2] < tenDepth[y, x]:  # Check if this point sets the depth
                    original_visibility_mask[i] = True
                    if depth_set_by[y, x] >= 0:
                        original_visibility_mask[depth_set_by[y, x]] = False  # Mark the overwritten point as not visible
                    target_mask[y, x] = True
                    modified_depth_mask[y, x] = True
                    depth_set_by[y, x] = i
                    # Check if a previous original point had set the depth for this pixel
            elif modified_depth_mask[y, x] and points[i, 2] < tenDepth[y, x]:  # Check if a non-modified point overwrites the depth
                target_mask[y, x] = False
                if depth_set_by[y, x] >= 0:
                    original_visibility_mask[depth_set_by[y, x]] = False  # Mark the overwritten point as not visible
                depth_set_by[y, x] = i                
            tenDepth[y, x] = min(tenDepth[y, x], points[i, 2])
            
    # Find the pixels that no points map to
    mask_no_points = tenDepth == max_depth_value
    pixels_no_points = torch.nonzero(mask_no_points, as_tuple=False)
    #toggle for incorporating visibility basically
    #original_visibility_mask = mod_ids

    return tenDepth, pixels_no_points, target_mask, tenScreenX[original_visibility_mask], tenScreenY[original_visibility_mask], original_visibility_mask
    
# def estimate_depth(image, model_zoe_nk):
#     return predict_depth(image, model_zoe_nk)

def plot_img(img):
    img = (img - img.min())/(img.max() - img.min())
    img = (img*255).round().astype("uint8")
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show()    
    return

def normalize_depth_map(depth_map):
    width, height = depth_map.size
    depth_map = np.asarray(depth_map)
    depth_map = torch.from_numpy(np.array(depth_map))
    depth_map = depth_map.to("cuda", torch.float32)
    depth_map = depth_map.view(1, depth_map.shape[0], depth_map.shape[1])
    #print(depth_map.shape)
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(img_dim, img_dim),
        mode="bicubic",
        align_corners=False,
    )

    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    #depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)    
    depth_map = depth_map.to(torch.float32)
    output = depth_map.cpu().numpy()[0][0]
    #formatted = (output * 255 / np.max(output)).astype('uint8')
    #image = Image.fromarray(formatted)
    return output    




def umeyama_alignment(src, dst):
    """
    Perform the Umeyama alignment (rotation, scaling, and translation) between src and dst point sets.
    src, dst: Source and destination point sets (N, 3) 
    Return the aligned version of src.
    """
    # 1. Compute centroids
    centroid_src = np.mean(src, axis=0)
    centroid_dst = np.mean(dst, axis=0)
    
    # 2. Zero-mean the point sets
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst
    
    # 3. Compute the optimal rotation
    H = np.dot(src_centered.T, dst_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    # 4. Compute the optimal scale
    scale = np.sum(S) / np.sum(src_centered ** 2)
    
    # 5. Compute the translation
    t = centroid_dst - scale * np.dot(centroid_src, R)
    
    # Transform the source point set
    aligned_src = scale * np.dot(src, R) + t
    
    return scale, R, t

def align_point_clouds_with_mask_umeyama(pc1, pc2, mask_image):
    """Aligns pc2 to pc1 based on the Umeyama alignment of points outside the mask."""
    # Convert the mask image to a boolean numpy array
    mask_array = np.array(mask_image).astype(bool)
    
    # If the mask is a 3-channel image (like RGB), take any one channel
    if mask_array.shape[-1] == 3:
        mask_array = mask_array[..., 0]
    
    # Extract points outside the mask for each point cloud
    points_outside_mask_pc1 = pc1[~mask_array]
    points_outside_mask_pc2 = pc2[~mask_array]
    
    # Align points from pc2 to pc1 using the Umeyama method
    scale, R, t = umeyama_alignment(points_outside_mask_pc2, points_outside_mask_pc1)
    
    # Copy the aligned points to pc2
    #pc2[~mask_array] = aligned_pc2_outside_mask

    pc2 += [0,0,t[2]]
    
    return pc2


def depth_edit_transform(data_path, axis, angle, x, y, z, model_zoe_nk):

    img = Image.open(data_path + 'curr_image.png')
    bg_img = Image.open(data_path + 'curr_bg_image.png')
    cut_img = Image.open(data_path + 'curr_mask.png')
    #plot_img(img) 
    #plot_img(bg_img)

    print('getting points')

    #pts = get_points(img)
    #bg_pts = get_points(bg_img)

    print('computed points')

    #bg_pts = align_point_clouds_with_mask_umeyama(pts, bg_pts, cut_img)

    pts, bg_pts = get_aligned_pts(img, bg_img, cut_img, model_zoe_nk)

    #depth_map,  = zoe_points_to_depth_alt(pts.reshape((512**2, 3)), 512, 512)

    #plot_img(depth_map)

    #reshaped_orig_pts = pts.reshape((img_dim**2, 3))

    #reshaped_bg_pts = bg_pts.reshape((img_dim**2, 3))

    #trimesh_pc = trimesh.points.PointCloud(vertices=pts.reshape((512**2, 3)))
    #trimesh_pc.export("point_cloud.glb")

    trimesh_pc = trimesh.points.PointCloud(vertices=bg_pts.reshape((512**2, 3)))
    trimesh_pc.export("point_cloud_bg.glb")


    pts, mod_ids = transform_point_cloud(pts, axis, angle, x, y, z, cut_img)

    #points = pts.reshape((img_dim**2, 3))

    #orig_pts = pts


    if isinstance(mod_ids, np.ndarray):
        mod_ids = torch.from_numpy(mod_ids)

    #reproject points to depth map

    reshaped_bg_pts = bg_pts.reshape((img_dim**2, 3))

    reshaped_pts = pts.reshape((img_dim**2, 3))

    new_mod_ids = np.zeros(len(reshaped_bg_pts) + len(reshaped_pts[mod_ids]), dtype = np.uint8)

    new_mod_ids[np.arange(new_mod_ids.size) > len(reshaped_bg_pts) - 1] = 1

    modded_id_list = np.where(mod_ids)[0]

    idx_to_coord = {}

    for idx in modded_id_list:
        pt = reshaped_pts[idx]
        reshaped_bg_pts = np.vstack((reshaped_bg_pts, pt)) 
        idx_to_coord[len(reshaped_bg_pts) - 1] = divmod(idx, img_dim)
    
    rendered_depth, occluded_pixels, target_mask, transformed_positions_x, transformed_positions_y, orig_visibility_mask  = zoe_points_to_depth_merged(
        points=reshaped_bg_pts, 
        mod_ids=torch.from_numpy(new_mod_ids), 
        output_size=(img_dim, img_dim), 
        max_depth_value=reshaped_bg_pts[:, 2].max()
    )

    #plot_img(rendered_depth)

    infer_visible_original = np.zeros_like(mod_ids.reshape((img_dim,img_dim)), dtype = np.uint8)

    original_idxs = [idx_to_coord[key] for key in np.where(orig_visibility_mask)[0]]

    for idx in original_idxs:
        infer_visible_original[idx] = 1
    
    original_positions_y, original_positions_x = np.where(infer_visible_original)

    #target_mask_binary = target_mask.to(torch.int)
    # Convert the target mask to uint8
    #target_mask_uint8 = target_mask_binary.detach().cpu().numpy().astype(np.uint8) * 255

    target_mask_uint8 = target_mask.astype(np.uint8)*255

    # Define a kernel for the closing operation (you can adjust the size and shape)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (img_dim // 250 , img_dim // 250))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (img_dim // 50 , img_dim // 50))    

    # Perform the closing operation
    #target_mask_cleaned = cv2.morphologyEx(target_mask_uint8, cv2.MORPH_CLOSE, kernel)

    target_mask_cleaned = target_mask_uint8

    #target_mask_cleaned = cv2.medianBlur(target_mask_cleaned, 3)
    #target_mask_cleaned = cv2.medianBlur(target_mask_cleaned, 3)


    # Perform the closing operation
    #target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_OPEN, open_kernel)


    # Perform the closing operation
    target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_CLOSE, kernel)

    # Perform the closing operation
    target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_OPEN, open_kernel)


    # Perform the closing operation
    #target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_CLOSE, kernel)



    # Filter correspondences based on the mask
    filtered_original_x = []
    filtered_original_y = []
    filtered_transformed_x = []
    filtered_transformed_y = []

    for ox, oy, tx, ty in zip(original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y):
        if target_mask_cleaned[ty, tx] == 255:  # if the original point lies within the mask
            filtered_original_x.append(ox)
            filtered_original_y.append(oy)
            filtered_transformed_x.append(tx)
            filtered_transformed_y.append(ty)

    original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y = np.array(filtered_original_x), np.array(filtered_original_y), np.array(filtered_transformed_x), np.array(filtered_transformed_y)

    #save_positions(original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y, save_path + 'positions.npy')

    correspondences = stack_positions(original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y)

    img_tensor = np.array(cut_img)
    ref_mask = (img_tensor[:, :] == 255)
    mask = np.zeros_like(ref_mask, dtype = np.uint8) 
    mask[ref_mask.nonzero()] = 255 
    mask = max_pool_numpy(mask, 512 // img_dim)
    occluded_mask = mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    occluded_mask = cv2.dilate(occluded_mask.astype(np.uint8), kernel)

    #rendered_depth = rendered_depth.squeeze()
    #bg_img = bg_img[:,:,0]

    #visualize_img(rendered_depth.detach().cpu().numpy(), save_path + 'rendered_depth')

    # visualize_img(target_mask_cleaned, 'clean_target_mask')
    # visualize_img(target_mask_uint8, 'init_target_mask')    

    #plot_img(target_mask_cleaned)


    noise_mask = target_mask_uint8.astype(int) - target_mask_cleaned.astype(int)

    final_mask = target_mask_cleaned.astype(int) - target_mask_uint8.astype(int)
    final_mask[final_mask < 0] = 0
    noise_mask[noise_mask < 0] = 0

    #plot_img(final_mask)
    
    inpaint_mask = final_mask + noise_mask #+ occluded_mask
    inpaint_mask = (inpaint_mask > 0).astype(np.uint8)

    # visualize_img(inpaint_mask, 'inpaint_mask')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    inpaint_mask_dilated = cv2.dilate(inpaint_mask, kernel)



    lap_inpainted_depth_map = poisson_solve(np.array(rendered_depth), inpaint_mask_dilated)

    #lap_inpainted_depth_map[np.where(target_mask_cleaned == 0)] = 1 - bg_img[np.where(target_mask_cleaned == 0)]

    img = lap_inpainted_depth_map
    img = (img - img.min())/(img.max() - img.min())
    img = (img*255).round().astype("uint8")

    #plot_img(img)

    #visualize_img(img,'fixed_depth_map_denoised')

    img = target_mask_cleaned
    img = (img - img.min())/(img.max() - img.min())
    img = (img*255).round().astype("uint8")

    return lap_inpainted_depth_map, target_mask_cleaned, correspondences

def unpackRGBAtoDepth(img, scale=255.0):
    """
    This is what is being done in glsl:
    const float UnpackDownscale = 255. / 256.; // 0..1 -> fraction (excluding 1)
    const vec3 PackFactors = vec3( 256. * 256. * 256., 256. * 256.,  256. );
    const vec4 UnpackFactors = UnpackDownscale / vec4( PackFactors, 1. );
    float unpackRGBAToDepth( const in vec4 v ) {
        return dot( v, UnpackFactors );
    }
    """

    unpack_downscale = 255.0 / scale
    pack_factors = np.array([scale * scale * scale, scale * scale, scale, 1.0])
    unpack_factors = unpack_downscale / pack_factors

    return np.dot(img, unpack_factors)

def depth_edit_transform_true_depth(data_path, axis, angle, x, y, z, model_zoe_nk):

    img = Image.open(data_path + 'curr_image.png')
    bg_img = Image.open(data_path + 'curr_bg_image.png')
    cut_img = Image.open(data_path + 'curr_mask.png')
    fg_depth = Image.open(data_path + 'true_depth.png')

    fg_depth = unpackRGBAtoDepth(np.array(fg_depth))
    #plot_img(img) 
    #plot_img(bg_img)

    print('getting points')

    #pts = get_points(img)
    #bg_pts = get_points(bg_img)


    #bg_pts = align_point_clouds_with_mask_umeyama(pts, bg_pts, cut_img)

    pts, bg_pts = get_aligned_pts_true_depth(fg_depth, bg_img, cut_img, model_zoe_nk)

    #depth_map,  = zoe_points_to_depth_alt(pts.reshape((512**2, 3)), 512, 512)

    #plot_img(depth_map)

    #reshaped_orig_pts = pts.reshape((img_dim**2, 3))

    #reshaped_bg_pts = bg_pts.reshape((img_dim**2, 3))

    #trimesh_pc = trimesh.points.PointCloud(vertices=pts.reshape((512**2, 3)))
    #trimesh_pc.export("point_cloud.glb")

    print('computed points')


    trimesh_pc = trimesh.points.PointCloud(vertices=bg_pts.reshape((512**2, 3)))
    trimesh_pc.export("point_cloud_bg.glb")


    pts, mod_ids = transform_point_cloud(pts, axis, angle, x, y, z, cut_img)

    #points = pts.reshape((img_dim**2, 3))

    #orig_pts = pts


    if isinstance(mod_ids, np.ndarray):
        mod_ids = torch.from_numpy(mod_ids)

    #reproject points to depth map

    reshaped_bg_pts = bg_pts.reshape((img_dim**2, 3))

    reshaped_pts = pts.reshape((img_dim**2, 3))

    new_mod_ids = np.zeros(len(reshaped_bg_pts) + len(reshaped_pts[mod_ids]), dtype = np.uint8)

    new_mod_ids[np.arange(new_mod_ids.size) > len(reshaped_bg_pts) - 1] = 1

    modded_id_list = np.where(mod_ids)[0]

    idx_to_coord = {}

    for idx in modded_id_list:
        pt = reshaped_pts[idx]
        reshaped_bg_pts = np.vstack((reshaped_bg_pts, pt)) 
        idx_to_coord[len(reshaped_bg_pts) - 1] = divmod(idx, img_dim)
    
    rendered_depth, occluded_pixels, target_mask, transformed_positions_x, transformed_positions_y, orig_visibility_mask  = zoe_points_to_depth_merged(
        points=reshaped_bg_pts, 
        mod_ids=torch.from_numpy(new_mod_ids), 
        output_size=(img_dim, img_dim), 
        max_depth_value=reshaped_bg_pts[:, 2].max()
    )

    #plot_img(rendered_depth)

    infer_visible_original = np.zeros_like(mod_ids.reshape((img_dim,img_dim)), dtype = np.uint8)

    original_idxs = [idx_to_coord[key] for key in np.where(orig_visibility_mask)[0]]

    for idx in original_idxs:
        infer_visible_original[idx] = 1
    
    original_positions_y, original_positions_x = np.where(infer_visible_original)

    #target_mask_binary = target_mask.to(torch.int)
    # Convert the target mask to uint8
    #target_mask_uint8 = target_mask_binary.detach().cpu().numpy().astype(np.uint8) * 255

    target_mask_uint8 = target_mask.astype(np.uint8)*255

    # Define a kernel for the closing operation (you can adjust the size and shape)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (img_dim // 250 , img_dim // 250))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (img_dim // 50 , img_dim // 50))    

    # Perform the closing operation
    #target_mask_cleaned = cv2.morphologyEx(target_mask_uint8, cv2.MORPH_CLOSE, kernel)

    target_mask_cleaned = target_mask_uint8

    #target_mask_cleaned = cv2.medianBlur(target_mask_cleaned, 3)
    #target_mask_cleaned = cv2.medianBlur(target_mask_cleaned, 3)


    # Perform the closing operation
    #target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_OPEN, open_kernel)


    # Perform the closing operation
    target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_CLOSE, kernel)

    # Perform the closing operation
    target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_OPEN, open_kernel)


    # Perform the closing operation
    #target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_CLOSE, kernel)



    # Filter correspondences based on the mask
    filtered_original_x = []
    filtered_original_y = []
    filtered_transformed_x = []
    filtered_transformed_y = []

    for ox, oy, tx, ty in zip(original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y):
        if target_mask_cleaned[ty, tx] == 255:  # if the original point lies within the mask
            filtered_original_x.append(ox)
            filtered_original_y.append(oy)
            filtered_transformed_x.append(tx)
            filtered_transformed_y.append(ty)

    original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y = np.array(filtered_original_x), np.array(filtered_original_y), np.array(filtered_transformed_x), np.array(filtered_transformed_y)

    #save_positions(original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y, save_path + 'positions.npy')

    correspondences = stack_positions(original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y)

    img_tensor = np.array(cut_img)
    ref_mask = (img_tensor[:, :] == 255)
    mask = np.zeros_like(ref_mask, dtype = np.uint8) 
    mask[ref_mask.nonzero()] = 255 
    mask = max_pool_numpy(mask, 512 // img_dim)
    occluded_mask = mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    occluded_mask = cv2.dilate(occluded_mask.astype(np.uint8), kernel)

    #rendered_depth = rendered_depth.squeeze()
    #bg_img = bg_img[:,:,0]

    #visualize_img(rendered_depth.detach().cpu().numpy(), save_path + 'rendered_depth')

    # visualize_img(target_mask_cleaned, 'clean_target_mask')
    # visualize_img(target_mask_uint8, 'init_target_mask')    

    #plot_img(target_mask_cleaned)


    noise_mask = target_mask_uint8.astype(int) - target_mask_cleaned.astype(int)

    final_mask = target_mask_cleaned.astype(int) - target_mask_uint8.astype(int)
    final_mask[final_mask < 0] = 0
    noise_mask[noise_mask < 0] = 0

    #plot_img(final_mask)
    
    inpaint_mask = final_mask + noise_mask #+ occluded_mask
    inpaint_mask = (inpaint_mask > 0).astype(np.uint8)

    # visualize_img(inpaint_mask, 'inpaint_mask')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    inpaint_mask_dilated = cv2.dilate(inpaint_mask, kernel)



    lap_inpainted_depth_map = poisson_solve(np.array(rendered_depth), inpaint_mask_dilated)

    #lap_inpainted_depth_map[np.where(target_mask_cleaned == 0)] = 1 - bg_img[np.where(target_mask_cleaned == 0)]

    img = lap_inpainted_depth_map
    img = (img - img.min())/(img.max() - img.min())
    img = (img*255).round().astype("uint8")

    #plot_img(img)

    #visualize_img(img,'fixed_depth_map_denoised')

    img = target_mask_cleaned
    img = (img - img.min())/(img.max() - img.min())
    img = (img*255).round().astype("uint8")

    return lap_inpainted_depth_map, target_mask_cleaned, correspondences



def depth_edit_transform_syn_depth(data_path, axis, angle, x, y, z):

    img = Image.open(data_path + 'curr_image.png')
    #bg_img = Image.open(data_path + 'curr_bg_image.png')
    cut_img = Image.open(data_path + 'curr_mask.png')
    #fg_depth = Image.open(data_path + 'true_depth.png')

    #fg_depth = unpackRGBAtoDepth(np.array(fg_depth))
    #plot_img(img) 
    #plot_img(bg_img)

    print('getting points')

    #pts = get_points(img)
    #bg_pts = get_points(bg_img)


    #bg_pts = align_point_clouds_with_mask_umeyama(pts, bg_pts, cut_img)

    fg_depth = np.load(data_path + 'fg_depth.npy')
    bg_depth = np.load(data_path + 'bg_depth.npy')

    pts, bg_pts = get_aligned_pts_syn_depth(fg_depth, bg_depth)

    #depth_map,  = zoe_points_to_depth_alt(pts.reshape((512**2, 3)), 512, 512)

    #plot_img(depth_map)

    #reshaped_orig_pts = pts.reshape((img_dim**2, 3))

    #reshaped_bg_pts = bg_pts.reshape((img_dim**2, 3))

    #trimesh_pc = trimesh.points.PointCloud(vertices=pts.reshape((512**2, 3)))
    #trimesh_pc.export("point_cloud.glb")

    print('computed points')


    trimesh_pc = trimesh.points.PointCloud(vertices=bg_pts.reshape((512**2, 3)))
    trimesh_pc.export("point_cloud_bg.glb")


    pts, mod_ids = transform_point_cloud(pts, axis, angle, x, y, z, cut_img)

    #points = pts.reshape((img_dim**2, 3))

    #orig_pts = pts


    if isinstance(mod_ids, np.ndarray):
        mod_ids = torch.from_numpy(mod_ids)

    #reproject points to depth map

    reshaped_bg_pts = bg_pts.reshape((img_dim**2, 3))

    reshaped_pts = pts.reshape((img_dim**2, 3))

    new_mod_ids = np.zeros(len(reshaped_bg_pts) + len(reshaped_pts[mod_ids]), dtype = np.uint8)

    new_mod_ids[np.arange(new_mod_ids.size) > len(reshaped_bg_pts) - 1] = 1

    modded_id_list = np.where(mod_ids)[0]

    idx_to_coord = {}

    for idx in modded_id_list:
        pt = reshaped_pts[idx]
        reshaped_bg_pts = np.vstack((reshaped_bg_pts, pt)) 
        idx_to_coord[len(reshaped_bg_pts) - 1] = divmod(idx, img_dim)
    
    rendered_depth, occluded_pixels, target_mask, transformed_positions_x, transformed_positions_y, orig_visibility_mask  = zoe_points_to_depth_merged(
        points=reshaped_bg_pts, 
        mod_ids=torch.from_numpy(new_mod_ids), 
        output_size=(img_dim, img_dim), 
        max_depth_value=reshaped_bg_pts[:, 2].max()
    )

    #plot_img(rendered_depth)

    infer_visible_original = np.zeros_like(mod_ids.reshape((img_dim,img_dim)), dtype = np.uint8)

    original_idxs = [idx_to_coord[key] for key in np.where(orig_visibility_mask)[0]]

    for idx in original_idxs:
        infer_visible_original[idx] = 1
    
    original_positions_y, original_positions_x = np.where(infer_visible_original)

    #target_mask_binary = target_mask.to(torch.int)
    # Convert the target mask to uint8
    #target_mask_uint8 = target_mask_binary.detach().cpu().numpy().astype(np.uint8) * 255

    target_mask_uint8 = target_mask.astype(np.uint8)*255

    # Define a kernel for the closing operation (you can adjust the size and shape)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (img_dim // 250 , img_dim // 250))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (img_dim // 50 , img_dim // 50))    

    # Perform the closing operation
    #target_mask_cleaned = cv2.morphologyEx(target_mask_uint8, cv2.MORPH_CLOSE, kernel)

    target_mask_cleaned = target_mask_uint8

    #target_mask_cleaned = cv2.medianBlur(target_mask_cleaned, 3)
    #target_mask_cleaned = cv2.medianBlur(target_mask_cleaned, 3)


    # Perform the closing operation
    #target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_OPEN, open_kernel)


    # Perform the closing operation
    target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_CLOSE, kernel)

    # Perform the closing operation
    target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_OPEN, open_kernel)


    # Perform the closing operation
    #target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_CLOSE, kernel)



    # Filter correspondences based on the mask
    filtered_original_x = []
    filtered_original_y = []
    filtered_transformed_x = []
    filtered_transformed_y = []

    for ox, oy, tx, ty in zip(original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y):
        if target_mask_cleaned[ty, tx] == 255:  # if the original point lies within the mask
            filtered_original_x.append(ox)
            filtered_original_y.append(oy)
            filtered_transformed_x.append(tx)
            filtered_transformed_y.append(ty)

    original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y = np.array(filtered_original_x), np.array(filtered_original_y), np.array(filtered_transformed_x), np.array(filtered_transformed_y)

    #save_positions(original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y, save_path + 'positions.npy')

    correspondences = stack_positions(original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y)

    img_tensor = np.array(cut_img)
    ref_mask = (img_tensor[:, :] == 255)
    mask = np.zeros_like(ref_mask, dtype = np.uint8) 
    mask[ref_mask.nonzero()] = 255 
    mask = max_pool_numpy(mask, 512 // img_dim)
    occluded_mask = mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    occluded_mask = cv2.dilate(occluded_mask.astype(np.uint8), kernel)

    #rendered_depth = rendered_depth.squeeze()
    #bg_img = bg_img[:,:,0]

    #visualize_img(rendered_depth.detach().cpu().numpy(), save_path + 'rendered_depth')

    # visualize_img(target_mask_cleaned, 'clean_target_mask')
    # visualize_img(target_mask_uint8, 'init_target_mask')    

    #plot_img(target_mask_cleaned)


    noise_mask = target_mask_uint8.astype(int) - target_mask_cleaned.astype(int)

    final_mask = target_mask_cleaned.astype(int) - target_mask_uint8.astype(int)
    final_mask[final_mask < 0] = 0
    noise_mask[noise_mask < 0] = 0

    #plot_img(final_mask)
    
    inpaint_mask = final_mask + noise_mask #+ occluded_mask
    inpaint_mask = (inpaint_mask > 0).astype(np.uint8)

    # visualize_img(inpaint_mask, 'inpaint_mask')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    inpaint_mask_dilated = cv2.dilate(inpaint_mask, kernel)



    lap_inpainted_depth_map = poisson_solve(np.array(rendered_depth), inpaint_mask_dilated)

    #lap_inpainted_depth_map[np.where(target_mask_cleaned == 0)] = 1 - bg_img[np.where(target_mask_cleaned == 0)]

    img = lap_inpainted_depth_map
    img = (img - img.min())/(img.max() - img.min())
    img = (img*255).round().astype("uint8")

    #plot_img(img)

    #visualize_img(img,'fixed_depth_map_denoised')

    img = target_mask_cleaned
    img = (img - img.min())/(img.max() - img.min())
    img = (img*255).round().astype("uint8")

    return lap_inpainted_depth_map, target_mask_cleaned, correspondences


# def depth_edit_translate(data_path, x, y, z, model_zoe_nk):

#     img = Image.open(data_path + 'curr_image.png')
#     bg_img = Image.open(data_path + 'curr_bg_image.png')
#     cut_img = Image.open(data_path + 'curr_mask.png')
#     #plot_img(img) 
#     #plot_img(bg_img)

#     print('getting points')

#     pts = get_points(img, model_zoe_nk)
#     bg_pts = get_points(bg_img, model_zoe_nk)

#     print('computed points')

#     pts = align_point_clouds_with_mask_umeyama(bg_pts, pts, cut_img)

#     #depth_map,  = zoe_points_to_depth_alt(pts.reshape((512**2, 3)), 512, 512)

#     #plot_img(depth_map)

#     #reshaped_orig_pts = pts.reshape((img_dim**2, 3))

#     #reshaped_bg_pts = bg_pts.reshape((img_dim**2, 3))

#     #trimesh_pc = trimesh.points.PointCloud(vertices=pts.reshape((512**2, 3)))
#     #trimesh_pc.export("point_cloud.glb")

#     #trimesh_pc = trimesh.points.PointCloud(vertices=bg_pts.reshape((512**2, 3)))
#     #trimesh_pc.export("point_cloud_bg.glb")


#     pts, mod_ids = modify_point_cloud(pts, x, y, z, cut_img)

#     #points = pts.reshape((img_dim**2, 3))

#     #orig_pts = pts


#     if isinstance(mod_ids, np.ndarray):
#         mod_ids = torch.from_numpy(mod_ids)

#     #reproject points to depth map

#     reshaped_bg_pts = bg_pts.reshape((img_dim**2, 3))

#     reshaped_pts = pts.reshape((img_dim**2, 3))

#     new_mod_ids = np.zeros(len(reshaped_bg_pts) + len(reshaped_pts[mod_ids]), dtype = np.uint8)

#     new_mod_ids[np.arange(new_mod_ids.size) > len(reshaped_bg_pts) - 1] = 1

#     modded_id_list = np.where(mod_ids)[0]

#     idx_to_coord = {}

#     for idx in modded_id_list:
#         pt = reshaped_pts[idx]
#         reshaped_bg_pts = np.vstack((reshaped_bg_pts, pt)) 
#         idx_to_coord[len(reshaped_bg_pts) - 1] = divmod(idx, img_dim)
    
#     rendered_depth, occluded_pixels, target_mask, transformed_positions_x, transformed_positions_y, orig_visibility_mask  = zoe_points_to_depth_merged(
#         points=reshaped_bg_pts, 
#         mod_ids=torch.from_numpy(new_mod_ids), 
#         output_size=(img_dim, img_dim), 
#         max_depth_value=reshaped_bg_pts[:, 2].max()
#     )

#     #plot_img(rendered_depth)


#     infer_visible_original = np.zeros_like(mod_ids.reshape((img_dim,img_dim)), dtype = np.uint8)

#     original_idxs = [idx_to_coord[key] for key in np.where(orig_visibility_mask)[0]]

#     for idx in original_idxs:
#         infer_visible_original[idx] = 1
    
#     original_positions_y, original_positions_x = np.where(infer_visible_original)

#     #target_mask_binary = target_mask.to(torch.int)
#     # Convert the target mask to uint8
#     #target_mask_uint8 = target_mask_binary.detach().cpu().numpy().astype(np.uint8) * 255

#     target_mask_uint8 = target_mask.astype(np.uint8)*255

#     # Define a kernel for the closing operation (you can adjust the size and shape)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (img_dim // 50 , img_dim // 50))


#     # Perform the closing operation
#     #target_mask_cleaned = cv2.morphologyEx(target_mask_uint8, cv2.MORPH_CLOSE, kernel)

#     target_mask_cleaned = target_mask_uint8

#     # Perform the closing operation
#     target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_CLOSE, kernel)

#     # Perform the closing operation
#     target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_OPEN, kernel)

#     # Filter correspondences based on the mask
#     filtered_original_x = []
#     filtered_original_y = []
#     filtered_transformed_x = []
#     filtered_transformed_y = []

#     for ox, oy, tx, ty in zip(original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y):
#         if target_mask_cleaned[ty, tx] == 255:  # if the original point lies within the mask
#             filtered_original_x.append(ox)
#             filtered_original_y.append(oy)
#             filtered_transformed_x.append(tx)
#             filtered_transformed_y.append(ty)

#     original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y = np.array(filtered_original_x), np.array(filtered_original_y), np.array(filtered_transformed_x), np.array(filtered_transformed_y)

#     #save_positions(original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y, save_path + 'positions.npy')

#     correspondences = stack_positions(original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y)

#     img_tensor = np.array(cut_img)
#     ref_mask = (img_tensor[:, :] == 255)
#     mask = np.zeros_like(ref_mask, dtype = np.uint8) 
#     mask[ref_mask.nonzero()] = 255 
#     mask = max_pool_numpy(mask, 512 // img_dim)
#     occluded_mask = mask

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     occluded_mask = cv2.dilate(occluded_mask.astype(np.uint8), kernel)

#     #rendered_depth = rendered_depth.squeeze()
#     #bg_img = bg_img[:,:,0]

#     #visualize_img(rendered_depth.detach().cpu().numpy(), save_path + 'rendered_depth')

#     #visualize_img(target_mask_cleaned, 'clean_target_mask')
#     #visualize_img(target_mask_uint8, 'init_target_mask')    

#     #plot_img(target_mask_cleaned)

#     final_mask = target_mask_cleaned.astype(int) - target_mask_uint8.astype(int)
#     final_mask[final_mask < 0] = 0

#     #plot_img(final_mask)
    
#     inpaint_mask = final_mask #+ occluded_mask
#     inpaint_mask = (inpaint_mask > 0).astype(np.uint8)

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
#     inpaint_mask_dilated = cv2.dilate(inpaint_mask, kernel)

#     lap_inpainted_depth_map = poisson_solve(np.array(rendered_depth), inpaint_mask_dilated)

#     #lap_inpainted_depth_map[np.where(target_mask_cleaned == 0)] = 1 - bg_img[np.where(target_mask_cleaned == 0)]

#     img = lap_inpainted_depth_map
#     img = (img - img.min())/(img.max() - img.min())
#     img = (img*255).round().astype("uint8")

#     #plot_img(img)

#     #visualize_img(img,'fixed_depth_map_denoised')

#     img = target_mask_cleaned
#     img = (img - img.min())/(img.max() - img.min())
#     img = (img*255).round().astype("uint8")

#     return lap_inpainted_depth_map, target_mask_cleaned, correspondences


