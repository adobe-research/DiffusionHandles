from typing import Tuple

import torch
import numpy as np
import cv2
import scipy.sparse

from diffhandles.utils import pack_correspondences

def normalize_depth(depth, bounds=None, return_bounds=False):
    if depth.dim() != 4:
        raise RuntimeError(f'Expected depth to have 4 dimensions, got {depth.dim()}')
    
    if bounds is None:
        max_depth = depth.view(depth.shape[0], -1).max(dim=-1).values[..., None, None, None]
        min_depth = depth.view(depth.shape[0], -1).min(dim=-1).values[..., None, None, None]
    else:
        min_depth, max_depth = bounds
    
    if return_bounds:
        return 255 * (depth - min_depth) / (max_depth - min_depth), (min_depth, max_depth)
    else:
        return 255 * (depth - min_depth) / (max_depth - min_depth)

def transform_depth(
        pts: torch.Tensor, bg_pts: torch.Tensor, fg_mask: torch.Tensor,
        intrinsics: torch.Tensor, img_res: int,
        rot_angle: float, rot_axis: torch.Tensor, translation: torch.Tensor,
        depth_bounds: Tuple[float, float] = None):

    device = fg_mask.device
    
    pts, mod_ids = transform_point_cloud(
        points=pts.cpu().numpy(),
        axis=rot_axis.cpu().numpy(),
        angle_degrees=rot_angle,
        x=translation[0].item(),
        y=translation[1].item(),
        z=translation[2].item(),
        mask=fg_mask.cpu().numpy()[0, 0])

    #points = pts.reshape((self.img_res**2, 3))

    #orig_pts = pts

    if isinstance(mod_ids, np.ndarray):
        mod_ids = torch.from_numpy(mod_ids)

    #reproject points to depth map

    reshaped_bg_pts = bg_pts.cpu().numpy().reshape((img_res**2, 3))

    reshaped_pts = pts.reshape((img_res**2, 3))

    new_mod_ids = np.zeros(len(reshaped_bg_pts) + len(reshaped_pts[mod_ids]), dtype = np.uint8)

    new_mod_ids[np.arange(new_mod_ids.size) > len(reshaped_bg_pts) - 1] = 1

    modded_id_list = np.where(mod_ids)[0]

    idx_to_coord = {}

    for idx in modded_id_list:
        pt = reshaped_pts[idx]
        reshaped_bg_pts = np.vstack((reshaped_bg_pts, pt))
        idx_to_coord[len(reshaped_bg_pts) - 1] = divmod(idx, img_res)

    (rendered_depth, target_mask, transformed_positions_x, transformed_positions_y, orig_visibility_mask) = points_to_depth(
        points=torch.from_numpy(reshaped_bg_pts).to(device=device),
        intrinsics=intrinsics,
        output_size=(img_res, img_res),
        point_mask=torch.from_numpy(new_mod_ids).to(device=device),
    )

    # without conversion to disparty
    direct_depth = rendered_depth

    # get normalized disparity
    rendered_depth = normalize_depth(1.0/rendered_depth, bounds=depth_bounds)
    
    rendered_depth = rendered_depth[0, 0, ...].cpu().numpy()

    #plot_img(rendered_depth)

    infer_visible_original = np.zeros_like(mod_ids.reshape((img_res, img_res)), dtype = np.uint8)

    original_idxs = [idx_to_coord[key] for key in np.where(orig_visibility_mask)[0]]

    for idx in original_idxs:
        infer_visible_original[idx] = 1

    original_positions_y, original_positions_x = np.where(infer_visible_original)

    #target_mask_binary = target_mask.to(torch.int)
    # Convert the target mask to uint8
    #target_mask_uint8 = target_mask_binary.detach().cpu().numpy().astype(np.uint8) * 255

    target_mask_uint8 = target_mask.astype(np.uint8)*255

    # Define a kernel for the closing operation (you can adjust the size and shape)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (img_res // 250 , img_res // 250))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (img_res // 50 , img_res // 50))

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

    # correspondences = np.stack((original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y), axis=-1)
    correspondences = pack_correspondences(
        torch.from_numpy(original_positions_x),
        torch.from_numpy(original_positions_y),
        torch.from_numpy(transformed_positions_x),
        torch.from_numpy(transformed_positions_y))

    # # img_tensor = np.array(cut_img)
    # img_tensor = np.array(self.fg_mask.cpu().numpy()[0, 0]) # TODO: check that this works
    # ref_mask = (img_tensor[:, :] > 0.5)
    # mask = np.zeros_like(ref_mask, dtype = np.uint8)
    # mask[ref_mask.nonzero()] = 255
    # mask = max_pool_numpy(mask, 512 // self.img_res)
    # occluded_mask = mask

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # occluded_mask = cv2.dilate(occluded_mask.astype(np.uint8), kernel)

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

    # img = lap_inpainted_depth_map
    # img = (img - img.min())/(img.max() - img.min())
    # img = (img*255).round().astype("uint8")

    #plot_img(img)

    #visualize_img(img,'fixed_depth_map_denoised')

    # img = target_mask_cleaned
    # img = (img - img.min())/(img.max() - img.min())
    # img = (img*255).round().astype("uint8")

    lap_inpainted_depth_map = torch.from_numpy(lap_inpainted_depth_map).to(device=device, dtype=torch.float32)[None, None]
    target_mask_cleaned = torch.from_numpy(target_mask_cleaned.astype(np.float32) / 255.0).to(device=device)[None, None]

    return lap_inpainted_depth_map, target_mask_cleaned, correspondences, direct_depth

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
    # #cut_img = Image.open('car-cut.png')
    # # cut_img = mask
    # #cut_img = Image.open('cup-table-cut (2).png')    
    # img_tensor = np.array(mask)
    # ref_mask = (img_tensor[:, :] > 0.5)
    # mask = np.zeros_like(ref_mask, dtype = np.uint8) 
    # mask[ref_mask.nonzero()] = 255
    # # mask = max_pool_numpy(mask, 512 // img_dim) # doesn't do anything as img_dim=512

    # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Adjust the size as needed
    # #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # Adjust the size as needed    
    # #mask = cv2.erode(mask, kernel)
    # #mask = cv2.dilate(mask, kernel)
    
    # #visualize_img(mask, 'curr_mask')
    # mask = (mask[:,:] != 0)

    mask = mask.astype(bool)

    modified_indices = mask.flatten()  # Flattened version of the mask to match the reshaped points
    
    # Convert angle from degrees to radians
    angle = np.radians(angle_degrees)
    
    # Ensure axis is a unit vector
    axis = axis / np.linalg.norm(axis)
    
    # Compute the centroid of the masked points
    masked_points = points[mask]

    # trimesh_pc = trimesh.points.PointCloud(vertices=masked_points)
    # trimesh_pc.export("point_cloud_obj.glb")


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

    # trimesh_pc = trimesh.points.PointCloud(vertices=rotated_points_flattened + centroid + np.array([x,y,z]))
    # trimesh_pc.export("modified_point_cloud_obj.glb")
    
    # Reshape the points back to 512x512x3 and translate back to the original position
    rotated_points = rotated_points_flattened.reshape(512, 512, 3) + centroid + np.array([x, y, z])#+ np.array([ 0.0, 0.0, -0.175]) #+ np.array([ 0.5, 0.15, 1.0])#+ np.array([0, 0, -0.1])#+ np.array([-0.035, 0.01, 0.15])
    
    return rotated_points, modified_indices

def poisson_solve(input_image, mask):
    # Get the indices of the unknown pixels
    unknown_pixels = np.where(mask)

    #print(unknown_pixels)
    
    # Compute the number of unknown pixels
    num_unknowns = len(unknown_pixels[0])

    # Generate an index map for the unknown pixels
    index_map = -np.ones_like(input_image, dtype=int)
    index_map[unknown_pixels] = np.arange(num_unknowns)

    # # Compute the Laplacian of the input image
    # lap = laplacian(input_image)

    # Generate the system matrix
    A = scipy.sparse.lil_matrix((num_unknowns, num_unknowns))

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
    solution = scipy.sparse.linalg.spsolve(A.tocsr(), b)

    # Generate the output image
    output_image = input_image.copy()
    output_image[unknown_pixels] = solution

    return output_image

def depth_to_points(depth: torch.Tensor, intrinsics: torch.Tensor, extrinsics_R: torch.Tensor = None, extrinsics_t: torch.Tensor = None):

    device = depth.device
    
    if depth.shape[0] != 1:
        raise ValueError("Only batch size 1 is supported")

    depth = depth.squeeze(dim=0).cpu().numpy()
    intrinsics = intrinsics.numpy()
    intrinsics_inv = np.linalg.inv(intrinsics)
    if extrinsics_R is None:
        extrinsics_R = np.eye(3)
    else:
        extrinsics_R = extrinsics_R.cpu().numpy()
    if extrinsics_t is None:
        extrinsics_t = np.zeros(3)
    else:
        extrinsics_t = extrinsics_t.cpu().numpy()

    # M converts from your coordinate to PyTorch3D's coordinate system
    M = np.eye(3)
    M[0, 0] = -1.0
    M[1, 1] = -1.0

    height, width = depth.shape[1:3]

    # print(height)
    # print(width)

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    # coord = torch.as_tensor(coord, dtype=torch.float32, device=device)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    # print(D.shape, Kinv[None, None, None, ...].shape, coord[:, :, :, :, None].shape )
    points = D * intrinsics_inv[None, None, None, ...] @ coord[:, :, :, :, None]
    # pts3D_1 live in your coordinate system. Convert them to Py3D's
    points = M[None, None, None, ...] @ points
    # camera to world coordinates
    points = extrinsics_R[None, None, None, ...] @ points + extrinsics_t[None, None, None, :, None]

    return torch.from_numpy(points[:, :, :, :3, 0][0]).to(device=device)

def points_to_depth(points: torch.Tensor, intrinsics: torch.Tensor, output_size: Tuple[int, int], extrinsics_R: torch.Tensor = None, extrinsics_t: torch.Tensor = None, point_mask: torch.Tensor = None):
    """
    input points are expected to be in camera coordinate frame
    """
    
    device = points.device
    
    points = points.cpu().numpy()
    intrinsics = intrinsics.cpu().numpy()
    if extrinsics_R is None:
        extrinsics_R = np.eye(3)
    else:
        extrinsics_R = extrinsics_R.cpu().numpy()
    if extrinsics_t is None:
        extrinsics_t = np.zeros(3)
    else:
        extrinsics_t = extrinsics_t.cpu().numpy()
    if point_mask is None:
        point_mask = np.zeros(len(points), dtype = np.uint8)
    else:
        point_mask = point_mask.cpu().numpy()

    # to camera coordinates
    points = points[..., np.newaxis]
    points = np.linalg.inv(extrinsics_R) @ (points.T - extrinsics_t[:, None]).T
    points = points[:, :, 0]

    # M_inv converts to your coordinate from PyTorch3D's coordinate system
    M_inv = np.eye(3)
    M_inv[0, 0] = -1.0
    M_inv[1, 1] = -1.0
    points = (M_inv @ points.T).T

    # Projection to Image Plane
    projected = (intrinsics @ points.T).T
    u = projected[:, 0] / projected[:, 2]
    v = projected[:, 1] / projected[:, 2]

    u = np.around(np.clip(u, 0, output_size[1] - 1)).astype(int)
    v = np.around(np.clip(v, 0, output_size[0] - 1)).astype(int)

    depth_map = np.full(output_size, np.inf)
    # dist_to_cam = np.full(output_size, np.inf)
    depth_mask = np.full(output_size, False)
    modified_depth_mask = np.full(output_size, False)
    masked_point_visible_mask = np.full_like(point_mask, False)
    depth_set_by = np.full(output_size, -1, dtype=np.int64)

    # depth buffer
    for i in range(points.shape[0]):
        if points[i, 2] < depth_map[v[i], u[i]]:
            depth_map[v[i], u[i]] = points[i, 2]
            # dist_to_cam[v[i], u[i]] = (points[i,0] ** 2 + points[i, 1]**2 + points[i, 2]**2)**0.5
            if point_mask[i]:
                masked_point_visible_mask[i] = True
                if depth_set_by[v[i], u[i]] >= 0:
                    masked_point_visible_mask[depth_set_by[v[i], u[i]]] = False
                depth_mask[v[i], u[i]] = True
                modified_depth_mask[v[i], u[i]] = True
                depth_set_by[v[i], u[i]] = i
            elif modified_depth_mask[v[i], u[i]]:
                depth_mask[v[i], u[i]] = False
                if depth_set_by[v[i], u[i]] >= 0:
                    masked_point_visible_mask[depth_set_by[v[i], u[i]]] = False
                depth_set_by[v[i], u[i]] = i

    masked_point_visible_mask = masked_point_visible_mask.astype(bool)

    # mask_no_points = depth_map == max_depth_value
    # pixels_no_points = np.column_stack(np.where(mask_no_points))

    # far_inv_depth = 0.03 # inverse depth at far plane (empiricaly ~ similar to MiDaS depth ranges)
    # near_inv_depth = 100.0 # inverse depth at near plane (empiricaly ~ similar to MiDaS depth ranges)

    # #smoothed_dist_to_cam = cv2.medianBlur(dist_to_cam.astype(np.float32), ksize=3)
    # #dist_to_cam[dist_to_cam == np.inf] = smoothed_dist_to_cam[dist_to_cam == np.inf]

    # near_depth = dist_to_cam.min()
    # far_depth = dist_to_cam.max()

    # dist_to_cam = 1.0 / dist_to_cam
    # dist_to_cam = (dist_to_cam - 1/far_depth) / ((1/near_depth) - (1/far_depth))
    # dist_to_cam = dist_to_cam * (near_inv_depth - far_inv_depth) + far_inv_depth

    # #dist_to_cam = cv2.medianBlur(dist_to_cam.astype(np.float32), ksize = 1)
    
    # # depth to disparity
    # depth_map = 1.0 / depth_map

    #depth_map[depth_map == np.inf] = 1e-8
    #depth_map[depth_map == 0] = 1e-8

    #smoothed_depth_map = cv2.medianBlur(depth_map.astype(np.float32), ksize=3)
    #depth_map[depth_map < 1e-8] = smoothed_depth_map[depth_map < 1e-8]
    #smoothed_depth_map = depth_map #cv2.medianBlur(depth_map.astype(np.float32), ksize=1)
    # depth_image = Image.fromarray(depth_map_normalized.astype(np.uint8))

    depth_map = torch.from_numpy(depth_map)[None, None, ...].to(device=device, dtype=torch.float32)

    return depth_map, depth_mask, u[masked_point_visible_mask], v[masked_point_visible_mask], masked_point_visible_mask
