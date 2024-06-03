import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from typing import Tuple

import torch
import numpy as np
import cv2
import scipy.sparse

from diffhandles.utils import pack_correspondences
from diffhandles.mesh import Mesh
from diffhandles.renderer import Camera
from diffhandles.pytorch3d_renderer import PyTorch3DRenderer, PyTorch3DRendererArgs

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

def depth_to_mesh(depth: torch.Tensor, intrinsics: torch.Tensor, extrinsics_R: torch.Tensor = None, extrinsics_t: torch.Tensor = None, mask: torch.Tensor = None):
   
    if mask is not None:
        mask = mask.view(mask.shape[-2], mask.shape[-1])
    
    # create vertices by getting world position for each pixel of the depth map that is inside the mask
    verts = depth_to_world_coords(depth, intrinsics=intrinsics, extrinsics_R=extrinsics_R, extrinsics_t=extrinsics_t)
    if mask is not None:
        verts = verts[mask]
    verts = verts.view(-1, 3).contiguous()

    # get 2D coordinates in the depth image for each vertex
    vert_img_coords = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, depth.shape[-2], device=depth.device),
        torch.linspace(0, 1, depth.shape[-1], device=depth.device),
        indexing='xy'), dim=-1)
    if mask is not None:
        vert_img_coords = vert_img_coords[mask]
    vert_img_coords = vert_img_coords.view(-1, 2).contiguous()

    # create faces as two counter-clockwise triangles for each square spanned by 4 adjacent pixels that are all inside the mask
    # (upper left triangle, lower right triangle), assuming pixel index [0,0] is in the upper left corner
    if mask is not None:
        vertex_idx = torch.cumsum(mask.view(-1), dim=0).view(depth.shape[-2], depth.shape[-1])-1
        vertex_idx[~mask] = -1
    else:
        vertex_idx = torch.arange(depth.shape[-2]*depth.shape[-1], device=depth.device, dtype=torch.int64).view(depth.shape[-2], depth.shape[-1])
    tris_upper_left = torch.stack([x.reshape(-1) for x in [vertex_idx[1:, :-1], vertex_idx[:-1, 1:], vertex_idx[:-1, :-1]]], dim=-1)
    tris_lower_right = torch.stack([x.reshape(-1) for x in [vertex_idx[1:, :-1], vertex_idx[1:, 1:], vertex_idx[:-1, 1:]]], dim=-1)
    faces = torch.stack([tris_upper_left, tris_lower_right], dim=1).view(-1, 3)
    faces = faces[faces.min(dim=-1).values >= 0]
    faces = faces.contiguous()

    mesh = Mesh(verts=verts, faces=faces)

    # add vertex image coordinates and an indicator if a mask was given or not as color attribute (for rendering)
    mesh.add_vert_attribute("color", torch.cat([
        vert_img_coords,
        torch.full_like(vert_img_coords[:, [0]], fill_value=0 if mask is None else 1)
        ], dim=-1))

    return mesh

def transform_depth(
        depth: torch.Tensor, bg_depth: torch.Tensor, fg_mask: torch.Tensor, intrinsics: torch.Tensor,
        rot_angle: float = None, rot_axis: torch.Tensor = None, translation: torch.Tensor = None,
        use_input_depth_normalization = False, depth_transform_mode: str = "pc"):

    if depth_transform_mode == "mesh":
        return transform_depth_mesh(
            depth=depth, bg_depth=bg_depth, fg_mask=fg_mask, intrinsics=intrinsics,
            rot_angle=rot_angle, rot_axis=rot_axis, translation=translation,
            use_input_depth_normalization=use_input_depth_normalization)
    elif depth_transform_mode == "pc":
        return transform_depth_pc(
            depth=depth, bg_depth=bg_depth, fg_mask=fg_mask, intrinsics=intrinsics,
            rot_angle=rot_angle, rot_axis=rot_axis, translation=translation,
            use_input_depth_normalization=use_input_depth_normalization)
    else:
        raise ValueError(f"Unknown depth transform mode '{depth_transform_mode}'.")
    
def transform_depth_mesh(
        depth: torch.Tensor, bg_depth: torch.Tensor, fg_mask: torch.Tensor, intrinsics: torch.Tensor,
        rot_angle: float = None, rot_axis: torch.Tensor = None, translation: torch.Tensor = None,
        use_input_depth_normalization = False):

    if not fg_mask.any():
        # foreground mask is empty, there is no foreground object
        # return the image depth and empty correspondences
        if use_input_depth_normalization:
            _, depth_bounds = normalize_depth(1.0/depth, return_bounds=True)
        else:
            depth_bounds = None
        correspondences = pack_correspondences(
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
        )
        return normalize_depth(1.0/depth, bounds=depth_bounds), correspondences

    # default transformation parameters
    if rot_angle is None:
        rot_angle = 0.0
    if rot_axis is None:
        rot_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=depth.device)
    if translation is None:
        translation = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=depth.device)

    rot_angle = torch.tensor(rot_angle, dtype=torch.float32, device=depth.device)

    bg_depth_mesh = depth_to_mesh(depth=bg_depth, intrinsics=intrinsics)
    fg_depth_mesh = depth_to_mesh(depth=depth, intrinsics=intrinsics, mask=fg_mask[0, 0]>0.5)

    # transform foreground mesh
    fg_depth_mesh.verts.copy_(transform_points(
        points=fg_depth_mesh.verts, rot_angle=rot_angle, rot_axis=rot_axis, translation=translation))
    
    # verts = fg_depth_mesh.verts

    # # move centroid to origin to rotate about centroid
    # centroid = verts.mean(dim=0, keepdim=True)
    # verts = verts - centroid
    
    # # Use Rodriguez rotation formula to rotate with axis and angle
    # rot_axis = rot_axis / torch.linalg.norm(rot_axis, ord=2)
    # rot_angle = rot_angle * (torch.pi / 180.0)
    # cos_theta = torch.cos(rot_angle)
    # sin_theta = torch.sin(rot_angle)
    # term1 = verts * cos_theta
    # term2 = torch.cross(rot_axis[None, ...], verts) * sin_theta
    # term3 = rot_axis * torch.sum(verts * rot_axis[None, ...], dim=-1, keepdim=True) * (1 - cos_theta)
    # verts = term1 + term2 + term3

    # # move centroid back to original position and add translation
    # verts = verts + centroid + translation[None, ...]

    # fg_depth_mesh.verts.copy_(verts)

    camera = Camera(intrinsics=intrinsics)

    renderer = PyTorch3DRenderer(
        output_names=['world_position', 'flat_vertex_color'],
        args=PyTorch3DRendererArgs(
            device=depth.device,
            output_res=(depth.shape[-2], depth.shape[-1]),
            cull_backfaces=True,
            blur_radius=0.00001) # need a small blur radius otherwise the renderer may have artifacts if triangle edges align with pixel centers
    )
    renderer.update_scene(scene_elements={
        'meshes': [bg_depth_mesh, fg_depth_mesh],
        'cameras': [camera]})
    rendered_outputs = renderer.render()

    edited_depth = rendered_outputs['world_position'][None, ..., 2]
    edited_src_img_coords = rendered_outputs['flat_vertex_color'][0, ..., :2]
    edited_fg_mask = rendered_outputs['flat_vertex_color'][0, ..., 2] > 0.5

    edited_img_coords = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, edited_depth.shape[-2], device=edited_depth.device),
        torch.linspace(0, 1, edited_depth.shape[-1], device=edited_depth.device),
        indexing='xy'), dim=-1)

    edited_src_img_coords = edited_src_img_coords * torch.tensor([[depth.shape[-1]-1, depth.shape[-2]-1]], device=depth.device, dtype=torch.float32)
    edited_img_coords = edited_img_coords * torch.tensor([[edited_depth.shape[-1]-1, edited_depth.shape[-2]-1]], device=edited_depth.device, dtype=torch.float32)

    edited_src_img_coords = torch.round(edited_src_img_coords).to(dtype=torch.int64)
    edited_img_coords = torch.round(edited_img_coords).to(dtype=torch.int64)

    edited_src_img_coords = edited_src_img_coords[edited_fg_mask].to(device='cpu')
    edited_img_coords = edited_img_coords[edited_fg_mask].to(device='cpu')
    
    correspondences = pack_correspondences(
        edited_src_img_coords[:, 0],
        edited_src_img_coords[:, 1],
        edited_img_coords[:, 0],
        edited_img_coords[:, 1])

    # TODO: do this conversion from depth to disparity and the depth normalization later on (normalization probably in the diffuser)
    if use_input_depth_normalization:
        _, depth_bounds = normalize_depth(1.0/depth, return_bounds=True)
    else:
        depth_bounds = None
    edited_disparity = normalize_depth(1.0/edited_depth, bounds=depth_bounds)

    return edited_disparity, correspondences


def transform_depth_pc(
        depth: torch.Tensor, bg_depth: torch.Tensor, fg_mask: torch.Tensor, intrinsics: torch.Tensor,
        rot_angle: float = None, rot_axis: torch.Tensor = None, translation: torch.Tensor = None,
        use_input_depth_normalization = False):

    if not fg_mask.any():
        # foreground mask is empty, there is no foreground object
        # return the image depth and empty correspondences
        if use_input_depth_normalization:
            _, depth_bounds = normalize_depth(1.0/depth, return_bounds=True)
        else:
            depth_bounds = None
        correspondences = pack_correspondences(
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
        )
        return normalize_depth(1.0/depth, bounds=depth_bounds), correspondences
    
    # default transformation parameters
    if rot_angle is None:
        rot_angle = 0.0
    if rot_axis is None:
        rot_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=depth.device)
    if translation is None:
        translation = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=depth.device)

    bg_pts = depth_to_world_coords(bg_depth, intrinsics=intrinsics)
    pts = depth_to_world_coords(depth, intrinsics=intrinsics)
    
    device = fg_mask.device
    
    if fg_mask.shape[-2] != fg_mask.shape[-1]:
        raise RuntimeError(f'Expected fg_mask to be square, got shape {fg_mask.shape[-2]} x {fg_mask.shape[-1]}.')
    img_res = fg_mask.shape[-1]
    
    # TODO: change to transform_points and make sure results are exactly the same,
    # then delete transform_point_cloud (which is not used anywhere else)
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
    num_bg_pts = reshaped_bg_pts.shape[0]

    reshaped_pts = pts.reshape((img_res**2, 3))

    new_mod_ids = np.zeros(len(reshaped_bg_pts) + len(reshaped_pts[mod_ids]), dtype = np.uint8)

    new_mod_ids[np.arange(new_mod_ids.size) > len(reshaped_bg_pts) - 1] = 1

    modded_id_list = np.where(mod_ids)[0]

    # idx_to_coord = {}
    # for idx in modded_id_list:
    #     pt = reshaped_pts[idx]
    #     reshaped_bg_pts = np.vstack((reshaped_bg_pts, pt))
    #     idx_to_coord[len(reshaped_bg_pts) - 1] = divmod(idx, img_res)

    modded_coords = np.stack([modded_id_list // img_res, modded_id_list % img_res], axis=-1)
    idx_to_coord = {i+num_bg_pts: (modded_coords[i,0].item(), modded_coords[i,1].item()) for i in range(modded_coords.shape[0])}
    reshaped_bg_pts = np.vstack([reshaped_bg_pts, reshaped_pts[modded_id_list]])

    (rendered_depth, target_mask, transformed_positions_x, transformed_positions_y, orig_visibility_mask) = points_to_depth(
        points=torch.from_numpy(reshaped_bg_pts).to(device=device),
        intrinsics=intrinsics,
        output_size=(img_res, img_res),
        point_mask=torch.from_numpy(new_mod_ids).to(device=device),
    )

    # without conversion to disparty
    direct_depth = rendered_depth

    # get normalized disparity
    # TODO: depth should be converted to disparity and normalized in the diffuser, not here
    #       (since requiring disparity and the normalization is specific to the depth-to-image diffuser)
    if use_input_depth_normalization:
        _, depth_bounds = normalize_depth(1.0/depth, return_bounds=True)
    else:
        depth_bounds = None
    rendered_depth = normalize_depth(1.0/rendered_depth, bounds=depth_bounds)
    
    rendered_depth = rendered_depth[0, 0, ...].cpu().numpy()

    #plot_img(rendered_depth)

    infer_visible_original = np.zeros_like(mod_ids.reshape((img_res, img_res)), dtype = np.uint8)

    original_idxs = [idx_to_coord[key] for key in np.where(orig_visibility_mask)[0]]

    for idx in original_idxs:
        infer_visible_original[idx] = 1

    original_positions_y, original_positions_x = np.where(infer_visible_original)

    target_mask_uint8 = target_mask.astype(np.uint8)*255

    # Define a kernel for the closing operation (you can adjust the size and shape)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (img_res // 250 , img_res // 250))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (img_res // 50 , img_res // 50))

    target_mask_cleaned = target_mask_uint8

    # Perform the closing operation
    target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_CLOSE, kernel)

    # Perform the closing operation
    target_mask_cleaned = cv2.morphologyEx(target_mask_cleaned, cv2.MORPH_OPEN, open_kernel)

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

    # correspondences = np.stack((original_positions_x, original_positions_y, transformed_positions_x, transformed_positions_y), axis=-1)
    correspondences = pack_correspondences(
        torch.from_numpy(original_positions_x),
        torch.from_numpy(original_positions_y),
        torch.from_numpy(transformed_positions_x),
        torch.from_numpy(transformed_positions_y))


    noise_mask = target_mask_uint8.astype(int) - target_mask_cleaned.astype(int)

    final_mask = target_mask_cleaned.astype(int) - target_mask_uint8.astype(int)
    final_mask[final_mask < 0] = 0
    noise_mask[noise_mask < 0] = 0

    inpaint_mask = final_mask + noise_mask #+ occluded_mask
    inpaint_mask = (inpaint_mask > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    inpaint_mask_dilated = cv2.dilate(inpaint_mask, kernel)

    lap_inpainted_depth_map = poisson_solve(np.array(rendered_depth), inpaint_mask_dilated)

    lap_inpainted_depth_map = torch.from_numpy(lap_inpainted_depth_map).to(device=device, dtype=torch.float32)[None, None]
    target_mask_cleaned = torch.from_numpy(target_mask_cleaned.astype(np.float32) / 255.0).to(device=device)[None, None]

    return lap_inpainted_depth_map, correspondences

# def transform_point_cloud2(points, axis, angle_degrees, x, y, z):
#     """
#     Rotate point cloud around the centroid of points selected by the mask.
    
#     Parameters:
#     - points: numpy array of shape (512, 512, 3)
#     - axis: rotation axis, numpy array of shape (3,)
#     - angle_degrees: rotation angle in degrees
#     - mask: boolean array of shape (512, 512) indicating which pixels to consider for the centroid
    
#     Returns:
#     - rotated_points: numpy array of shape (512, 512, 3)
#     """
#     # #cut_img = Image.open('car-cut.png')
#     # # cut_img = mask
#     # #cut_img = Image.open('cup-table-cut (2).png')    
#     # img_tensor = np.array(mask)
#     # ref_mask = (img_tensor[:, :] > 0.5)
#     # mask = np.zeros_like(ref_mask, dtype = np.uint8) 
#     # mask[ref_mask.nonzero()] = 255
#     # # mask = max_pool_numpy(mask, 512 // img_dim) # doesn't do anything as img_dim=512

#     # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Adjust the size as needed
#     # #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     # #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # Adjust the size as needed    
#     # #mask = cv2.erode(mask, kernel)
#     # #mask = cv2.dilate(mask, kernel)
    
#     # #visualize_img(mask, 'curr_mask')
#     # mask = (mask[:,:] != 0)

#     # mask = mask.astype(bool)

#     # modified_indices = mask.flatten()  # Flattened version of the mask to match the reshaped points
    
#     # Convert angle from degrees to radians
#     angle = np.radians(angle_degrees)
    
#     # Ensure axis is a unit vector
#     axis = axis / np.linalg.norm(axis)
    
#     # Compute the centroid of the masked points
#     masked_points = points

#     # trimesh_pc = trimesh.points.PointCloud(vertices=masked_points)
#     # trimesh_pc.export("point_cloud_obj.glb")


#     centroid = np.mean(masked_points, axis=0)
    
#     # Translate points to place centroid at the origin
#     translated_points = points - centroid
    
#     # Flatten the translated points
#     flattened_points = translated_points.reshape(-1, 3)
    
#     # Use the Rodriguez rotation formula
#     cos_theta = np.cos(angle)
#     sin_theta = np.sin(angle)
    
#     term1 = flattened_points * cos_theta
#     term2 = np.cross(axis, flattened_points) * sin_theta
#     term3 = axis * np.dot(flattened_points, axis)[:, np.newaxis] * (1 - cos_theta)
    
#     rotated_points_flattened = term1 + term2 + term3

#     # trimesh_pc = trimesh.points.PointCloud(vertices=rotated_points_flattened + centroid + np.array([x,y,z]))
#     # trimesh_pc.export("modified_point_cloud_obj.glb")
    
#     # Reshape the points back to 512x512x3 and translate back to the original position
#     rotated_points = rotated_points_flattened + centroid + np.array([x, y, z])#+ np.array([ 0.0, 0.0, -0.175]) #+ np.array([ 0.5, 0.15, 1.0])#+ np.array([0, 0, -0.1])#+ np.array([-0.035, 0.01, 0.15])
    
#     return rotated_points

def transform_points(
        points: torch.Tensor, rot_angle: torch.Tensor = None, rot_axis: torch.Tensor = None, translation: torch.Tensor = None):
    
    # move centroid to origin to rotate about centroid
    centroid = points.mean(dim=0, keepdim=True)
    points = points - centroid
    
    # Use Rodriguez rotation formula to rotate with axis and angle
    rot_axis = rot_axis / torch.linalg.norm(rot_axis, ord=2)
    rot_angle = rot_angle * (torch.pi / 180.0)
    cos_theta = torch.cos(rot_angle)
    sin_theta = torch.sin(rot_angle)
    term1 = points * cos_theta
    term2 = torch.cross(rot_axis[None, ...], points) * sin_theta
    term3 = rot_axis * torch.sum(points * rot_axis[None, ...], dim=-1, keepdim=True) * (1 - cos_theta)
    points = term1 + term2 + term3

    # move centroid back to original position and add translation
    points = points + centroid + translation[None, ...]

    return points

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

def depth_to_world_coords(depth: torch.Tensor, intrinsics: torch.Tensor, extrinsics_R: torch.Tensor = None, extrinsics_t: torch.Tensor = None):

    if depth.shape[0] != 1:
        raise ValueError("Only batch size 1 is supported")

    depth = depth.squeeze(dim=0)
    intrinsics_inv = torch.linalg.inv(intrinsics)
    if extrinsics_R is None:
        extrinsics_R = torch.eye(3, device=intrinsics.device, dtype=intrinsics.dtype)
    else:
        extrinsics_R = extrinsics_R
    if extrinsics_t is None:
        extrinsics_t = torch.zeros(3, device=intrinsics.device, dtype=intrinsics.dtype)
    else:
        extrinsics_t = extrinsics_t

    # M converts from your coordinate to PyTorch3D's coordinate system
    M = torch.eye(3, device=intrinsics.device, dtype=intrinsics.dtype)
    M[0, 0] = -1.0
    M[1, 1] = -1.0

    height, width = depth.shape[1:3]

    if height < 2 or width < 2:
        raise RuntimeError(f'Expected depth to have at least 2 pixels in each dimension, got {height} x {width}.')
    
    # print(height)
    # print(width)

    # normalize so image coordinates are in [-1, 1]^2
    # since this coordinate range is mapped to inside the view frustum by the intrinsic matrix
    # (assuming corner pixel centers are at located at corners or edges of the image plane)
    normalized_width = (width-1)  / (max(width, height)-1)
    normalized_height = (height-1)  / (max(width, height)-1)
    x = torch.linspace(
        -normalized_width, normalized_width,
        steps=width, device=intrinsics.device, dtype=intrinsics.dtype)
    y = torch.linspace(
        -normalized_height, normalized_height,
        steps=height, device=intrinsics.device, dtype=intrinsics.dtype)
    coord = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1)
    coord = torch.cat((coord, torch.ones_like(coord)[:, :, [0]]), dim=-1)  # z=1
    # coord = torch.as_tensor(coord, dtype=torch.float32, device=device)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    points = D * intrinsics_inv[None, None, None, ...] @ coord[:, :, :, :, None]
    # pts3D_1 live in your coordinate system. Convert them to Py3D's
    points = M[None, None, None, ...] @ points
    # camera to world coordinates (world to cam is (R @ p) + t, this is the inverse of that)
    points = extrinsics_R[None, None, None, ...].transpose(-2, -1) @ (points - extrinsics_t[None, None, None, :, None])

    return points[:, :, :, :3, 0][0]

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

    # projection to image plane
    projected = (intrinsics @ points.T).T
    u = projected[:, 0] / projected[:, 2]
    v = projected[:, 1] / projected[:, 2]

    # image plane coordinates [-1, 1]^2 -> [0, max(output_size)-1]^2
    # (assuming the fov is for the larger image dimension)
    u = (u*0.5+0.5) * (max(output_size)-1)
    v = (v*0.5+0.5) * (max(output_size)-1)

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
