import torch 
import numpy as np
import math
import trimesh
import matplotlib
import matplotlib.cm
import cv2
from PIL import Image
from scipy.ndimage import convolve
from scipy.sparse import lil_matrix
from scipy.sparse import linalg
from scipy.ndimage import binary_dilation

# repo = "isl-org/ZoeDepth"
# model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)

def laplacian(image):
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return convolve(image, laplacian_kernel, mode='constant')

def solve_laplacian_depth(fg_depth, bg_depth, mask):
    # Compute the number of unknown pixels
    unknown_pixels = np.where(mask)
    num_unknowns = len(unknown_pixels[0])

    # Generate an index map for the unknown pixels
    index_map = -np.ones_like(fg_depth, dtype=int)
    index_map[unknown_pixels] = np.arange(num_unknowns)

    # Compute the Laplacian of the bg_depth
    lap_bg = laplacian(bg_depth)

    # Generate the system matrix
    A = lil_matrix((num_unknowns, num_unknowns))
    b = np.zeros(num_unknowns)

    for index, (y, x) in enumerate(zip(*unknown_pixels)):
        A[index, index] = 4

        if y > 0:
            if mask[y-1, x]:  
                A[index, index_map[y-1, x]] = -1
            else:
                b[index] += fg_depth[y-1, x]

        if y < fg_depth.shape[0] - 1:
            if mask[y+1, x]:  
                A[index, index_map[y+1, x]] = -1
            else:
                b[index] += fg_depth[y+1, x]

        if x > 0:
            if mask[y, x-1]:  
                A[index, index_map[y, x-1]] = -1
            else:
                b[index] += fg_depth[y, x-1]

        if x < fg_depth.shape[1] - 1:
            if mask[y, x+1]:  
                A[index, index_map[y, x+1]] = -1
            else:
                b[index] += fg_depth[y, x+1]

        # Incorporate the bg_depth Laplacian into the right-hand side
        b[index] -= lap_bg[y, x]

    # Solve the linear system
    solution = linalg.spsolve(A.tocsr(), b)

    # Generate the output depth map
    output_depth = fg_depth.copy()
    output_depth[unknown_pixels] = solution

    return output_depth

def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img

def get_intrinsics(H,W):
    """
    Intrinsics for a pinhole camera model.
    Assume fov of 55 degrees and central principal point.
    """
    f = 0.5 * W / np.tan(0.5 * 6.24 * np.pi / 180.0) #car benchmark
    #f = 0.5 * W / np.tan(0.5 * 7.18 * np.pi / 180.0) #airplane benchmark
    #f = 0.5 * W / np.tan(0.5 * 14.9 * np.pi / 180.0) #chair, cup, lamp, stool benchmark        
    #f = 0.5 * W / np.tan(0.5 * 7.23 * np.pi / 180.0) #plant benchmark            
    f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)    
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]])

def depth_to_points(depth: torch.Tensor, R=None, t=None):

    if depth.shape[0] != 1:
        raise ValueError("Only batch size 1 is supported")

    depth = depth.squeeze().cpu().numpy()

    K = get_intrinsics(depth.shape[1], depth.shape[2])
    Kinv = np.linalg.inv(K)
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    # M converts from your coordinate to PyTorch3D's coordinate system
    M = np.eye(3)
    M[0, 0] = -1.0
    M[1, 1] = -1.0

    height, width = depth.shape[1:3]

    print(height)
    print(width)

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    # coord = torch.as_tensor(coord, dtype=torch.float32, device=device)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    # print(D.shape, Kinv[None, None, None, ...].shape, coord[:, :, :, :, None].shape )
    pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    # pts3D_1 live in your coordinate system. Convert them to Py3D's
    pts3D_1 = M[None, None, None, ...] @ pts3D_1
    # from reference to targe tviewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    # pts3D_2 = pts3D_1
    # depth_2 = pts3D_2[:, :, :, 2, :]  # b,1,h,w
    return torch.from_numpy(pts3D_2[:, :, :, :3, 0][0])

def zoe_points_to_depth_merged(points, mod_ids, output_size=(512, 512), R=None, t=None, max_depth_value=float('inf')):
    K = get_intrinsics(output_size[1], output_size[0])
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    # Coordinate transformations
    points = points[..., np.newaxis]
    points = np.linalg.inv(R) @ (points.T - t[:, None]).T
    points = points[:, :, 0]
    M_inv = np.eye(3)
    M_inv[0, 0] = -1.0
    M_inv[1, 1] = -1.0
    points = (M_inv @ points.T).T

    # Projection to Image Plane
    projected = (K @ points.T).T
    u = projected[:, 0] / projected[:, 2]
    v = projected[:, 1] / projected[:, 2]

    u = np.around(np.clip(u, 0, output_size[1] - 1)).astype(int)
    v = np.around(np.clip(v, 0, output_size[0] - 1)).astype(int)

    depth_map = np.full(output_size, np.inf)
    dist_to_cam = np.full(output_size, np.inf)
    target_mask = np.full(output_size, False)
    modified_depth_mask = np.full(output_size, False)
    original_visibility_mask = np.full_like(mod_ids, False)
    depth_set_by = np.full(output_size, -1, dtype=np.int64)


    for i in range(points.shape[0]):
        if points[i, 2] < depth_map[v[i], u[i]]:
            depth_map[v[i], u[i]] = points[i, 2]
            dist_to_cam[v[i], u[i]] = (points[i,0] ** 2 + points[i, 1]**2 + points[i, 2]**2)**0.5
            if mod_ids[i]:
                original_visibility_mask[i] = True
                if depth_set_by[v[i], u[i]] >= 0:
                    original_visibility_mask[depth_set_by[v[i], u[i]]] = False
                target_mask[v[i], u[i]] = True
                modified_depth_mask[v[i], u[i]] = True
                depth_set_by[v[i], u[i]] = i
            elif modified_depth_mask[v[i], u[i]]:
                target_mask[v[i], u[i]] = False
                if depth_set_by[v[i], u[i]] >= 0:
                    original_visibility_mask[depth_set_by[v[i], u[i]]] = False
                depth_set_by[v[i], u[i]] = i

 

    mask_no_points = depth_map == max_depth_value
    pixels_no_points = np.column_stack(np.where(mask_no_points))

    far_inv_depth = 0.03 # inverse depth at far plane (empiricaly ~ similar to MiDaS depth ranges)
    near_inv_depth = 100.0 # inverse depth at near plane (empiricaly ~ similar to MiDaS depth ranges)

    #smoothed_dist_to_cam = cv2.medianBlur(dist_to_cam.astype(np.float32), ksize=3)
    #dist_to_cam[dist_to_cam == np.inf] = smoothed_dist_to_cam[dist_to_cam == np.inf]

    near_depth = dist_to_cam.min()
    far_depth = dist_to_cam.max()

    dist_to_cam = 1.0 / dist_to_cam
    dist_to_cam = (dist_to_cam - 1/far_depth) / ((1/near_depth) - (1/far_depth))
    dist_to_cam = dist_to_cam * (near_inv_depth - far_inv_depth) + far_inv_depth

    #dist_to_cam = cv2.medianBlur(dist_to_cam.astype(np.float32), ksize = 1)
    depth_map = 1.0 / depth_map
 
    #depth_map[depth_map == np.inf] = 1e-8
    #depth_map[depth_map == 0] = 1e-8
 
    #smoothed_depth_map = cv2.medianBlur(depth_map.astype(np.float32), ksize=3)
    #depth_map[depth_map < 1e-8] = smoothed_depth_map[depth_map < 1e-8]
    #smoothed_depth_map = depth_map #cv2.medianBlur(depth_map.astype(np.float32), ksize=1)
    max_depth = depth_map.max()
    min_depth = depth_map.min()
    depth_map_normalized = 255 * ((depth_map - min_depth) / (max_depth - min_depth) )
    depth_image = Image.fromarray(depth_map_normalized.astype(np.uint8))

    original_visibility_mask = original_visibility_mask.astype(bool)

    return depth_map_normalized, pixels_no_points, target_mask, u[original_visibility_mask], v[original_visibility_mask], original_visibility_mask


#def align_depths(bg_depth, fg_depth, mask):
    #find the corresponding points 
    #then


def zoe_points_to_depth_merged_z(points, mod_ids, output_size=(512, 512), R=None, t=None, max_depth_value=float('inf')):
    K = get_intrinsics(output_size[1], output_size[0])
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    # Coordinate transformations
    points = points[..., np.newaxis]
    points = np.linalg.inv(R) @ (points.T - t[:, None]).T
    points = points[:, :, 0]
    M_inv = np.eye(3)
    M_inv[0, 0] = -1.0
    M_inv[1, 1] = -1.0
    points = (M_inv @ points.T).T

    # Projection to Image Plane
    projected = (K @ points.T).T
    u = projected[:, 0] / projected[:, 2]
    v = projected[:, 1] / projected[:, 2]

    u = np.around(np.clip(u, 0, output_size[1] - 1)).astype(int)
    v = np.around(np.clip(v, 0, output_size[0] - 1)).astype(int)

    depth_map = np.full(output_size, np.inf)
    dist_to_cam = np.full(output_size, np.inf)
    target_mask = np.full(output_size, False)
    modified_depth_mask = np.full(output_size, False)
    original_visibility_mask = np.full_like(mod_ids, False)
    depth_set_by = np.full(output_size, -1, dtype=np.int64)


    for i in range(points.shape[0]):
        if points[i, 2] < depth_map[v[i], u[i]]:
            depth_map[v[i], u[i]] = points[i, 2]
            dist_to_cam[v[i], u[i]] = (points[i,0] ** 2 + points[i, 1]**2 + points[i, 2]**2)**0.5
            if mod_ids[i]:
                original_visibility_mask[i] = True
                if depth_set_by[v[i], u[i]] >= 0:
                    original_visibility_mask[depth_set_by[v[i], u[i]]] = False
                target_mask[v[i], u[i]] = True
                modified_depth_mask[v[i], u[i]] = True
                depth_set_by[v[i], u[i]] = i
            elif modified_depth_mask[v[i], u[i]]:
                target_mask[v[i], u[i]] = False
                if depth_set_by[v[i], u[i]] >= 0:
                    original_visibility_mask[depth_set_by[v[i], u[i]]] = False
                depth_set_by[v[i], u[i]] = i

 

    mask_no_points = depth_map == max_depth_value
    pixels_no_points = np.column_stack(np.where(mask_no_points))

    far_inv_depth = 0.03 # inverse depth at far plane (empiricaly ~ similar to MiDaS depth ranges)
    near_inv_depth = 100.0 # inverse depth at near plane (empiricaly ~ similar to MiDaS depth ranges)

    #smoothed_dist_to_cam = cv2.medianBlur(dist_to_cam.astype(np.float32), ksize=3)
    #dist_to_cam[dist_to_cam == np.inf] = smoothed_dist_to_cam[dist_to_cam == np.inf]

    near_depth = dist_to_cam.min()
    far_depth = dist_to_cam.max()

    near_depth = depth_map.min()
    far_depth = depth_map.max()


    depth_map = 1.0 / depth_map
    depth_map = (depth_map - 1/far_depth) / ((1/near_depth) - (1/far_depth))
    depth_map = depth_map * (near_inv_depth - far_inv_depth) + far_inv_depth

    #dist_to_cam = cv2.medianBlur(dist_to_cam.astype(np.float32), ksize = 1)
    #depth_map = 1.0 / depth_map
 
    #depth_map[depth_map == np.inf] = 1e-8
    #depth_map[depth_map == 0] = 1e-8
 
    #smoothed_depth_map = cv2.medianBlur(depth_map.astype(np.float32), ksize=3)
    #depth_map[depth_map < 1e-8] = smoothed_depth_map[depth_map < 1e-8]
    #smoothed_depth_map = depth_map #cv2.medianBlur(depth_map.astype(np.float32), ksize=1)
    max_depth = depth_map.max()
    min_depth = depth_map.min()
    depth_map_normalized = 255 * ((depth_map - min_depth) / (max_depth - min_depth) )
    depth_image = Image.fromarray(depth_map_normalized.astype(np.uint8))

    original_visibility_mask = original_visibility_mask.astype(bool)

    return depth_map_normalized, pixels_no_points, target_mask, u[original_visibility_mask], v[original_visibility_mask], original_visibility_mask



def zoe_points_to_depth_alt(points, width, height, R=None, t=None):
    K = get_intrinsics(width, height)
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    # M converts from PyTorch3D's coordinate system to your coordinate
    M_inv = np.eye(3)
    M_inv[0, 0] = -1.0
    M_inv[1, 1] = -1.0

    # Transform points from target viewpoint back to reference viewpoint
    points = points[..., np.newaxis]
    points = np.linalg.inv(R) @ (points.T - t[:, None]).T
    points = points[:, :, 0] 

    # Convert back to your coordinate system from Py3D's
    points = (M_inv @ points.T).T

    # Project 3D points onto 2D image plane
    projected = (K @ points.T).T
    u = projected[:, 0] / projected[:, 2]
    v = projected[:, 1] / projected[:, 2]

    # Convert coordinates to integers for indexing
    u = np.clip(u, 0, width - 1).astype(int)
    v = np.clip(v, 0, height - 1).astype(int)

    depth_map = np.full((height, width), np.inf)
    for i in range(points.shape[0]):
        depth_map[v[i], u[i]] = min(depth_map[v[i], u[i]], points[i, 2])

    # Create an inpainting mask
    mask = (depth_map == np.inf).astype(np.uint8) * 255

    # Replace np.inf with a dummy value (like 0) for inpainting
    depth_map[depth_map == np.inf] = 0
    
    # Use OpenCV to inpaint missing values
    #inpainted_depth_map = cv2.inpaint(depth_map.astype(np.float32), mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

  
    smoothed_depth_map = cv2.medianBlur(depth_map.astype(np.float32), ksize=5)


    # Scale the depth map values to 0-255 range
    max_depth = smoothed_depth_map.max()
    min_depth = smoothed_depth_map.min()
    depth_map_normalized = 255 * (smoothed_depth_map - min_depth) / (max_depth - min_depth)


    depth_image = Image.fromarray(depth_map_normalized.astype(np.uint8))

    return depth_image


def zoe_points_to_depth(points, width, height, R=None, t=None):
    # Ensure points are in N,3 format
    assert points.ndim == 2 and points.shape[1] == 3
    
    K = get_intrinsics(width, height)
    
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)
    
    # M converts from PyTorch3D's coordinate system to your coordinate
    M_inv = np.eye(3)
    M_inv[0, 0] = -1.0
    M_inv[1, 1] = -1.0

    # Transform points from target viewpoint back to reference viewpoint
    points_transformed = np.linalg.inv(R) @ (points.T - t[:, None])
    points_transformed = M_inv @ points_transformed  # Convert back to your coordinate system from Py3D's
    
    # Project 3D points onto 2D image plane
    projected = K @ points_transformed[:3, :]
    u = (projected[0, :] / projected[2, :]).astype(int)
    v = (projected[1, :] / projected[2, :]).astype(int)
    
    # Clip coordinates to be within the image boundaries
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)
    
    depth_map = np.full((height, width), np.inf)  # Initialize with "infinite" depth
    for idx, (x, y) in enumerate(zip(u, v)):
        depth_map[y, x] = min(depth_map[y, x], points_transformed[2, idx])

    # Normalize between 0 and 255 for PIL image
    max_depth = depth_map[depth_map != np.inf].max()  # Excluding infinite values
    min_depth = depth_map[depth_map != np.inf].min()
    depth_map = 255 * (depth_map - min_depth) / (max_depth - min_depth)
    depth_map[depth_map == np.inf] = 0  # Set the background (infinite values) to 0
    
    # Convert to PIL image
    pil_image = Image.fromarray(depth_map.astype(np.uint8))
    
    return pil_image

def depth_edges_mask(depth):
    """Returns a mask of edges in the depth map.
    Args:
    depth: 2D numpy array of shape (H, W) with dtype float32.
    Returns:
    mask: 2D numpy array of shape (H, W) with dtype bool.
    """
    # Compute the x and y gradients of the depth map.
    depth_dx, depth_dy = np.gradient(depth)
    # Compute the gradient magnitude.
    depth_grad = np.sqrt(depth_dx ** 2 + depth_dy ** 2)
    # Compute the edge mask.
    mask = depth_grad > 0.05
    return mask


# def predict_depth(image, model_zoe_nk):
#     return Image.fromarray(model_zoe_nk.infer_pil(image))

def create_triangles(h, w, mask=None):
    """
    Reference: https://github.com/google-research/google-research/blob/e96197de06613f1b027d20328e06d69829fa5a89/infinite_nature/render_utils.py#L68
    Creates mesh triangle indices from a given pixel grid size.
        This function is not and need not be differentiable as triangle indices are
        fixed.
    Args:
    h: (int) denoting the height of the image.
    w: (int) denoting the width of the image.
    Returns:
    triangles: 2D numpy array of indices (int) with shape (2(W-1)(H-1) x 3)
    """
    x, y = np.meshgrid(range(w - 1), range(h - 1))
    tl = y * w + x
    tr = y * w + x + 1
    bl = (y + 1) * w + x
    br = (y + 1) * w + x + 1
    triangles = np.array([tl, bl, tr, br, tr, bl])
    triangles = np.transpose(triangles, (1, 2, 0)).reshape(
        ((w - 1) * (h - 1) * 2, 3))
    if mask is not None:
        mask = mask.reshape(-1)
        triangles = triangles[mask[triangles].all(1)]
    return triangles

# def get_mesh(model, image, keep_edges=False):
#     image.thumbnail((1024,1024))  # limit the size of the input image
#     depth = predict_depth(model, image)
#     pts3d = depth_to_points(depth[None])
#     pts3d = pts3d.reshape(-1, 3)

#     # Create a trimesh mesh from the points
#     # Each pixel is connected to its 4 neighbors
#     # colors are the RGB values of the image

#     verts = pts3d.reshape(-1, 3)
#     image = np.array(image)
#     if keep_edges:
#         triangles = create_triangles(image.shape[0], image.shape[1])
#     else:
#         triangles = create_triangles(image.shape[0], image.shape[1], mask=~depth_edges_mask(depth))
#     colors = image.reshape(-1, 3)
#     mesh = trimesh.Trimesh(vertices=verts, faces=triangles, vertex_colors=colors)

#     # Save as glb
#     #glb_file = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
#     #glb_path = glb_file.name
#     #mesh.export(glb_path)
#     return mesh

def get_points(image, model_zoe_nk):
    depth = model_zoe_nk.infer_pil(image)
    pts3d = depth_to_points(depth[None])
    #pts3d = pts3d.reshape(-1, 3)
    return pts3d

def get_aligned_pts(img, bg_img, mask, model_zoe_nk):
    fg_depth = np.array(Image.fromarray(model_zoe_nk.infer_pil(img)))
    bg_depth = np.array(Image.fromarray(model_zoe_nk.infer_pil(bg_img)))
    mask = np.array(mask)
    mask = binary_dilation(mask, iterations=15)
    new_depth = solve_laplacian_depth(fg_depth, bg_depth, mask)
    fg_pts = depth_to_points(fg_depth[None])
    bg_pts = depth_to_points(new_depth[None])
    return fg_pts, bg_pts


def get_aligned_pts_true_depth(fg_depth, bg_img, mask, model_zoe_nk):
    #fg_depth = np.array(predict_depth(img))
    bg_depth = np.array(Image.fromarray(model_zoe_nk.infer_pil(bg_img)))
    mask = np.array(mask)
    mask = binary_dilation(mask, iterations=15)
    new_depth = solve_laplacian_depth(fg_depth, bg_depth, mask)
    fg_pts = depth_to_points(fg_depth[None])
    bg_pts = depth_to_points(new_depth[None])
    return fg_pts, bg_pts

def get_aligned_pts_syn_depth(fg_depth, bg_depth):
    fg_pts = depth_to_points(fg_depth[None])
    bg_pts = depth_to_points(bg_depth[None])
    return fg_pts, bg_pts