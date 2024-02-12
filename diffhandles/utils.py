import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse
import scipy.sparse.linalg
import scipy.ndimage

def max_pool_numpy(mask, kernel_size):
    # Convert numpy mask to tensor
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)  # Convert to [1, 1, H, W] shape

    # Max pooling
    pooled_tensor = F.max_pool2d(mask_tensor, kernel_size)

    # Convert tensor back to numpy
    pooled_mask = pooled_tensor.squeeze().numpy()

    return pooled_mask

# def normalize_depth(depth_map: torch.Tensor):
#     # width, height = depth_map.size
#     # depth_map = np.asarray(depth_map)
#     # depth_map = torch.from_numpy(np.array(depth_map))
#     # depth_map = depth_map.to("cuda", torch.float32)
#     # depth_map = depth_map.view(1, depth_map.shape[0], depth_map.shape[1])
#     #print(depth_map.shape)
#     # depth_map = torch.nn.functional.interpolate(
#     #     depth_map,
#     #     size=(img_dim, img_dim),
#     #     mode="bicubic",
#     #     align_corners=False,
#     # )

#     # normalize to [-1, 1]
#     depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
#     depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
#     depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
#     # depth_map = (depth_map - depth_min) / (depth_max - depth_min)    
#     # depth_map = depth_map # .to(torch.float32)
#     # output = depth_map.cpu().numpy()[0][0]
#     #formatted = (output * 255 / np.max(output)).astype('uint8')
#     #image = Image.fromarray(formatted)
#     return depth_map

def laplacian(image):
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return scipy.ndimage.convolve(image, laplacian_kernel, mode='constant')

def solve_laplacian_depth(fg_depth: np.array, bg_depth, mask):
    # Compute the number of unknown pixels
    unknown_pixels = np.where(mask)
    num_unknowns = len(unknown_pixels[0])

    # Generate an index map for the unknown pixels
    index_map = -np.ones_like(fg_depth, dtype=int)
    index_map[unknown_pixels] = np.arange(num_unknowns)

    # Compute the Laplacian of the bg_depth
    lap_bg = laplacian(bg_depth)

    # Generate the system matrix
    A = scipy.sparse.lil_matrix((num_unknowns, num_unknowns))
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
    solution = scipy.sparse.linalg.spsolve(A.tocsr(), b)

    # Generate the output depth map
    output_depth = fg_depth.copy()
    output_depth[unknown_pixels] = solution

    return output_depth

def normalize_attn_torch(attn_map):
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    attn_map = 10*(attn_map - 0.5)
    attn_map = torch.sigmoid(attn_map)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    return attn_map

def pack_correspondences(original_x, original_y, transformed_x, transformed_y):
    correspondences = torch.stack((original_x, original_y, transformed_x, transformed_y), dim=-1)
    return correspondences

def unpack_correspondences(correspondences):
    original_x, original_y, transformed_x, transformed_y = torch.split(correspondences, 1, dim=-1)
    return original_x, original_y, transformed_x, transformed_y
