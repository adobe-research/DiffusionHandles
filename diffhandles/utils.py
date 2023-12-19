import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse
import scipy.sparse.linalg
import scipy.ndimage
# import scipy.ndimage

def max_pool_numpy(mask, kernel_size):
    # Convert numpy mask to tensor
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)  # Convert to [1, 1, H, W] shape

    # Max pooling
    pooled_tensor = F.max_pool2d(mask_tensor, kernel_size)

    # Convert tensor back to numpy
    pooled_mask = pooled_tensor.squeeze().numpy()

    return pooled_mask

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
    # mask = max_pool_numpy(mask, 512 // img_dim) # doesn't do anything as img_dim=512

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

def normalize_depth(depth_map: torch.Tensor):
    # width, height = depth_map.size
    # depth_map = np.asarray(depth_map)
    # depth_map = torch.from_numpy(np.array(depth_map))
    # depth_map = depth_map.to("cuda", torch.float32)
    # depth_map = depth_map.view(1, depth_map.shape[0], depth_map.shape[1])
    #print(depth_map.shape)
    # depth_map = torch.nn.functional.interpolate(
    #     depth_map,
    #     size=(img_dim, img_dim),
    #     mode="bicubic",
    #     align_corners=False,
    # )

    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    #depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)    
    depth_map = depth_map # .to(torch.float32)
    # output = depth_map.cpu().numpy()[0][0]
    #formatted = (output * 255 / np.max(output)).astype('uint8')
    #image = Image.fromarray(formatted)
    return depth_map

def laplacian(image):
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return scipy.ndimage.convolve(image, laplacian_kernel, mode='constant')

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
