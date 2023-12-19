import torch
import numpy as np
from PIL import Image

from diffhandles.depth_estimator import DepthEstimator

class ZoeDepthEstimator(DepthEstimator):

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)

    def to(self, device: torch.device):
        self.model = self.model.to(device)
    
    def estimate_depth(self, img: torch.Tensor):
        return self.model.infer(img)

    def get_intrinsics(self, h: int, w: int):
        """
        Intrinsics for a pinhole camera model.
        Assume fov of 55 degrees and central principal point.
        """
        f = 0.5 * w / np.tan(0.5 * 6.24 * np.pi / 180.0) #car benchmark
        #f = 0.5 * W / np.tan(0.5 * 7.18 * np.pi / 180.0) #airplane benchmark
        #f = 0.5 * W / np.tan(0.5 * 14.9 * np.pi / 180.0) #chair, cup, lamp, stool benchmark        
        #f = 0.5 * W / np.tan(0.5 * 7.23 * np.pi / 180.0) #plant benchmark            
        f = 0.5 * w / np.tan(0.5 * 55 * np.pi / 180.0)    
        cx = 0.5 * w
        cy = 0.5 * h
        return np.array([[f, 0, cx],
                        [0, f, cy],
                        [0, 0, 1]])

    def depth_to_points(self, depth: torch.Tensor, R=None, t=None):

        if depth.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported")

        depth = depth.squeeze().cpu().numpy()

        K = self.get_intrinsics(depth.shape[1], depth.shape[2])
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

    def points_to_depth_merged(self, points, mod_ids, output_size=(512, 512), R=None, t=None, max_depth_value=float('inf')):
        K = self.get_intrinsics(output_size[1], output_size[0])
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