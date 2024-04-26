import torch
import cv2

def compute_foreground_loss(
        activations, activations_orig, processed_correspondences,
        patch_size, activations_size):
    
    activations_orig_map = torch.nn.functional.interpolate(activations_orig[None, ...], activations_size, mode='bilinear')[0]
    activations_map = torch.nn.functional.interpolate(activations[None, ...], activations_size, mode='bilinear')[0]

    loss = local_average_feat_l1_loss(
        activations_orig_map, activations_map,
        processed_correspondences['original_x'], processed_correspondences['original_y'],
        processed_correspondences['transformed_x'], processed_correspondences['transformed_y'],
        patch_size)

    return loss

def compute_background_loss(
        activations, activations_orig, processed_correspondences,
        patch_size, activations_size, loss_type='global_avg'):

    activations_orig_map = torch.nn.functional.interpolate(activations_orig[None, ...], activations_size, mode='bilinear')[0]
    activations_map = torch.nn.functional.interpolate(activations[None, ...], activations_size, mode='bilinear')[0]

    if loss_type == 'global_avg':
        loss = average_feat_l1_loss(
            feat_map_1=activations_orig_map, feat_map_2=activations_map,
            x1=processed_correspondences['background_x_orig'], y1=processed_correspondences['background_y_orig'],
            x2=processed_correspondences['background_x_trans'], y2=processed_correspondences['background_y_trans'])
    elif loss_type == 'local_avg':
        loss = local_average_feat_l1_loss(
            feat_map_1=activations_orig_map, feat_map_2=activations_map,
            x1=processed_correspondences['background_x'], y1=processed_correspondences['background_y'],
            x2=processed_correspondences['background_x'], y2=processed_correspondences['background_y'],
            patch_size=patch_size)
    else:
        raise ValueError(f'Unknown background loss type: {loss_type}')

    return loss

def average_feat_l1_loss(
        feat_map_1, feat_map_2,
        x1, y1, x2, y2):
    
    feats1_avg = feat_map_1[..., y1, x1].mean(dim=-1)
    feats2_avg = feat_map_2[..., y2, x2].mean(dim=-1)

    return (feats1_avg - feats2_avg).abs().mean()

def local_average_feat_l1_loss(
        feat_map_1, feat_map_2,
        x1, y1, x2, y2,
        patch_size=1):

    # Create weight maps for the local averages (masks for the regions given in x1,y1 and x2,y2)
    weights_1 = torch.zeros((feat_map_1.shape[-2], feat_map_1.shape[-1]), device=feat_map_1.device, dtype=feat_map_1.dtype)
    weights_2 = torch.zeros((feat_map_2.shape[-2], feat_map_2.shape[-1]), device=feat_map_2.device, dtype=feat_map_2.dtype)
    weights_1[y1, x1] = 1
    weights_2[y2, x2] = 1
    
    # Create average pooling operator
    # Set padding to ensure output size equals input size
    pooling = torch.nn.AvgPool2d(patch_size, stride=1, padding=patch_size//2)

    # Apply average pooling and multiply by kernel size to get the sum
    feats1_local_avg = pooling(weights_1[None, None, ...] * feat_map_1[None, ...])
    feats2_local_avg = pooling(weights_2[None, None, ...] * feat_map_2[None, ...])

    # Apply average pooling and multiply by kernel size to get the sum for attention maps
    weight_avg_1 = pooling(weights_1[None, None, ...])
    weight_avg_2 = pooling(weights_2[None, None, ...])
    
    # Divide to get the weighted average
    EPS = 1e-10
    feats1_local_avg = feats1_local_avg / (weight_avg_1 + EPS)
    feats2_local_avg = feats2_local_avg / (weight_avg_2 + EPS)
    
    # Compute spatially weighted mean difference
    loss = (feats1_local_avg[0, :, y1, x1] - feats2_local_avg[0, :, y2, x2]).abs() # l1 distance
    loss = loss.mean(dim=-1) # average over pixels
    loss = loss.mean() # average over feature dimensions

    return loss
