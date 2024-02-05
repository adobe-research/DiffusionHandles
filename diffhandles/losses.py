import torch
import cv2

def compute_localized_transformed_appearance_loss(
        activations, activations_orig, processed_correspondences,
        attn_layer_low, attn_layer_high, patch_size, activations_size):
    
    loss = 0
    for i in range(attn_layer_low,attn_layer_high):
        # attn_map = attn_maps[i]
        # attn_map = attn_map.detach()
                        
        activations_orig_map = torch.nn.functional.interpolate(activations_orig.view(1, activations_orig.shape[0], activations_orig.shape[1], activations_orig.shape[2]), activations_size, mode = 'bilinear')
        activations_map = torch.nn.functional.interpolate(activations.view(1, activations.shape[0], activations.shape[1], activations.shape[2]), activations_size, mode = 'bilinear')
        pixel_loss, patch_loss, mean_loss = compute_gathered_loss_alt(activations_orig_map[0], activations_map[0], processed_correspondences, patch_size)
        
        loss += 1.0*patch_loss

    loss /= (attn_layer_high - attn_layer_low) #len(attn_maps_orig)
    return loss

def compute_background_loss(
        activations, activations_orig, processed_correspondences,
        attn_layer_low, attn_layer_high, activations_size):
    
    original_x = processed_correspondences['original_x']
    original_y = processed_correspondences['original_y']
    remaining_x = processed_correspondences['remaining_x']
    remaining_y = processed_correspondences['remaining_y']
    remaining_x_orig = processed_correspondences['remaining_x_orig']
    remaining_y_orig = processed_correspondences['remaining_y_orig']
    remaining_x_trans = processed_correspondences['remaining_x_trans']
    remaining_y_trans = processed_correspondences['remaining_y_trans']

    loss = 0
    for i in range(attn_layer_low,attn_layer_high):
        # attn_map = attn_maps[i]
        # # visualize_img(attn_map_orig.detach().cpu().numpy(), 'attn_orig_' + str(i))
        # # visualize_img(attn_map.detach().cpu().numpy(), 'attn_' + str(i))        
        # attn_map = attn_map.detach()

        mask = torch.zeros(activations_size, device=activations.device, dtype=torch.float32)
        mask_orig = torch.zeros(activations_size, device=activations.device, dtype=torch.float32)
        mask_trans = torch.zeros(activations_size, device=activations.device, dtype=torch.float32)
        mask_full = torch.zeros(activations_size, device=activations.device, dtype=torch.float32)        
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

        activations_orig_map = torch.nn.functional.interpolate(activations_orig.view(1, activations_orig.shape[0], activations_orig.shape[1], activations_orig.shape[2]), activations_size, mode = 'bilinear')
        appearance_orig = torch.mul(mask, activations_orig_map[0]) / mask.sum()
        activations_map = torch.nn.functional.interpolate(activations.view(1, activations.shape[0], activations.shape[1], activations.shape[2]), activations_size, mode = 'bilinear')

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

def compute_gathered_loss_alt(
        act_orig, act, processed_correspondences, patch_size=1):
    
    original_x = processed_correspondences['original_x']
    original_y = processed_correspondences['original_y']
    transformed_x = processed_correspondences['transformed_x']
    transformed_y = processed_correspondences['transformed_y']

    mask = torch.zeros((act.shape[-2], act.shape[-1]), device=act.device, dtype=act.dtype)
    trans_mask = torch.zeros((act.shape[-2], act.shape[-1]), device=act.device, dtype=act.dtype)
    mask[original_y, original_x] = 1

    # Initialize the transformed attention map with zeros
    # trans_attn_map = torch.zeros_like(attn_map)
    # trans_attn_map[transformed_y, transformed_x] = attn_map[original_y, original_x]

    # attn_map = blur_attn_map(attn_map)
    # trans_attn_map = blur_attn_map(trans_attn_map)

    trans_mask[transformed_y, transformed_x] = mask[original_y, original_x]

    # trans_mask = trans_mask
    
    
    # visualize_img(attn_map.detach().cpu().numpy(), 'tricky_attn')
    # visualize_img(trans_attn_map.detach().cpu().numpy(), 'trans_tricky_attn')
    
    #attn_map = blur_attn_map(attn_map)

    # Weight the activations
    weighted_act_orig = torch.mul(mask, act_orig) / torch.sum(mask)
    weighted_act = torch.mul(trans_mask, act) / torch.sum(trans_mask)

    
    # Compute spatially weighted mean difference
    diff = torch.abs(weighted_act[:, transformed_y, transformed_x] - weighted_act_orig[:, original_y , original_x]) #/ torch.sum(attn_map)
    
    per_pixel_diff = diff.mean(dim=1) 
    per_pixel_diff = per_pixel_diff.mean()
    
    kernel_size = patch_size
    padding = kernel_size // 2  # Set padding to ensure output size equals input size

    # Create average pooling layer
    pooling = torch.nn.AvgPool2d(kernel_size, stride=1, padding=padding)
    
    weighted_act_orig = mask * act_orig
    weighted_act = trans_mask * act    


    # Apply average pooling and multiply by kernel size to get the sum
    weighted_sums = pooling(weighted_act.unsqueeze(0)) * (kernel_size * kernel_size)
    weighted_sums_orig = pooling(weighted_act_orig.unsqueeze(0)) * (kernel_size * kernel_size)

    # Apply average pooling and multiply by kernel size to get the sum for attention maps
    weight_sums = pooling(trans_mask.unsqueeze(0).unsqueeze(0)) * (kernel_size * kernel_size)
    weight_sums_orig = pooling(mask.unsqueeze(0).unsqueeze(0)) * (kernel_size * kernel_size)
    
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

    
    weighted_act_orig = torch.mul(mask, act_orig) / torch.sum(mask)
    weighted_act = torch.mul(trans_mask, act) / torch.sum(trans_mask)
    
    # Global mean difference
    global_weighted_act_orig = torch.sum(weighted_act_orig, dim=(1,2)) #/ attn_map.sum()
    global_weighted_act = torch.sum(weighted_act, dim=(1,2)) #/ trans_attn_map.sum()
    global_mean_diff = torch.abs(global_weighted_act_orig - global_weighted_act).mean()

    # Here you can return the individual loss components or a weighted sum of them.
    # I'll return the individual components for clarity.
    return per_pixel_diff, per_patch_diff, global_mean_diff   

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