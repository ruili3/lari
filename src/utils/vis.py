import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import os
import logging
from pathlib import Path
import numpy as np
import os
import torch
import matplotlib
from plyfile import PlyData, PlyElement
import random
from PIL import Image
import imageio

def prob_to_mask(prob):
    """
    Transforms a probability map of stopping points (shape: (n_layer+1, H, W))
    into a binary mask (shape: (H, W, n_layer, 1)) where for each pixel, layers 
    with index ≤ stopping index (as given by argmax) are marked valid.
    """
    num_layer_plus1, H, W = prob.shape
    # Get stopping index for each pixel; values are in {0, 1, ..., n_layer}
    stopping_indices = torch.argmax(prob, dim=0)  # (H, W)
    
    # Create a tensor with layer indices [1, 2, ..., n_layer]
    layer_indices = torch.arange(1, num_layer_plus1, device=prob.device).view(-1, 1, 1)
    
    # Compare: a layer is valid if its index is <= the stopping index.
    pred_mask = (layer_indices <= stopping_indices.unsqueeze(0))
    
    # Permute and unsqueeze to get shape (H, W, n_layer, 1)
    pred_mask = pred_mask.permute(1, 2, 0).unsqueeze(-1)
    return pred_mask




def colorize(value, vmin=None, vmax=None, cmap='rainbow', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
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

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img



def denormalize(x):
    """Reverses the imagenet normalization applied to the input.

    Args:
        x (torch.Tensor - shape(N,3,H,W)): input tensor

    Returns:
        torch.Tensor - shape(N,3,H,W): Denormalized input
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return x * std + mean




def get_pcd_base(H, W, u0, v0, fx, fy):
    x_row = np.arange(0, W)
    x = np.tile(x_row, (H, 1))
    x = x.astype(np.float32)
    u_m_u0 = x - u0

    y_col = np.arange(0, H)  # y_col = np.arange(0, height)
    y = np.tile(y_col, (W, 1)).T
    y = y.astype(np.float32)
    v_m_v0 = y - v0

    x = u_m_u0 / fx
    y = v_m_v0 / fy
    z = np.ones_like(x)
    pw = np.stack([x, y, z], axis=2)  # [h, w, c]
    return pw


def reconstruct_pcd(depth, fx, fy, u0, v0, pcd_base=None, mask=None):
    if type(depth) == torch.__name__:
        depth = depth.cpu().numpy().squeeze()
    # depth = cv2.medianBlur(depth, 5)
    if pcd_base is None:
        H, W = depth.shape
        pcd_base = get_pcd_base(H, W, u0, v0, fx, fy)
    pcd = depth[:, :, None] * pcd_base
    if mask is not None:
        pcd[mask] = 0
    return pcd


def save_point_cloud(pcd, rgb, filename, binary=True):
    """Save an RGB point cloud as a PLY file.
    :paras
        @pcd: Nx3 matrix, the XYZ coordinates
        @rgb: Nx3 matrix, the rgb colors for each 3D point
    """

    if rgb is None:
        gray_concat = np.tile(np.array([128], dtype=np.uint8),
                              (pcd.shape[0], 3))
        points_3d = np.hstack((pcd, gray_concat))
    else:
        assert pcd.shape[0] == rgb.shape[0]
        points_3d = np.hstack((pcd, rgb))
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                 ('green', 'u1'), ('blue', 'u1')]
    if binary is True:
        # Format into Numpy structured array
        vertices = []
        for row_idx in range(points_3d.shape[0]):
            cur_point = points_3d[row_idx]
            vertices.append(
                tuple(
                    dtype(point)
                    for dtype, point in zip(python_types, cur_point)))
        vertices_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(vertices_array, 'vertex')

        # write
        PlyData([el]).write(filename)
    else:
        x = np.squeeze(points_3d[:, 0])
        y = np.squeeze(points_3d[:, 1])
        z = np.squeeze(points_3d[:, 2])
        r = np.squeeze(points_3d[:, 3])
        g = np.squeeze(points_3d[:, 4])
        b = np.squeeze(points_3d[:, 5])

        ply_head = 'ply\n' \
                    'format ascii 1.0\n' \
                    'element vertex %d\n' \
                    'property float x\n' \
                    'property float y\n' \
                    'property float z\n' \
                    'property uchar red\n' \
                    'property uchar green\n' \
                    'property uchar blue\n' \
                    'end_header' % r.shape[0]
        # ---- Save ply data to disk
        np.savetxt(filename, np.column_stack[x, y, z, r, g, b], fmt='%f %f %f %d %d %d', header=ply_head, comments='')




def point_cloud_alignment_and_colorization(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    img: torch.tensor,
    color_behind_layer = [255, 165, 0] # default: orange color
):
    '''
    Align the predicted point cloud to the ground truth and assign colors to
    the predictions for visualization

    img shape: 3 h w (already normalized into [0,1])
    '''
    ori_shape = pred_arr.shape  # input shape

    gt_arr = gt_arr.squeeze()
    gt = gt_arr.reshape(-1, 3)  # [H, W, layers, 3] -> [M, 3]
    pred_arr = pred_arr.squeeze()
    pred = pred_arr.reshape(-1, 3)


    # assign color to point clouds: [M,3] -> [M, 6]
    img = img.permute(1, 2, 0).unsqueeze(2).cpu().numpy() # H W 1 3
    img = (img * 255.0).astype(np.uint8)
    color_behind_layer = np.array([[[color_behind_layer]]]).astype(np.uint8) # 1 1 1 3
    color_behind_layer = np.broadcast_to(color_behind_layer, (img.shape[0], img.shape[1], ori_shape[2]-1, 3)) # H W Layer-1 3
    layered_color = np.concatenate((img, color_behind_layer), axis=2) # H W L 3
    layered_color = layered_color.reshape(-1, 3)


    valid_mask_arr = valid_mask_arr.squeeze().reshape(-1) # [H,W,layers] -> [M]

    gt = gt[valid_mask_arr.astype(bool)] # V,3
    pred = pred[valid_mask_arr.astype(bool)] 
    color = layered_color[valid_mask_arr.astype(bool)] # V,3

    gt = np.concatenate((gt, color), axis=1) # V 6
    pred = np.concatenate((pred, color), axis=1) # V 6


    # numpy solver
    B = np.concatenate(
        [gt[:, 0],
        gt[:, 1],
        gt[:, 2]],
        axis=0
    ) # [3M]

    A = np.concatenate(
        [np.stack((pred[:,0], np.zeros(pred[:,0].shape)), axis=-1),
         np.stack((pred[:,1], np.zeros(pred[:,1].shape)), axis=-1),
         np.stack((pred[:,2], np.ones(pred[:,2].shape)), axis=-1)
         ]
    ) # [3M, 2]

    X = np.linalg.lstsq(A, B, rcond=None)[0]
    scale, shift = X

    # do with matrix
    aligned_pred_matrix = (pred_arr * scale)
    aligned_pred_matrix[:,:,:,2] = aligned_pred_matrix[:,:,:,2] + shift # apply z-shifts
    aligned_pred_matrix = aligned_pred_matrix.reshape(ori_shape) # restore dimensions
    # do with 3dpts
    pred[:, :3] = pred[:, :3] * scale
    pred[:,2] = pred[:,2] + shift

    return aligned_pred_matrix, pred, gt # M, 6


def make_wandb_vis(image, lari_gt, lari_pred, valid_mask, pred_mask, n_vis_layer=5, n_3dpts=10000):
    '''
    Input:
    Assume the geoemtry inputs are [H, W, layered, 3] pytorch arraies,
    the masks are [H W layered 1] pytorch arraries
    the images are [3 H W] pytorch arraies

    Process:
    Align Pred point maps with GT, then save masked Pred and GT with image,
    respectively

    Output:
    two numpy images, each in shape [h, lay * w, 3]

    '''
    # H W L 3
    lari_gt = lari_gt[:,:,:n_vis_layer,:].squeeze().cpu().detach().numpy()
    lari_pred = lari_pred[:,:,:n_vis_layer,:].squeeze().cpu().detach().numpy()

    # H W L
    valid_mask = valid_mask[:,:,:n_vis_layer].squeeze().cpu().detach().numpy()
    pred_mask = pred_mask[:,:,:n_vis_layer].squeeze().cpu().detach().numpy()

    h, w = image.shape[-2:]
    
    # image
    image = denormalize(image).squeeze()
    image = torch.clip(image, min=0, max=1.0)
    im = transforms.ToPILImage()(image)


    # h w l 3 | v 3
    align_pred_lapt, align_pred_3dpts, gt_3dpts = point_cloud_alignment_and_colorization(lari_gt, 
                                                                                 lari_pred, 
                                                                                 valid_mask,
                                                                                 image)


    # ------ prepare for the point cloud ------
    n_valid_pts3d = gt_3dpts.shape[0]
    pts3d_sampled_idx = np.random.randint(0, n_valid_pts3d, min(n_3dpts, n_valid_pts3d))
    # n_sampled_pts, 3 or 6 (with color)
    align_pred_3dpts = align_pred_3dpts[pts3d_sampled_idx]
    gt_3dpts = gt_3dpts[pts3d_sampled_idx]

    # to sample from unmasked prediction
    pts3d_pre_unmask = align_pred_lapt.reshape(-1, 3)
    pts3d_sampled_idx_ori = np.random.randint(0, pts3d_pre_unmask.shape[0], min(n_3dpts*3, pts3d_pre_unmask.shape[0]))
    pts3d_pre_unmask = pts3d_pre_unmask[pts3d_sampled_idx_ori]

    # ------ prepare for the depth ------
    align_pred_depth = align_pred_lapt[:,:,:,-1] # h w layer
    gt_depth = lari_gt[:,:,:,-1]


    
    min_gt_val = gt_depth.min()
    if min_gt_val < 0: # for GTs that are scale-shift transformed during evaluation
        gt_depth = gt_depth - min_gt_val
        align_pred_depth = align_pred_depth - min_gt_val


    # validmask = gt_depth > 0
    valid_values = gt_depth[valid_mask]
    # Handle empty valid values
    if valid_values.size == 0:
        print("Warning: No valid depth values found. Using default range.")
        vis_depth_range = [0, 1]
    else:
        vis_depth_range = [
            np.percentile(valid_values, 0),
            np.percentile(valid_values, 100),
        ]

    image = Image.new("RGB", (n_vis_layer * w + 1,  4 * h))
    image.paste(im, (0, 0))
    for layer_id in range(n_vis_layer):
        d = colorize(gt_depth[:,:,layer_id], vis_depth_range[0], vis_depth_range[1], invalid_mask=~valid_mask[:,:,layer_id], cmap="Spectral")
        p = colorize(align_pred_depth[:,:,layer_id], vis_depth_range[0], vis_depth_range[1], cmap="Spectral")

        m_pred = Image.fromarray((pred_mask[:,:,layer_id] * 255.0).astype(np.uint8), mode="L").convert("RGB")
        m_gt = Image.fromarray((valid_mask[:,:,layer_id] * 255.0).astype(np.uint8), mode="L").convert("RGB")

        # make a 2-row | n_layer+1 image grid
        image.paste(Image.fromarray(p), (w + layer_id * w, 0))
        image.paste(Image.fromarray(d), (w + layer_id * w, h))
        image.paste(m_gt, (w + layer_id * w, 2*h))
        image.paste(m_pred, (w + layer_id * w, 3*h))

    return np.array(image), align_pred_3dpts, gt_3dpts, pts3d_pre_unmask