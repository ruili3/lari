# import torchvision.transforms as transforms
# import torch.nn.functional as F
# import cv2
# import os
# import logging
# from pathlib import Path
import numpy as np
# import os
import torch
import matplotlib
# import cv2
# import random
# from PIL import Image
# import imageio

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