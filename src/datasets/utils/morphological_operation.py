import torch
import torch.nn.functional as F

def morphological_open_cpu(mask, kernel_size=3, iterations=1):
    """
    Apply morphological opening (erosion followed by dilation) on CPU.
    
    Args:
        mask (torch.Tensor): Binary mask tensor of shape (H, W, L) with values 0 or 1.
        kernel_size (int): Size of the square structuring element.
        iterations (int): Number of times to apply the operation.
    
    Returns:
        torch.Tensor: Processed mask tensor of shape (H, W, L).
    """
    # Rearrange mask to shape (L, 1, H, W) so that we can process all layers in parallel.
    mask_proc = mask.permute(2, 0, 1).unsqueeze(1).float()  # shape: (L, 1, H, W)
    
    for _ in range(iterations):
        # Erosion: For binary images, erosion = 1 - dilation(1 - image)
        inverted = 1 - mask_proc
        eroded = 1 - F.max_pool2d(inverted, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        # Dilation: Apply max pooling to the eroded result.
        mask_proc = F.max_pool2d(eroded, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    # Rearrange back to original shape (H, W, L)
    opened_mask = mask_proc.squeeze(1).permute(1, 2, 0)
    return opened_mask.bool()


def morphological_close_cpu(mask, kernel_size=3, iterations=1):
    """
    Apply morphological closing (dilation followed by erosion) on CPU.
    
    Args:
        mask (torch.Tensor): Binary mask tensor of shape (H, W, L) with values 0 or 1.
        kernel_size (int): Size of the square structuring element.
        iterations (int): Number of times to apply the operation.
    
    Returns:
        torch.Tensor: Processed mask tensor of shape (H, W, L).
    """
    # Rearrange mask to shape (L, 1, H, W)
    mask_proc = mask.permute(2, 0, 1).unsqueeze(1).float()
    
    for _ in range(iterations):
        # Dilation: Apply max pooling directly.
        dilated = F.max_pool2d(mask_proc, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        # Erosion: For binary images, erosion = 1 - dilation(1 - image)
        inverted = 1 - dilated
        mask_proc = 1 - F.max_pool2d(inverted, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    # Rearrange back to (H, W, L)
    closed_mask = mask_proc.squeeze(1).permute(1, 2, 0)
    return closed_mask.bool()