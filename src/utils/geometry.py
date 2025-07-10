import torch
import numpy as np
from scipy.spatial import cKDTree as KDTree
from src.utils.misc import invalid_to_zeros, invalid_to_nans
# from src.utils.device import to_numpy




def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """ Output a (H,W,2) array of int32 
        with output[j,i,0] = i + origin[0]
             output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid


def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')


def depthmap_to_pts3d(depth, pseudo_focal, pp=None, **_):
    """
    Args:
        - depthmap (BxHxW array):
        - pseudo_focal: [B,H,W] ; [B,2,H,W] or [B,1,H,W]
    Returns:
        pointmap of absolute coordinates (BxHxWx3 array)
    """

    if len(depth.shape) == 4:
        B, H, W, n = depth.shape
    else:
        B, H, W = depth.shape
        n = None

    if len(pseudo_focal.shape) == 3:  # [B,H,W]
        pseudo_focalx = pseudo_focaly = pseudo_focal
    elif len(pseudo_focal.shape) == 4:  # [B,2,H,W] or [B,1,H,W]
        pseudo_focalx = pseudo_focal[:, 0]
        if pseudo_focal.shape[1] == 2:
            pseudo_focaly = pseudo_focal[:, 1]
        else:
            pseudo_focaly = pseudo_focalx
    else:
        raise NotImplementedError("Error, unknown input focal shape format.")

    assert pseudo_focalx.shape == depth.shape[:3]
    assert pseudo_focaly.shape == depth.shape[:3]
    grid_x, grid_y = xy_grid(W, H, cat_dim=0, device=depth.device)[:, None]

    # set principal point
    if pp is None:
        grid_x = grid_x - (W - 1) / 2
        grid_y = grid_y - (H - 1) / 2
    else:
        grid_x = grid_x.expand(B, -1, -1) - pp[:, 0, None, None]
        grid_y = grid_y.expand(B, -1, -1) - pp[:, 1, None, None]

    if n is None:
        pts3d = torch.empty((B, H, W, 3), device=depth.device)
        pts3d[..., 0] = depth * grid_x / pseudo_focalx
        pts3d[..., 1] = depth * grid_y / pseudo_focaly
        pts3d[..., 2] = depth
    else:
        pts3d = torch.empty((B, H, W, 3, n), device=depth.device)
        pts3d[..., 0, :] = depth * (grid_x / pseudo_focalx)[..., None]
        pts3d[..., 1, :] = depth * (grid_y / pseudo_focaly)[..., None]
        pts3d[..., 2, :] = depth
    return pts3d



def ldi_to_pts3d(ldi, camera_intrinsics):
    n_layers = ldi.shape[-1]
    pts3d = []
    mask = []
    for ll in range(n_layers):
        depth = ldi[:,:,ll]
        pts3d_l, mask_l = depthmap_to_camera_coordinates(depth, camera_intrinsics)
        pts3d.append(pts3d_l)
        mask.append(mask_l)
    
    pts3d = np.stack(pts3d, axis=-2) # H W n_layer 3
    mask = np.stack(mask, axis=-1)# H W n_layer

    return pts3d, mask




def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = (depthmap > 0.0)
    return X_cam, valid_mask




def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K







def scale_shift_inv_alignment_inverse(prediction, target, mask):
    '''
    Perform scale-shift alignment to <pts3d_pred> with least square's solution
    pred, gt: B H W L 3
    mask: B H W L 1
    '''

    assert mask.sum() != 0

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2, 3, 4)) # B -- sum(x1^2 + y1^2 + z1^2)
    a_01 = torch.sum(mask.squeeze(-1) * prediction[:,:,:,:,2], (1, 2, 3)) # B -- sum(z1)
    a_11 = torch.sum(mask, (1, 2, 3, 4)) # B -- valid_points of 1
    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2, 3, 4)) # B -- sum(x1y1 + x2y2 + x3y3)
    b_1 = torch.sum(mask.squeeze(-1) * target[:,:,:,:,2], (1, 2, 3)) # B -- sum(z2)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)
    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0 #1e-3

    # B
    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    # apply to the original data
    mask_update = torch.logical_and(mask.squeeze(-1), valid[:, None, None, None]) # B H W L
    # prediction_update = prediction.clone()
    prediction_update = x_0[...,None,None,None,None] * prediction.clone()
    prediction_update[..., 2] = prediction_update[..., 2] + x_1[:, None, None, None] # apply scale to all xyz and shift to z
    gt_update = mask_update[..., None] * target # B H W L 3


    return prediction_update, gt_update, mask_update, (x_0, x_1), valid







def scale_shift_firstlayer_alignment_inverse(prediction, target, mask):
    '''
    Perform scale-shift alignment to <pts3d_pred> with least square's solution
    using only the first layer
    pred, gt: B H W L 3
    mask: B H W L 1
    '''

    mask_init = mask.clone()
    prediction_init = prediction.clone()
    target_init = target.clone()


    mask = mask[..., 0:1, :] # B H W L=1 1
    prediction = prediction[..., 0:1, :] # B H W L=1 3
    target = target[..., 0:1, :] # B H W L=1 3


    assert mask.sum() != 0

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2, 3, 4)) # B -- sum(x1^2 + y1^2 + z1^2)
    a_01 = torch.sum(mask.squeeze(-1) * prediction[:,:,:,:,2], (1, 2, 3)) # B -- sum(z1)
    a_11 = torch.sum(mask, (1, 2, 3, 4)) # B -- valid_points of 1
    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2, 3, 4)) # B -- sum(x1y1 + x2y2 + x3y3)
    b_1 = torch.sum(mask.squeeze(-1) * target[:,:,:,:,2], (1, 2, 3)) # B -- sum(z2)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)
    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0 #1e-3

    # B
    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    # apply to the original data
    mask_update = torch.logical_and(mask_init.squeeze(-1), valid[:, None, None, None]) # B H W L
    
    # prediction_update = prediction.clone()
    prediction_update = x_0[...,None,None,None,None] * prediction_init.clone()
    prediction_update[..., 2] = prediction_update[..., 2] + x_1[:, None, None, None] # apply scale to all xyz and shift to z
    gt_update = mask_update[..., None] * target_init # B H W L 3

    return prediction_update, gt_update, mask_update, (x_0, x_1), valid





def scale_shift_commonlayers_alignment_inverse(prediction, target, mask):
    '''
    Perform scale-shift alignment to <pts3d_pred> with least square's solution
    using only the first layer
    pred, gt: B H W L 3
    mask: B H W L 1
    '''

    mask_init = mask.clone()
    prediction_init = prediction.clone()
    target_init = target.clone()


    common_n_layers = min(target.shape[-2], prediction.shape[-2])

    mask = mask[..., 0:common_n_layers, :] # B H W L=1 1
    prediction = prediction[..., 0:common_n_layers, :] # B H W L=1 3
    target = target[..., 0:common_n_layers, :] # B H W L=1 3


    assert mask.sum() != 0

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2, 3, 4)) # B -- sum(x1^2 + y1^2 + z1^2)
    a_01 = torch.sum(mask.squeeze(-1) * prediction[:,:,:,:,2], (1, 2, 3)) # B -- sum(z1)
    a_11 = torch.sum(mask, (1, 2, 3, 4)) # B -- valid_points of 1
    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2, 3, 4)) # B -- sum(x1y1 + x2y2 + x3y3)
    b_1 = torch.sum(mask.squeeze(-1) * target[:,:,:,:,2], (1, 2, 3)) # B -- sum(z2)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)
    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0 #1e-3

    # B
    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    # apply to the original data
    mask_update = torch.logical_and(mask_init.squeeze(-1), valid[:, None, None, None]) # B H W L
    
    # prediction_update = prediction.clone()
    prediction_update = x_0[...,None,None,None,None] * prediction_init.clone()
    prediction_update[..., 2] = prediction_update[..., 2] + x_1[:, None, None, None] # apply scale to all xyz and shift to z
    gt_update = mask_update[..., None] * target_init # B H W L 3

    return prediction_update, gt_update, mask_update, (x_0, x_1), valid
