from copy import copy, deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.geometry import scale_shift_inv_alignment_inverse

def Sum(*losses_and_masks):
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


class BaseCriterion(nn.Module):
    def __init__(self, reduction='mean', **kwargs):
        super().__init__()
        self.reduction = reduction


class LLoss (BaseCriterion):
    """ L-norm loss
    """

    def forward(self, a, b, **kwargs):
        # assert a.shape == b.shape and a.ndim >= 2, f'Bad shape = {a.shape}'
        dist = self.distance(a, b, **kwargs)
        # assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b, **kwargs):
        raise NotImplementedError()


class L21Loss (LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b, **kwargs):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance


class L21_NoClip(LLoss):
    """ 
    Euclidean distance between 3d points with the largest/smallest 
    5% losses removed for robustness.
    """
    def __init__(self, reduction='mean', **kwargs):
        super().__init__(reduction, **kwargs)
        self.clip_ratio = kwargs.get("clip_ratio", None)

    def distance(self, a, b, **kwargs):
        '''
        a, b: in shape B H W L 3
        '''
        valid_mask = kwargs.get("valid_mask", None) # should be in B H W L
        assert valid_mask is not None and valid_mask.max() == 1

        loss = torch.norm(a[valid_mask] - b[valid_mask], dim=-1) # normalized L2 distance, N_valid

        return loss



class L1Seg(LLoss):
    '''
    a binary segmentation loss referred from MoGe
    '''
    def distance(self, a, b, **kwargs):
        # force to have 1 dim less to meet the requirements in LLoss
        return torch.abs(a.float() - b.float()).squeeze()


class CELoss(LLoss):
    '''
    a cross-entropy loss
    '''
    def distance(self, a, b, **kwargs):
        res = torch.nn.functional.cross_entropy(input=a, target=b, ignore_index=-100)
        return res


class BCELogitLoss(LLoss):
    '''
    a cross-entropy loss
    '''
    def distance(self, a, b, **kwargs):
        res = torch.nn.functional.binary_cross_entropy_with_logits(input=a, target=b)
        return res





L21 = L21Loss()
L2CN = L21_NoClip()
L1Mask = L1Seg()
CE = CELoss()
BCELogit = BCELogitLoss()


class Criterion (nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, BaseCriterion), f'{criterion} is not a proper criterion!'
        self.criterion = copy(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

    def with_reduction(self, mode='none'):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = mode  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss (nn.Module):
    """ Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()

        
    dlee: it's a chained structure where multiple (>2) losses are chained with
    this self._loss2 argument 
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res
    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f'{self._alpha:g}*{name}'
        if self._loss2:
            name = f'{name} + {self._loss2}'
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details





class SSIRegrSingle3D(Criterion, MultiLoss):
    """ 
    compute the loss between predicted 3D point cloud and the GT 3D points,
    by least-square alignment
    """

    def __init__(self, criterion, max_invalid=None):
        super().__init__(criterion)
        # manually set invalid pixels of GT to a large value 
        self.max_invalid = max_invalid

    def get_all_pts3d(self, pred, data):
        pts3d_gt = data["pts3d"]
        mask_gt = data["mask"]
        pts3d_pred = pred["pts3d"]

        # compute scale-shift factors to prediction, if samples with no valid factors,
        # return zero predictions and ground truths
        pts3d_pred, pts3d_gt, mask_det_and_gt, scale_shift, _ = scale_shift_inv_alignment_inverse(pts3d_pred, pts3d_gt, mask_gt)

        return pts3d_pred, pts3d_gt, mask_det_and_gt, scale_shift


    def compute_loss(self, pred, data, **kw):
        pts3d_pred, pts3d_gt, mask_gt, scale_shift = self.get_all_pts3d(pred, data, **kw) # B H W L 3

        if self.max_invalid is not None:
            mask_gt_valid = mask_gt.unsqueeze(-1).expand(pts3d_gt.shape).bool() # B H W L 3
            scale_, shift = scale_shift # B
            # NOTE: perform scale-shift operation to the invalid value, to ensure the invalid prediction of the orignal network stays consistent
            invalid_val = torch.tensor([self.max_invalid], dtype=torch.float32, device=pts3d_gt.device)[..., None, None, None, None].expand(pts3d_gt.shape)
            invalid_val = invalid_val * scale_[..., None, None, None, None]
            invalid_val[..., -1] = invalid_val[..., -1] + shift[..., None, None, None]
            # set invalid values
            pts3d_gt = torch.where(mask_gt_valid, pts3d_gt, invalid_val)
            # mark the previous valid mask as "all valid" for complete supervision
            mask_gt = mask_gt.new_ones(mask_gt.shape).bool()


        # direct loss on 3d point clouds
        l1 = self.criterion(pts3d_pred, pts3d_gt, valid_mask=mask_gt)
        self_name = type(self).__name__
        details = {self_name + 'pts3d': float(l1.mean())}
        return Sum((l1, mask_gt)), details




class RegrMask(Criterion, MultiLoss):
    """ 
    compute binary segmentation loss for valid region segmentation
    """

    def __init__(self, criterion):
        super().__init__(criterion)

    def compute_loss(self, pred, data, **kw):
        # should be in (b h w l 1)
        mask_gt = data["mask"].float()
        mask_pred = pred["mask"]

        # direct loss on 3d point clouds
        loss = self.criterion(mask_gt, mask_pred)
        self_name = type(self).__name__
        details = {self_name + 'mask': float(loss.mean())}
        return Sum((loss, None)), details




class RegrStopPoint(Criterion, MultiLoss):
    """ 
    compute binary segmentation loss for valid region segmentation
    """

    def __init__(self, criterion):
        super().__init__(criterion)

    def depth_to_stop_index(self, pts3d):
        '''
        taking the index of the farthest (largest depth) layer of each pixel 
        as the stopping index (NOTE: starting from 1 for all valid areas)
        
        input: B H W L 3
        '''
        # print("min z-val:{}".format(pts3d[...,-1].min()))

        assert pts3d[...,-1].min() == 0 # invalid area should be 0

        stop_index = torch.argmax(pts3d[...,-1], dim=-1, keepdim=False) + 1
        # the invalid area will be marked as index "0"
        valid_mask = (pts3d[..., 0, -1] != 0)
        stop_index = stop_index * valid_mask
        return stop_index        

    def mask_to_stop_index(self, valid_mask):
        """
        Converts a layered binary mask (shape: (B, H, W, L, 1)) into a stopping index map (shape: (B, H, W)).
        
        Args:
            valid_mask (torch.Tensor): A binary mask of shape (B, H, W, L, 1) where L is the number of layers.
            
        Returns:
            torch.Tensor: A stopping index map of shape (B, H, W) with values in {0, 1, ..., L}.
                        (0 indicates an invalid pixel with no intersection.)
        """
        # Remove the last singleton dimension, making the shape (B, H, W, L)
        mask = valid_mask.squeeze()
        stop_index = mask.sum(dim=-1)
        
        return stop_index


    def compute_loss(self, pred, data, **kw):
        prob_gt = self.mask_to_stop_index(data["mask"])
        prob_pred = pred["seg_prob"] # b l h w

        loss = self.criterion(prob_pred, prob_gt)

        self_name = type(self).__name__
        details = {self_name + 'prob': float(loss.mean())}
        return Sum((loss, None)), details