from src.utils.geometry import scale_shift_inv_alignment_inverse, scale_shift_commonlayers_alignment_inverse
from copy import copy, deepcopy
import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance




class SSI3DScore(nn.Module):
    """ 
    Compute the 3D metrics (CD and F-score) between the sampled prediction and GT points.
    """

    def __init__(self, num_eval_pts, fs_thres, pts_sampling_mode, eval_layers=None, ldi_vis_only=False):
        super().__init__()
        self.num_eval_pts = num_eval_pts
        self.fs_thres = fs_thres
        self.pts_sampling_mode = pts_sampling_mode
        self.eval_layers = eval_layers
        assert self.eval_layers in [None, "visible", "unseen", "all"]

        self.ldi_vis_only = ldi_vis_only # only do alignment

    def get_all_pts3d(self, pred, data):
        return NotImplementedError()


    def chamfer_and_fscore(self, pred, gt, eval_layers):
        """
        Compute Chamfer Distance and F-score between predicted and ground truth point clouds.
        """

        dist_tuple, _ = chamfer_distance(pred, gt, batch_reduction=None, point_reduction=None, norm=2)
        dist_pred, dist_gt = dist_tuple # B, N

        # Pytorch3D returns Sqared Sum of the distance, we need to manually compute the squared-root
        dist_pred = torch.sqrt(dist_pred)
        dist_gt = torch.sqrt(dist_gt)

        # Mean Chamfer Distance
        chamfer_dist = (dist_pred.mean(dim=1) + dist_gt.mean(dim=1)) / 2

        details = {}
        details = {'CD_{}_{}'.format(self.num_eval_pts, eval_layers if eval_layers else "full"): (float(chamfer_dist.mean()), int(chamfer_dist.shape[0]))}

        if not isinstance(self.fs_thres, list):
            f_score = self.fscore_from_cd(dist_pred, dist_gt, self.fs_thres)
            details.update({"f_score_{}_{}".format(self.fs_thres, eval_layers if eval_layers else "full"): (float(f_score.mean()), int(f_score.shape[0]))})
        else:
            for thres in self.fs_thres:
                f_score = self.fscore_from_cd(dist_pred, dist_gt, thres)
                details.update({"f_score_{}_{}".format(thres, eval_layers if eval_layers else "full"): (float(f_score.mean()), int(f_score.shape[0]))})

        # B
        return details



    def fscore_from_cd(self, dist_pred, dist_gt, fs_thres):
        # Compute F-score
        f_pred = (dist_pred < fs_thres).float().mean(dim=1)
        f_gt = (dist_gt < fs_thres).float().mean(dim=1)
        f_score = 2 * f_pred * f_gt / (f_pred + f_gt + 1e-8)  # Avoid division by zero
        return f_score
    



    def uniform_sample_3dpts_with_interp(self, point_map, mask, num_samples):
        """
        Efficiently sample a specified number of points uniformly across the batch.
        If a sample has fewer valid points than required, it duplicates valid points.
        """
        B, H, W, L, _ = point_map.shape
        device = point_map.device

        # Flatten spatial dimensions
        mask_flat = mask.reshape(B, -1)  # Shape: (B, H*W*L)
        point_map_flat = point_map.reshape(B, -1, 3)  # Shape: (B, H*W*L, 3)

        # Get valid indices for each batch
        valid_indices = torch.nonzero(mask_flat, as_tuple=True)  # Shape: (valid_points,)
        
        batch_ids = valid_indices[0]  # Shape: (valid_points,)
        point_ids = valid_indices[1]  # Shape: (valid_points,)

        # Count valid points per batch
        valid_counts = mask_flat.sum(dim=1)  # Shape: (B,)

        # Compute offsets for each batch in `point_ids`
        offsets = torch.cat([torch.tensor([0], device=device), valid_counts.cumsum(0)[:-1]])  # (B,)
        
        # Generate random sampling indices within each batch
        rand_ids = torch.randint(0, valid_counts.max(), (B, num_samples), device=device) % valid_counts.unsqueeze(1)  # (B, num_samples)
        
        # Compute final sampled indices (global indices in `point_ids`)
        final_sampled_indices = point_ids[rand_ids + offsets.unsqueeze(1)]  # (B, num_samples)

        # Gather the sampled 3D points
        sampled_points = torch.gather(point_map_flat, 1, final_sampled_indices.unsqueeze(-1).expand(-1, -1, 3))

        return sampled_points



    def forward(self, pred, data, **kw):
        return NotImplementedError()
    


class SSI3DScore_Object(SSI3DScore):

    def get_all_pts3d(self, pred, data):
        pts3d_gt = data["pts3d"]
        mask_gt = data["mask"]
        pts3d_pred = pred["pts3d"]

        # perform scale and shift alignment
        pts3d_pred, pts3d_gt, mask_det_and_gt, _, _ = scale_shift_inv_alignment_inverse(pts3d_pred, pts3d_gt, mask_gt)

        bs = mask_det_and_gt.shape[0]
        valid_batch_mask = (torch.sum(mask_det_and_gt.view(bs, -1), dim=-1) != 0) # shape: B

        return pts3d_pred, pts3d_gt, valid_batch_mask


    def forward(self, pred, data, **kw):
        # scale-shift alignment based on LDIs
        pts3d_pred, _, valid_batch_mask = self.get_all_pts3d(pred, data, **kw)
        pts3d_pred_ori = pts3d_pred

        # align the layer number of the mask with the predictions 
        if pts3d_pred.shape[-2] < data["mask"].shape[-2]:
            mask_with_pred_layer = data["mask"][:,:,:,:pts3d_pred.shape[-2],:].squeeze(-1)
        else:
            mask_with_pred_layer = data["mask"].squeeze(-1)

        details_overall = {}
        
        pts3d_uniform_gt = data["pcd_eval"]
        pts3d_pred_eval = self.uniform_sample_3dpts_with_interp(pts3d_pred, mask_with_pred_layer, self.num_eval_pts)
        
        assert pts3d_pred_eval.shape[1] == pts3d_uniform_gt.shape[1], "the prediction and the uniform GT does not match in NUM_PTS!!"

        if valid_batch_mask is not None:
            details = self.chamfer_and_fscore(pts3d_pred_eval[valid_batch_mask], pts3d_uniform_gt[valid_batch_mask], eval_layers=None)
        else:
            details = self.chamfer_and_fscore(pts3d_pred_eval, pts3d_uniform_gt, eval_layers=None)
        
        details_overall.update(details)

        return (pts3d_pred_eval, pts3d_uniform_gt, pts3d_pred_ori), details_overall


class SSI3DScore_Scene(SSI3DScore):
    '''
    3D evaluation metric for depth models
    '''
    
    def get_all_pts3d(self, pred, data, **kw):
        pts3d_gt = data["pts3d"]
        mask_gt = data["mask"]
        pts3d_pred = pred["pts3d"]

        # compute scale-shift factors using common layers of the prediction and GT
        pts3d_pred, pts3d_gt, mask_det_and_gt, scale_shift, _ = scale_shift_commonlayers_alignment_inverse(pts3d_pred, pts3d_gt, mask_gt)

        bs = mask_det_and_gt.shape[0]
        valid_batch_mask = (torch.sum(mask_det_and_gt.view(bs, -1), dim=-1) != 0) # shape: B

        return pts3d_pred, pts3d_gt, valid_batch_mask


    def forward(self, pred, data, **kw):
        # scale-shift alignment based on LDIs
        pts3d_pred, _, valid_batch_mask = self.get_all_pts3d(pred, data, **kw)
        pts3d_pred_ori = pts3d_pred

        # align the layer number of the mask with the predictions 
        if pts3d_pred.shape[-2] < data["mask"].shape[-2]:
            mask_with_pred_layer = data["mask"][:,:,:,:pts3d_pred.shape[-2],:].squeeze(-1)
        else:
            mask_with_pred_layer = data["mask"].squeeze(-1)

        details_overall = {}
        for eval_layers in ["visible", "unseen", None]:
            # select GT
            if not eval_layers:
                pts3d_uniform_gt = data["pcd_eval"]
            else:
                pts3d_uniform_gt = data["pcd_eval_{}".format(eval_layers)] # B N 3

            # sample pred
            if eval_layers is None: # sample from the whole point set
                pts3d_pred_eval = self.uniform_sample_3dpts_with_interp(pts3d_pred, mask_with_pred_layer, self.num_eval_pts)
            elif eval_layers == "visible": # sample from the frist layer
                pts3d_pred_eval = self.uniform_sample_3dpts_with_interp(pts3d_pred[:,:,:,:1,:], mask_with_pred_layer[:,:,:,:1], self.num_eval_pts)
            elif eval_layers == "unseen": # sample from the remaining layers
                pts3d_pred_eval = self.uniform_sample_3dpts_with_interp(pts3d_pred[:,:,:,1:,:], mask_with_pred_layer[:,:,:,1:], self.num_eval_pts)

            assert pts3d_pred_eval.shape[1] == pts3d_uniform_gt.shape[1], "the prediction and the uniform GT does not match in NUM_PTS!!"

            if valid_batch_mask is not None:
                details = self.chamfer_and_fscore(pts3d_pred_eval[valid_batch_mask], pts3d_uniform_gt[valid_batch_mask], eval_layers=eval_layers)
            else:
                details = self.chamfer_and_fscore(pts3d_pred_eval, pts3d_uniform_gt, eval_layers=eval_layers)
            
            details_overall.update(details)


        return (pts3d_pred_eval, pts3d_uniform_gt, pts3d_pred_ori), details_overall