
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import torch.version
from typing import *
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from src.lari.model.blocks import ResidualConvBlock, make_upsampler, make_output_block
from src.lari.utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift, gaussian_blur_2d


class PointHead(nn.Module):
    def __init__(
        self, 
        num_features: int,
        dim_in: int, 
        dim_out: int, 
        dim_proj: int = 512,
        dim_upsample: List[int] = [256, 128, 128],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_norm: Literal['group_norm', 'layer_norm'] = 'group_norm',
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1,
        num_output_layer: int = 5
    ):
        super().__init__()
        
        self.num_output_layer = num_output_layer

        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=dim_in, out_channels=dim_proj, kernel_size=1, stride=1, padding=0,) for _ in range(num_features)
        ])

        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                make_upsampler(in_ch + 2, out_ch),
                *(ResidualConvBlock(out_ch, out_ch, dim_times_res_block_hidden * out_ch, activation="relu", norm=res_block_norm) for _ in range(num_res_blocks))
            ) for in_ch, out_ch in zip([dim_proj] + dim_upsample[:-1], dim_upsample)
        ])

        # layer iterations
        self.first_layer_block = make_output_block(dim_upsample[-1] + 2, dim_out, 
                                                   dim_times_res_block_hidden, last_res_blocks, last_conv_channels, last_conv_size, res_block_norm,) 

        self.remaining_layer_block = nn.ModuleList([make_output_block(dim_upsample[-1] + 2, dim_out, 
                                                                      dim_times_res_block_hidden, last_res_blocks, last_conv_channels, last_conv_size, res_block_norm,) 
                                                            for _ in range(self.num_output_layer - 1)])
        

            
    def forward(self, hidden_states: torch.Tensor, image: torch.Tensor):
        img_h, img_w = image.shape[-2:]
        patch_h, patch_w = img_h // 14, img_w // 14

        # Process the hidden states
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous())
                for proj, (feat, clstoken) in zip(self.projects, hidden_states)
        ], dim=1).sum(dim=1)
        
        # Upsample stage
        # (patch_h, patch_w) -> (patch_h * 2, patch_w * 2) -> (patch_h * 4, patch_w * 4) -> (patch_h * 8, patch_w * 8)
        for i, block in enumerate(self.upsample_blocks):
            # UV coordinates is for awareness of image aspect ratio
            uv = normalized_view_plane_uv(width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, uv], dim=1)
            for layer in block:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        
        # (patch_h * 8, patch_w * 8) -> (img_h, img_w)
        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)
        uv = normalized_view_plane_uv(width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device)
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, uv], dim=1)


        pts_list = []
        for layer_id in range(self.num_output_layer):
            if layer_id == 0:
                blocks = self.first_layer_block
            else:
                blocks = self.remaining_layer_block[layer_id-1]
            
            # for each block
            if isinstance(blocks, nn.ModuleList):
                raise NotImplementedError()
            else:
                res = torch.utils.checkpoint.checkpoint(blocks, x, use_reentrant=False)[:,:3, :,:]
                pts_list.append(res[:, :3, :,:])

        pts = torch.stack(pts_list, dim=-1)
        seg = pts.new_zeros(pts.shape)[:, :1, ...]

        # <b 3 h w l>, <b 1 h w l>
        output = [pts, seg]

        return output