from typing import *
from numbers import Number
from functools import partial
from pathlib import Path
import importlib
import warnings
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import torch.version
from huggingface_hub import hf_hub_download
from src.lari.model.utils import wrap_dinov2_attention_with_sdpa, wrap_module_with_gradient_checkpointing, unwrap_module_with_gradient_checkpointing
from src.lari.model.heads import PointHead


class LaRIModel(nn.Module):
    image_mean: torch.Tensor
    image_std: torch.Tensor

    def __init__(self, 
        encoder: str = 'dinov2_vitl14', 
        intermediate_layers: Union[int, List[int]] = 4,
        dim_proj: int = 512,
        dim_upsample: List[int] = [256, 128, 64],
        dim_times_res_block_hidden: int = 2,
        num_res_blocks: int = 2,
        output_mask: bool = True,
        split_head: bool = True,
        remap_output: Literal[False, True, 'linear', 'sinh', 'exp', 'sinh_exp'] = 'exp',
        res_block_norm: Literal['group_norm', 'layer_norm'] = 'group_norm',
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1,
        use_pretrained: Literal["dinov2", "moge_full", "moge_backbone", None] = None,
        pretrained_path: str = "",
        num_output_layer: str = None,
        head_type = None,
        **deprecated_kwargs
    ):
        super(LaRIModel, self).__init__()
        if deprecated_kwargs:
            warnings.warn(f"The following deprecated/invalid arguments are ignored: {deprecated_kwargs}")

        self.encoder = encoder
        self.remap_output = remap_output
        self.intermediate_layers = intermediate_layers
        self.head_type = head_type
        self.output_mask = output_mask
        self.split_head = split_head
        self.use_pretrained = use_pretrained
        self.pretrained_path = pretrained_path
        self.num_output_layer = num_output_layer
        
        hub_loader = getattr(importlib.import_module(".dinov2.hub.backbones", __package__), encoder)
        # hub_loader = getattr(importlib.import_module("dinov2.hub.backbones", __package__), encoder)

        self.backbone = hub_loader(pretrained=True if self.use_pretrained == "dinov2" else False)
        dim_feature = self.backbone.blocks[0].attn.qkv.in_features
        
        if self.head_type == "point":
            self.head = PointHead(
                num_features=intermediate_layers if isinstance(intermediate_layers, int) else len(intermediate_layers), 
                dim_in=dim_feature, 
                dim_out=3, 
                dim_proj=dim_proj,
                dim_upsample=dim_upsample,
                dim_times_res_block_hidden=dim_times_res_block_hidden,
                num_res_blocks=num_res_blocks,
                res_block_norm=res_block_norm,
                last_res_blocks=last_res_blocks,
                last_conv_channels=last_conv_channels,
                last_conv_size=last_conv_size,
                num_output_layer = num_output_layer
            )
        else:
            raise NotImplementedError()


        if torch.__version__ >= '2.0':
            self.enable_pytorch_native_sdpa()

        self._load_pretrained()
    

    def _load_pretrained(self):
        '''
        Load pre-trained weights
        '''
        if self.use_pretrained == "dinov2" or self.use_pretrained is None: return

        if self.use_pretrained == "moge_full" and self.pretrained_path != "":
            checkpoint = torch.load(self.pretrained_path, map_location='cpu', weights_only=True)
            if self.head_type == "point":
                key_transition_map = {"output_block": "first_layer_block"}
                model_state_dict = {}

                # change the key name of the dict
                for key, val in checkpoint['model'].items():
                    for trans_src, trans_target in key_transition_map.items():
                        if trans_src in key:
                            model_state_dict[key.replace(trans_src, trans_target)] = val
                        else:
                            model_state_dict[key] = val

                self.load_state_dict(model_state_dict, strict=False)
                del model_state_dict

        
            else:
                return
            
        else:
            return

    @staticmethod
    def cache_pretrained_backbone(encoder: str, pretrained: bool):
        _ = torch.hub.load('facebookresearch/dinov2', encoder, pretrained=pretrained)

    def load_pretrained_backbone(self):
        "Load the backbone with pretrained dinov2 weights from torch hub"
        state_dict = torch.hub.load('facebookresearch/dinov2', self.encoder, pretrained=True).state_dict()
        self.backbone.load_state_dict(state_dict)
    
    def enable_backbone_gradient_checkpointing(self):
        for i in range(len(self.backbone.blocks)):
            self.backbone.blocks[i] = wrap_module_with_gradient_checkpointing(self.backbone.blocks[i])

    def enable_pytorch_native_sdpa(self):
        for i in range(len(self.backbone.blocks)):
            self.backbone.blocks[i].attn = wrap_dinov2_attention_with_sdpa(self.backbone.blocks[i].attn)

    def forward(self, image: torch.Tensor, mixed_precision: bool = False) -> Dict[str, torch.Tensor]:
        raw_img_h, raw_img_w = image.shape[-2:]
        patch_h, patch_w = raw_img_h // 14, raw_img_w // 14

        # Apply image transformation for DINOv2
        image_14 = F.interpolate(image, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=False, antialias=True)

        # Get intermediate layers from the backbone
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=mixed_precision):
            features = self.backbone.get_intermediate_layers(image_14, self.intermediate_layers, return_class_token=True)

        # Predict points and mask (mask scores)
        points, mask = self.head(features, image)

        is_output_prob = False
        if mask.ndim == 5: 
            # <b, h, w, layer, 3>, <b, h, w, layer, 1>
            points, mask = points.permute(0, 2, 3, 4, 1), mask.permute(0,2,3,4,1)
        elif mask.ndim == 4: # <b, h, w, layer, 3>, <b, layer, h, w>
            points = points.permute(0, 2, 3, 4, 1)
            is_output_prob = True

        if self.remap_output == 'linear' or self.remap_output == False:
            pass
        elif self.remap_output =='sinh' or self.remap_output == True:
            points = torch.sinh(points)
        elif self.remap_output == 'exp':
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output =='sinh_exp':
            xy, z = points.split([2, 1], dim=-1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Invalid remap output type: {self.remap_output}")
        
        return_dict = {'pts3d': points}

        if not is_output_prob:
            return_dict['mask'] = mask
        else:
            return_dict["seg_prob"] = mask
        
        return return_dict