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
from src.lari.model.dpt_seg_head import DPTSegHead



class DinoSegModel(nn.Module):

    def __init__(self, 
        encoder: str = 'dinov2_vitl14', 
        intermediate_layers: Union[int, List[int]] = 4,
        dim_proj: int = 512,
        use_pretrained: Literal["dinov2", "moge_full", "moge_backbone", None] = None,
        pretrained_path: str = None,
        num_output_layer: str = None,
        output_type: str = "ray_stop", # "seg_sep"
        **deprecated_kwargs
    ):
        super(DinoSegModel, self).__init__()
        if deprecated_kwargs:
            warnings.warn(f"The following deprecated/invalid arguments are ignored: {deprecated_kwargs}")

        self.encoder = encoder
        self.intermediate_layers = intermediate_layers
        self.use_pretrained = use_pretrained
        self.pretrained_path = pretrained_path
        self.num_output_layer = num_output_layer
        self.output_type = output_type
        assert self.output_type in ["seg_sep", "ray_stop"]

        hub_loader = getattr(importlib.import_module(".dinov2.hub.backbones", __package__), encoder)

        self.backbone = hub_loader(pretrained=True if self.use_pretrained == "dinov2" else False)
        dim_feature = self.backbone.blocks[0].attn.qkv.in_features
        

        

        self.head = DPTSegHead(in_channels=dim_feature, 
                                features=dim_proj, 
                                use_bn=True, 
                                out_channels=[256, 512, 1024, 1024], 
                                use_clstoken=False,
                                num_classes = num_output_layer,
                                output_type = self.output_type
                                )


        if torch.__version__ >= '2.0':
            self.enable_pytorch_native_sdpa()

        self._load_pretrained()
    

    def _load_pretrained(self):
        '''
        Load data from MoGe model
        '''
        return

        


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path, IO[bytes]], model_kwargs: Optional[Dict[str, Any]] = None, **hf_kwargs) -> 'DinoSegModel':
        """
        Load a model from a checkpoint file.

        ### Parameters:
        - `pretrained_model_name_or_path`: path to the checkpoint file or repo id.
        - `model_kwargs`: additional keyword arguments to override the parameters in the checkpoint.
        - `hf_kwargs`: additional keyword arguments to pass to the `hf_hub_download` function. Ignored if `pretrained_model_name_or_path` is a local path.

        ### Returns:
        - A new instance of `MoGe` with the parameters loaded from the checkpoint.
        """
        if Path(pretrained_model_name_or_path).exists():
            checkpoint = torch.load(pretrained_model_name_or_path, map_location='cpu', weights_only=True)
        else:
            cached_checkpoint_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                repo_type="model",
                filename="model.pt",
                **hf_kwargs
            )
            checkpoint = torch.load(cached_checkpoint_path, map_location='cpu', weights_only=True)
        model_config = checkpoint['model_config']
        if model_kwargs is not None:
            model_config.update(model_kwargs)
        model = cls(**model_config)
        model.load_state_dict(checkpoint['model'])
        return model

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
        mask = self.head(features, patch_h, patch_w)

        # b c h w
        mask = F.interpolate(mask, (raw_img_h, raw_img_w), mode="bilinear", align_corners=False)
        
        out_dict = {}

        if self.output_type == "seg_sep":
            # mask = torch.nn.functional.sigmoid(mask) # for binary segmentation
            out_dict["mask"] = mask.permute(0, 2, 3, 1).unsqueeze(-1) # B H W L 1
        elif self.output_type == "ray_stop":
            out_dict["seg_prob"] = mask # B L+1 H W

        return out_dict