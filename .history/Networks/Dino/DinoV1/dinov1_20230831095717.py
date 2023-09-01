import torch
import torch.nn.functional as F
from .vit.vision_transformer import VisionTransformer
from .vit.vision_transformer import DINOHead
from torch import nn
from collections import defaultdict

"""
vit_tiny    embed_dim=192, depth=12, num_heads=3
vit_small   embed_dim=384, depth=12, num_heads=6
vit_base    embed_dim=768, depth=12, num_heads=12
vit_large   embed_dim=1024, depth=24, num_heads=16
vit_huge    embed_dim=1280, depth=32, num_heads=16
vit-g/14    embed_dim=1408, depth=40, num_heads=16, mlp=6144
vit-G/14    embed_dim=1664, depth=48, num_heads=16, mlp=8192
"""

# new module class
class DinoV1(nn.Module):
    def __init__(self, cfg):
        super(DinoV1, self).__init__()
        student_model_dict = dict()
        teacher_model_dict = dict()

        self.student_backbone = VisionTransformer(patch_size=cfg.NET.PATCH_SIZE, drop_path_rate=cfg.NET.DROP_PATH_RATE)
        self.teacher_backbone = VisionTransformer(patch_size=cfg.NET.PATCH_SIZE)
        self.student_head = DINOHead(cfg.NET.EMBED_DIM, cfg.NET.OUT_DIM, use_bn=cfg.NET.DINO.IS_BN_IN_HEAD, norm_last_layer=cfg.NET.DINO.IS_NORM_LAST_LAYER)
        self.teacher_head = DINOHead(cfg.NET.EMBED_DIM, cfg.NET.OUT_DIM, use_bn=cfg.NET.DINO.IS_BN_IN_HEAD)

        student_model_dict["backbone"] = self.student_backbone
        teacher_model_dict["backbone"] = self.teacher_backbone
        student_model_dict["head"] = self.student_head
        teacher_model_dict["head"] = self.teacher_head

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, crops_list):
        if not isinstance(crops_list, list):
            crops_list = [crops_list]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(torch.tensor([inp.shape[-1] for inp in crops_list]),return_counts=True)[1], 0)
        start_idx, output = 0, torch.empty(0).to(crops_list[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(crops_list[start_idx: end_idx]))
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)

        s1 = self.student_backbone(crops_list)
        t1 = self.teacher_backbone(crops_list[:2])
        student_output = self.student_head(s1)
        teacher_output = self.teacher_head(t1)
        return student_output, teacher_output
    
    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups
    
    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = self.get_params_groups_with_decay(
            model=m,
            lr_decay_rate=0.9,   # layerwise_decay: 0.9
            patch_embed_lr_mult=0.2,   # patch_embed_lr_mult: 0.2
        )
        fused_params_groups = self.fuse_params_groups(params_groups)
        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups
    
    def get_params_groups_with_decay(self, model, lr_decay_rate=1.0, patch_embed_lr_mult=1.0):
        chunked_blocks = False
        if hasattr(model, "n_blocks"):
            # logger.info("chunked fsdp")
            n_blocks = model.n_blocks
            chunked_blocks = model.chunked_blocks
        elif hasattr(model, "blocks"):
            # logger.info("first code branch")
            n_blocks = len(model.blocks)
        elif hasattr(model, "backbone"):
            # logger.info("second code branch")
            n_blocks = len(model.backbone.blocks)
        else:
            # logger.info("else code branch")
            n_blocks = 0
        all_param_groups = []

        for name, param in model.named_parameters():
            name = name.replace("_fsdp_wrapped_module.", "")
            if not param.requires_grad:
                continue
            decay_rate = self.get_vit_lr_decay_rate(
                name, lr_decay_rate, num_layers=n_blocks, force_is_backbone=n_blocks > 0, chunked_blocks=chunked_blocks
            )
            d = {"params": param, "is_last_layer": False, "lr_multiplier": decay_rate, "wd_multiplier": 1.0, "name": name}

            if "last_layer" in name:
                d.update({"is_last_layer": True})

            if name.endswith(".bias") or "norm" in name or "gamma" in name:
                d.update({"wd_multiplier": 0.0})

            if "patch_embed" in name:
                d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})

            all_param_groups.append(d)
            # logger.info(f"""{name}: lr_multiplier: {d["lr_multiplier"]}, wd_multiplier: {d["wd_multiplier"]}""")

        return all_param_groups
    
    def fuse_params_groups(self, all_params_groups, keys=("lr_multiplier", "wd_multiplier", "is_last_layer")):
        fused_params_groups = defaultdict(lambda: {"params": []})
        for d in all_params_groups:
            identifier = ""
            for k in keys:
                identifier += k + str(d[k]) + "_"

            for k in keys:
                fused_params_groups[identifier][k] = d[k]
            fused_params_groups[identifier]["params"].append(d["params"])

        return fused_params_groups.values()
    
    def get_vit_lr_decay_rate(self ,name, lr_decay_rate=1.0, num_layers=12, force_is_backbone=False, chunked_blocks=False):
        """
        Calculate lr decay rate for different ViT blocks.
        Args:
            name (string): parameter name.
            lr_decay_rate (float): base lr decay rate.
            num_layers (int): number of ViT blocks.
        Returns:
            lr decay rate for the given parameter.
        """
        layer_id = num_layers + 1
        if name.startswith("backbone") or force_is_backbone:
            if ".pos_embed" in name or ".patch_embed" in name or ".mask_token" in name or ".cls_token" in name:
                layer_id = 0
            elif force_is_backbone and (
                "pos_embed" in name or "patch_embed" in name or "mask_token" in name or "cls_token" in name
            ):
                layer_id = 0
            elif ".blocks." in name and ".residual." not in name:
                layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
            elif chunked_blocks and "blocks." in name and "residual." not in name:
                layer_id = int(name[name.find("blocks.") :].split(".")[2]) + 1
            elif "blocks." in name and "residual." not in name:
                layer_id = int(name[name.find("blocks.") :].split(".")[1]) + 1

        return lr_decay_rate ** (num_layers + 1 - layer_id)
