import torch.nn.functional as F
from .vit.vision_transformer import VisionTransformer
from .vit.vision_transformer import DINOHead
from torch import nn

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

        student_backbone = VisionTransformer(patch_size=cfg.NET.PATCH_SIZE, drop_path_rate=cfg.NET.DROP_PATH_RATE)
        teacher_backbone = VisionTransformer(patch_size=cfg.NET.PATCH_SIZE)
        student_head = DINOHead(cfg.NET.EMBED_DIM, cfg.NET.OUT_DIM, use_bn=cfg.NET.DINO.IS_BN_IN_HEAD, norm_last_layer=cfg.NET.DINO.IS_NORM_LAST_LAYER)
        teacher_head = DINOHead(cfg.NET.EMBED_DIM, cfg.NET.OUT_DIM, use_bn=cfg.NET.DINO.IS_BN_IN_HEAD)

        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        student_model_dict["head"] = student_head
        teacher_model_dict["head"] = teacher_head

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, crops_list):
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
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups
