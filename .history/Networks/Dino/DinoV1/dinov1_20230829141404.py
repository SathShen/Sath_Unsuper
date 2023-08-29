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
        
        self.student = VisionTransformer(patch_size=cfg.NET.PATCH_SIZE, drop_path_rate=cfg.NET.DROP_PATH_RATE)
        self.teacher = VisionTransformer(patch_size=cfg.NET.PATCH_SIZE)
        self.student_head = DINOHead(cfg.NET.EMBED_DIM, cfg.NET.OUT_DIM, use_bn=cfg.NET.DINO.IS_BN_IN_HEAD, norm_last_layer=cfg.NET.DINO.IS_NORM_LAST_LAYER)
        self.teacher_head = DINOHead(cfg.NET.EMBED_DIM, cfg.NET.OUT_DIM, use_bn=cfg.NET.DINO.IS_BN_IN_HEAD)

    def forward(self, imgs):
        s1 = self.student(imgs)
        t1 = self.teacher(imgs)
        student_output = self.student_head(s1)
        teacher_output = self.teacher_head(t1)
        return student_output, teacher_output
    

def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model