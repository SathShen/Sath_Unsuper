import torch.nn.functional as F
from .vit.vision_transformer import VisionTransformer as vits
from .vit.vision_transformer import DINOHead
from torch import nn

# new module class
class DinoV1(nn.Module):
    def __init__(self, cfgs):
        super(DinoV1, self).__init__()
        
        self.student = vits(patch_size=args.patch_size, drop_path_rate=args.drop_path_rate)
        self.teacher = vits(patch_size=args.patch_size)
        self.student_head = DINOHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer)
        self.teacher_head = DINOHead(embed_dim, args.out_dim, args.use_bn_in_head)

    def forward(self, imgs):
        s1 = self.student(imgs)
        t1 = self.teacher(imgs)
        student_output = self.student_head(s1)
        teacher_output = self.teacher_head(t1)
        return student_output, teacher_output