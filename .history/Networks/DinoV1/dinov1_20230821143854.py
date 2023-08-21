import torch.nn.functional as F
import vit.vision_transformer as vits
from vit.vision_transformer import DINOHead
from torch import nn

# new module class
class DinoV1(nn.Module):
    def __init__(self, cfgs):
        super(DinoV1, self).__init__()
        
        self.student = vits(patch_size=args.patch_size, drop_path_rate=args.drop_path_rate)
        self.teacher = vits(patch_size=args.patch_size)
        self.dino_head = DINOHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer)

    def forward(self, imgs):
        student_output = self.student(imgs)
        out = self.dino_head(student_output)
        return out