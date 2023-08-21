import torch.nn.functional as F
import vit.vision_transformer as vits
from vit.vision_transformer import DINOHead

# new module class
class DinoV1(nn.Module):
    def __init__(self):
        super(DinoV1, self).__init__()
        
        self.dino_head = DINOHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer)
        self.student = nn.Sequential(
            
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x