import torch.nn.functional as F
from .vit import DINOHead
from .vit import vision_transformer as vits
import torch.nn as nn


# new module class
class DinoV2(nn.Module):
    def __init__(self):
        super(DinoV2, self).__init__()
        
        self.head = None
        self.student = nn.Sequential(
            
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x