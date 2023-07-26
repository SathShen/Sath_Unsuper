import torch
from thop import profile
from Networks.Swin.swinv2unet import SwinV2UNet
import torch.nn as nn
from Utils import try_gpu

if __name__ == '__main__':
    net = SwinV2UNet(img_size=512, patch_size=8, in_chans=3, num_classes=2, embed_dim=192, depth=[2, 2, 6, 2], 
                    num_heads=[3, 6, 12, 24], window_size=8, mlp_ratio=4, qkv_bias=True, drop_rate= 0,
                    attn_drop_rate= 0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                    use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], device_id= 0)# 定义好的网络模型
    device = try_gpu(0)
    inputs = torch.randn(1, 3, 512, 512, device=device)
    flops, params = profile(net.to(device), inputs=(inputs,))
    print('flops: ', flops, 'params: ', params)