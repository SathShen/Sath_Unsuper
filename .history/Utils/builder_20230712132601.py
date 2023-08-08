# --------------------------------------------------------
# Swin Transformer Net builder
# --------------------------------------------------------
import torch
from Networks import *
from Utils.loss import *
import torch.nn as nn
from Utils.augmentation import *

def build_net(config):
    net_type = config.NET.TYPE
    net_name = config.NET.NAME

    if net_type == 'swin':
        if net_name == 'swin':
            net = SwinTransformer(img_size=config.AUG.CROP_SIZE,
                                patch_size=config.NET.SWIN.PATCH_SIZE,
                                in_chans=config.NET.SWIN.IN_CHANS,
                                num_classes=config.DATA.NUM_CLASSES,
                                embed_dim=config.NET.SWIN.EMBED_DIM,
                                depths=config.NET.SWIN.DEPTHS,
                                num_heads=config.NET.SWIN.NUM_HEADS,
                                window_size=config.NET.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.NET.SWIN.MLP_RATIO,
                                qkv_bias=config.NET.SWIN.QKV_BIAS,
                                qk_scale=config.NET.SWIN.QK_SCALE,
                                drop_rate=config.NET.DROP_RATE,
                                drop_path_rate=config.NET.DROP_PATH_RATE,
                                ape=config.NET.SWIN.APE,
                                norm_layer=nn.LayerNorm,
                                patch_norm=config.NET.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        elif net_name =='swinv2':
            net = SwinTransformerV2(img_size=config.AUG.CROP_SIZE,
                                  patch_size=config.NET.SWINV2.PATCH_SIZE,
                                  in_chans=config.NET.SWINV2.IN_CHANS,
                                  num_classes=config.DATA.NUM_CLASSES,
                                  embed_dim=config.NET.SWINV2.EMBED_DIM,
                                  depths=config.NET.SWINV2.DEPTHS,
                                  num_heads=config.NET.SWINV2.NUM_HEADS,
                                  window_size=config.NET.SWINV2.WINDOW_SIZE,
                                  mlp_ratio=config.NET.SWINV2.MLP_RATIO,
                                  qkv_bias=config.NET.SWINV2.QKV_BIAS,
                                  drop_rate=config.NET.DROP_RATE,
                                  drop_path_rate=config.NET.DROP_PATH_RATE,
                                  ape=config.NET.SWINV2.APE,
                                  patch_norm=config.NET.SWINV2.PATCH_NORM,
                                  use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                  pretrained_window_sizes=config.NET.SWINV2.PRETRAINED_WINDOW_SIZES)
        elif net_name =='swinmoe':
             net = SwinTransformerMoE(img_size=config.AUG.CROP_SIZE,
                                   patch_size=config.NET.SWIN_MOE.PATCH_SIZE,
                                   in_chans=config.NET.SWIN_MOE.IN_CHANS,
                                   num_classes=config.DATA.NUM_CLASSES,
                                   embed_dim=config.NET.SWIN_MOE.EMBED_DIM,
                                   depths=config.NET.SWIN_MOE.DEPTHS,
                                   num_heads=config.NET.SWIN_MOE.NUM_HEADS,
                                   window_size=config.NET.SWIN_MOE.WINDOW_SIZE,
                                   mlp_ratio=config.NET.SWIN_MOE.MLP_RATIO,
                                   qkv_bias=config.NET.SWIN_MOE.QKV_BIAS,
                                   qk_scale=config.NET.SWIN_MOE.QK_SCALE,
                                   drop_rate=config.NET.DROP_RATE,
                                   drop_path_rate=config.NET.DROP_PATH_RATE,
                                   ape=config.NET.SWIN_MOE.APE,
                                   patch_norm=config.NET.SWIN_MOE.PATCH_NORM,
                                   mlp_fc2_bias=config.NET.SWIN_MOE.MLP_FC2_BIAS,
                                   init_std=config.NET.SWIN_MOE.INIT_STD,
                                   use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                   pretrained_window_sizes=config.NET.SWIN_MOE.PRETRAINED_WINDOW_SIZES,
                                   moe_blocks=config.NET.SWIN_MOE.MOE_BLOCKS,
                                   num_local_experts=config.NET.SWIN_MOE.NUM_LOCAL_EXPERTS,
                                   top_value=config.NET.SWIN_MOE.TOP_VALUE,
                                   capacity_factor=config.NET.SWIN_MOE.CAPACITY_FACTOR,
                                   cosine_router=config.NET.SWIN_MOE.COSINE_ROUTER,
                                   normalize_gate=config.NET.SWIN_MOE.NORMALIZE_GATE,
                                   use_bpr=config.NET.SWIN_MOE.USE_BPR,
                                   is_gshard_loss=config.NET.SWIN_MOE.IS_GSHARD_LOSS,
                                   gate_noise=config.NET.SWIN_MOE.GATE_NOISE,
                                   cosine_router_dim=config.NET.SWIN_MOE.COSINE_ROUTER_DIM,
                                   cosine_router_init_t=config.NET.SWIN_MOE.COSINE_ROUTER_INIT_T,
                                   moe_drop=config.NET.SWIN_MOE.MOE_DROP,
                                   aux_loss_weight=config.NET.SWIN_MOE.AUX_LOSS_WEIGHT)
        elif net_name =='swinmlp':
             net = SwinMLP(img_size=config.AUG.CROP_SIZE,
                        patch_size=config.NET.SWIN_MLP.PATCH_SIZE,
                        in_chans=config.NET.SWIN_MLP.IN_CHANS,
                        num_classes=config.DATA.NUM_CLASSES,
                        embed_dim=config.NET.SWIN_MLP.EMBED_DIM,
                        depths=config.NET.SWIN_MLP.DEPTHS,
                        num_heads=config.NET.SWIN_MLP.NUM_HEADS,
                        window_size=config.NET.SWIN_MLP.WINDOW_SIZE,
                        mlp_ratio=config.NET.SWIN_MLP.MLP_RATIO,
                        drop_rate=config.NET.DROP_RATE,
                        drop_path_rate=config.NET.DROP_PATH_RATE,
                        ape=config.NET.SWIN_MLP.APE,
                        patch_norm=config.NET.SWIN_MLP.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        elif net_name == "swinv2unet":
            net = SwinV2UNet(img_size=config.AUG.CROP_SIZE,
                        patch_size=config.NET.SWINV2UNET.PATCH_SIZE,
                        in_chans=config.NET.SWINV2UNET.IN_CHANS,
                        num_classes=config.DATA.NUM_CLASSES,
                        embed_dim=config.NET.SWINV2UNET.EMBED_DIM,
                        depths=config.NET.SWINV2UNET.DEPTHS,
                        num_heads=config.NET.SWINV2UNET.NUM_HEADS,
                        window_size=config.NET.SWINV2UNET.WINDOW_SIZE,
                        mlp_ratio=config.NET.SWINV2UNET.MLP_RATIO,
                        qkv_bias=config.NET.SWINV2UNET.QKV_BIAS,
                        drop_rate=config.NET.DROP_RATE,
                        drop_path_rate=config.NET.DROP_PATH_RATE,
                        ape=config.NET.SWINV2UNET.APE,
                        patch_norm=config.NET.SWINV2UNET.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        pretrained_window_sizes=config.NET.SWINV2UNET.PRETRAINED_WINDOW_SIZES,
                        device_id=config.DEVICE_IDS[0])
            
    elif net_type == 'simmim':
        if net_name == 'simmim':
            encoder = SwinTransformerForSimMIM(img_size=config.AUG.CROP_SIZE,
                                            patch_size=config.NET.SWIN.PATCH_SIZE,
                                            in_chans=config.NET.SWIN.IN_CHANS,
                                            num_classes=0,
                                            embed_dim=config.NET.SWIN.EMBED_DIM,
                                            depths=config.NET.SWIN.DEPTHS,
                                            num_heads=config.NET.SWIN.NUM_HEADS,
                                            window_size=config.NET.SWIN.WINDOW_SIZE,
                                            mlp_ratio=config.NET.SWIN.MLP_RATIO,
                                            qkv_bias=config.NET.SWIN.QKV_BIAS,
                                            qk_scale=config.NET.SWIN.QK_SCALE,
                                            drop_rate=config.NET.DROP_RATE,
                                            drop_path_rate=config.NET.DROP_PATH_RATE,
                                            ape=config.NET.SWIN.APE,
                                            patch_norm=config.NET.SWIN.PATCH_NORM,
                                            use_checkpoint=config.TRAIN.USE_CHECKPOINT)
            encoder_stride = 32
            in_chans = config.NET.SWIN.IN_CHANS
            patch_size = config.NET.SWIN.PATCH_SIZE
            net = SimMIM(config=config.NET.SIMMIM,
                         encoder=encoder, 
                         encoder_stride=encoder_stride, 
                         in_chans=in_chans, 
                         patch_size=patch_size)
        elif net_name == 'simmimv2':
            encoder = SwinTransformerV2ForSimMIM(img_size=config.AUG.CROP_SIZE,
                                                patch_size=config.NET.SWINV2.PATCH_SIZE,
                                                in_chans=config.NET.SWINV2.IN_CHANS,
                                                num_classes=0,
                                                embed_dim=config.NET.SWINV2.EMBED_DIM,
                                                depths=config.NET.SWINV2.DEPTHS,
                                                num_heads=config.NET.SWINV2.NUM_HEADS,
                                                window_size=config.NET.SWINV2.WINDOW_SIZE,
                                                mlp_ratio=config.NET.SWINV2.MLP_RATIO,
                                                qkv_bias=config.NET.SWINV2.QKV_BIAS,
                                                drop_rate=config.NET.DROP_RATE,
                                                drop_path_rate=config.NET.DROP_PATH_RATE,
                                                ape=config.NET.SWINV2.APE,
                                                patch_norm=config.NET.SWINV2.PATCH_NORM,
                                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
            encoder_stride = 32
            in_chans = config.NET.SWINV2.IN_CHANS
            patch_size = config.NET.SWINV2.PATCH_SIZE
            net = SimMIM(config=config.NET.SIMMIM,
                         encoder=encoder, 
                         encoder_stride=encoder_stride, 
                         in_chans=in_chans, 
                         patch_size=patch_size)
    elif net_type == 'dlinknet':
        if net_name == 'dlinknet34':
            net = DlinkNet34(num_classes=config.DATA.NUM_CLASSES)
        if net_name == 'dlinknet50':
            net = DlinkNet50(num_classes=config.DATA.NUM_CLASSES)
        if net_name == 'dlinknet101':
            net = DlinkNet101(num_classes=config.DATA.NUM_CLASSES)
    elif net_type == 'unet':
        if net_name == 'unet':
            net = Unet(num_classes=config.DATA.NUM_CLASSES)
    else:
        raise NotImplementedError(f"Unkown net: {net_type}")
    return net


def build_loss(config):
    if config.TRAIN.LOSS.NAME == 'ce':
        loss_func = CrossEntropyLoss(size_average=config.TRAIN.LOSS.IS_AVERAGE,
                                     ignore_index=config.TRAIN.LOSS.IGNORE_INDEX)
    elif config.TRAIN.LOSS.NAME == 'bce':
        loss_func = BCELoss(size_average=config.TRAIN.LOSS.IS_AVERAGE,
                            ignore_index=config.TRAIN.LOSS.IGNORE_INDEX)
    elif config.TRAIN.LOSS.NAME == 'focal':
        loss_func = FocalLoss(alpha=config.TRAIN.LOSS.ALPHA,
                              gamma=config.TRAIN.LOSS.GAMMA,
                              size_average=config.TRAIN.LOSS.IS_AVERAGE,
                              ignore_index=config.TRAIN.LOSS.IGNORE_INDEX)
    elif config.TRAIN.LOSS.NAME == 'dice':
        loss_func = DiceLoss(size_average=config.TRAIN.LOSS.IS_AVERAGE,
                             ignore_index=config.TRAIN.LOSS.IGNORE_INDEX)
    elif config.TRAIN.LOSS.NAME == 'dicebce':
        loss_func = DiceBCELoss(a=config.TRAIN.LOSS.A,
                                b=config.TRAIN.LOSS.B,
                                size_average=config.TRAIN.LOSS.IS_AVERAGE,
                                ignore_index=config.TRAIN.LOSS.IGNORE_INDEX)
    elif config.TRAIN.LOSS.NAME == 'softiou':
        loss_func = SoftIoULoss(size_average=config.TRAIN.LOSS.IS_AVERAGE,
                                ignore_index=config.TRAIN.LOSS.IGNORE_INDEX)
    else:
        raise NotImplementedError(f"Unkown loss: {config.TRAIN.LOSS.NAME}")
    return loss_func


def build_optimizer(config, net):
    if config.TRAIN.OPTIMIZER.NAME == 'adam':
        optimizer = torch.optim.Adam([{'params': net.parameters(), 'initial_lr': config.TRAIN.LEARNING_RATE}], 
                                     lr=config.TRAIN.LEARNING_RATE, 
                                     betas=config.TRAIN.OPTIMIZER.BETAS, 
                                     weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, 
                                     eps=config.TRAIN.OPTIMIZER.EPS)
    elif config.TRAIN.OPTIMIZER.NAME == 'sgd':
        optimizer = torch.optim.SGD([{'params': net.parameters(), 'initial_lr': config.TRAIN.LEARNING_RATE}], 
                                     lr=config.TRAIN.LEARNING_RATE, 
                                     weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY,
                                     momentum=config.TRAIN.OPTIMIZER.MOMENTUM)
    elif config.TRAIN.OPTIMIZER.NAME == 'adamw':
        optimizer = torch.optim.AdamW([{'params': net.parameters(), 'initial_lr': config.TRAIN.LEARNING_RATE}], 
                                      lr=config.TRAIN.LEARNING_RATE, 
                                      betas=config.TRAIN.OPTIMIZER.BETAS, 
                                      weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY,
                                      eps=config.TRAIN.OPTIMIZER.EPS)
    elif config.TRAIN.OPTIMIZER.NAME == 'rmsprop':
        optimizer = torch.optim.RMSprop([{'params': net.parameters(), 'initial_lr': config.TRAIN.LEARNING_RATE}], 
                                        lr=config.TRAIN.LEARNING_RATE, 
                                        momentum=config.TRAIN.OPTIMIZER.MOMENTUM, 
                                        weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY,
                                        eps=config.TRAIN.OPTIMIZER.EPS)
    else:
        raise NotImplementedError(f"Unkown optimizer: {config.TRAIN.OPTIMIZER.NAME}")
    return optimizer


def build_scheduler(config, optimizer, last_epoch):
    if config.TRAIN.LR_SCHEDULER.NAME =='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS, 
                                                    gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE, 
                                                    last_epoch=last_epoch)
    elif config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                               T_max=config.TRAIN.LR_SCHEDULER.T_MAX, 
                                                               eta_min=config.TRAIN.LR_SCHEDULER.ETA_MIN, 
                                                               last_epoch=last_epoch)
    elif config.TRAIN.LR_SCHEDULER.NAME =='multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                         milestones=config.TRAIN.LR_SCHEDULER.MULTISTEPS, 
                                                         gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE, 
                                                         last_epoch=last_epoch)
    elif config.TRAIN.LR_SCHEDULER.NAME == 'cosinewarm':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         T_0=config.TRAIN.LR_SCHEDULER.T_MAX, 
                                                                         T_mult=config.TRAIN.LR_SCHEDULER.T_MULT, 
                                                                         eta_min=config.TRAIN.LR_SCHEDULER.ETA_MIN, 
                                                                         last_epoch=last_epoch)
    else:
        raise NotImplementedError(f"Unkown scheduler: {config.TRAIN.LR_SCHEDULER.NAME}")
    return scheduler


def build_transform(cfgs, is_aug):
    t = ExtCompose([   
        ExtToTensor()])
    if is_aug:
        t.append(ExtRandomHorizontalFlip())
        t.append(ExtRandomVerticalFlip())
        t.append(ExtRandomResizedCrop(size=cfgs.AUG.CROP_SIZE, 
                                      scale=(cfgs.AUG.CROP_PER, 1), 
                                      ratio=(1 - cfgs.AUG.RESIZE_RATIO, 1 + cfgs.AUG.RESIZE_RATIO)))
        t.append(ExtColorJitter(brightness=cfgs.AUG.INTENSITY, 
                                contrast=cfgs.AUG.CONTRAST, 
                                saturation=cfgs.AUG.SATURATION, 
                                hue=cfgs.AUG.HUE))
    return t