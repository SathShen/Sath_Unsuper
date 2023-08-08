# --------------------------------------------------------
# Swin Transformer Net builder
# --------------------------------------------------------
import torch
from Networks import *
from Utils.loss import *
import torch.nn as nn
from Utils.augmentation import *
from Networks import *

def build_model(args, only_teacher=False, img_size=224):
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)

def build_net(config):
    net_type = config.NET.TYPE
    net_name = config.NET.NAME

    if net_type == 'dino':
        if net_name == 'dinov2':
            net = DinoV2(config);
        # elif net_name == 'dinov1':
        #     net = DINOv1(config);
    if net_type == 'simmim':
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
    else:
        raise NotImplementedError(f"Unkown net: {net_type}")
    return net


def build_loss(config):
    if config.TRAIN.LOSS.NAME == 'dinov2':
        loss_func = DinoV2Loss(size_average=config.TRAIN.LOSS.IS_AVERAGE,
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


def build_lrScheduler(config, optimizer, last_epoch):
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