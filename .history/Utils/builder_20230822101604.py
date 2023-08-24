# --------------------------------------------------------
# Swin Transformer Net builder
# --------------------------------------------------------
import torch
from Networks import *
from Utils.loss import *
import torchvision.transforms as transforms
import numpy as np


class CosineScheduler(object):
    def __init__(self, max_value, min_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0, is_restart=False, T_0=5, T_mult=2):
        super().__init__()
        """
        Cosine scheduler with warmup and restarts.
        Args:
            max_value (float): max value of the scheduler
            min_value (float): min value of the scheduler
            total_iters (int): total number of iterations
            warmup_iters (int): number of warmup iterations
            start_warmup_value (float): initial value of the warmup
            freeze_iters (int): number of iterations to freeze the scheduler
            is_restart (bool): whether to restart the scheduler at the end of each cycle
            T_0 (int): number of iterations in the first cycle
            T_mult (int): factor to increase T_0 at each restart
        """
        self.final_value = min_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))
        warmup_schedule = np.linspace(start_warmup_value, max_value, warmup_iters)
        
        if is_restart:
            iters = np.arange(total_iters - warmup_iters - freeze_iters)
            T_i = T_mult ** i * T_0
            schedule = min_value + 0.5 * (max_value - min_value) * (1 + np.cos(np.pi * iters / T_0))
            schedule = min_value + 0.5 * (max_value - min_value) * (1 + np.cos(np.pi * iters / len(iters)))
        else:
            iters = np.arange(total_iters - warmup_iters - freeze_iters)
            schedule = min_value + 0.5 * (max_value - min_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


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
        elif net_name == 'dinov1':
            net = DinoV1(config);
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


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_scheduler = CosineScheduler(**lr)
    wd_scheduler = CosineScheduler(**wd)
    momentum_scheduler = CosineScheduler(**momentum)
    teacher_temp_scheduler = CosineScheduler(**teacher_temp)
    last_layer_lr_scheduler = CosineScheduler(**lr)

    last_layer_lr_scheduler.schedule[:cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH] = 0  # mimicking the original schedules

    return (
        lr_scheduler,
        wd_scheduler,
        momentum_scheduler,
        teacher_temp_scheduler,
        last_layer_lr_scheduler,
    )


def build_transform(cfgs, is_aug):
    t = transforms.Compose([  
        transforms.ToTensor()])
    if is_aug:
        t.append(transforms.RandomHorizontalFlip())
        t.append(transforms.RandomVerticalFlip())
        t.append(transforms.RandomResizedCrop(size=cfgs.AUG.CROP_SIZE, 
                                      scale=(cfgs.AUG.CROP_PER, 1), 
                                      ratio=(1 - cfgs.AUG.RESIZE_RATIO, 1 + cfgs.AUG.RESIZE_RATIO)))
        t.append(transforms.ColorJitter(brightness=cfgs.AUG.INTENSITY, 
                                contrast=cfgs.AUG.CONTRAST, 
                                saturation=cfgs.AUG.SATURATION, 
                                hue=cfgs.AUG.HUE))
    return t