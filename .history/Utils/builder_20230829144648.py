# --------------------------------------------------------
# Swin Transformer Net builder
# --------------------------------------------------------
import sys
sys.path.append('./')
import torch
from Networks import *
from Utils.loss import *
import torchvision.transforms as transforms
import numpy as np
from Utils.augmentation import *


class CosineScheduler(object):
    def __init__(self, start_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0, is_restart=False, T_0=5, T_mult=2):
        super().__init__()
        """
        freeze -> warmup -> cosine
        Cosine scheduler with warmup and restarts.
        Args:
            start_value (float): start cosine value of the scheduler
            final_value (float): final value of the scheduler
            total_iters (int): total number of iterations
            warmup_iters (int): number of warmup iterations
            start_warmup_value (float): initial value of the warmup
            freeze_iters (int): number of iterations to freeze the scheduler
            is_restart (bool): whether to restart the scheduler at the end of each cycle
            T_0 (int): number of iterations in the first cycle
            T_mult (int): factor to increase T_0 at each restart
        """
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))
        warmup_schedule = np.linspace(start_warmup_value, start_value, warmup_iters)
        
        if is_restart:
            num_iters = total_iters - warmup_iters - freeze_iters
            iters, T_is = self.get_restart_iters(num_iters, T_0, T_mult)
            schedule = final_value + 0.5 * (start_value - final_value) * (1 + np.cos(np.pi * iters / T_is))
        else:
            iters = np.arange(total_iters - warmup_iters - freeze_iters)
            schedule = final_value + 0.5 * (start_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]
        
    def get_restart_iters(self, num_iters, T_0, T_mult):
        iter_accu = 0
        num_periods = 0
        sub_iters_tuple = ()
        T_i_tuple = () 
        while iter_accu < num_iters:
            T_i = T_mult ** num_periods * T_0      # 2 ^ 0 * 5 = 5, 2 ^ 1 * 5 = 10
            iter_accu += T_i

            num_T = T_i
            if iter_accu > num_iters:
                num_T = T_i - (iter_accu - num_iters)
            sub_T_i = np.arange(num_T)
            sub_T_i.fill(T_i)
            T_i_tuple += (sub_T_i,)          # [5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, ...]
            
            sub_iter = np.arange(num_T)
            sub_iters_tuple += (sub_iter,)
            num_periods += 1
        iters = np.concatenate(sub_iters_tuple)
        T_is = np.concatenate(T_i_tuple)
        return iters, T_is


def build_net(cfg):
    net_type = cfg.NET.TYPE
    net_name = cfg.NET.NAME

    if net_type == 'dino':
        if net_name == 'dinov2':
            net = DinoV2(cfg);
        elif net_name == 'dinov1':
            net = DinoV1(cfg);
    if net_type == 'simmim':
        if net_name == 'simmim':
            encoder = SwinTransformerForSimMIM(img_size=cfg.AUG.CROP_SIZE,
                                            patch_size=cfg.NET.SWIN.PATCH_SIZE,
                                            in_chans=cfg.NET.SWIN.IN_CHANS,
                                            num_classes=0,
                                            embed_dim=cfg.NET.SWIN.EMBED_DIM,
                                            depths=cfg.NET.SWIN.DEPTHS,
                                            num_heads=cfg.NET.SWIN.NUM_HEADS,
                                            window_size=cfg.NET.SWIN.WINDOW_SIZE,
                                            mlp_ratio=cfg.NET.SWIN.MLP_RATIO,
                                            qkv_bias=cfg.NET.SWIN.QKV_BIAS,
                                            qk_scale=cfg.NET.SWIN.QK_SCALE,
                                            drop_rate=cfg.NET.DROP_RATE,
                                            drop_path_rate=cfg.NET.DROP_PATH_RATE,
                                            ape=cfg.NET.SWIN.APE,
                                            patch_norm=cfg.NET.SWIN.PATCH_NORM,
                                            use_checkpoint=cfg.TRAIN.USE_CHECKPOINT)
            encoder_stride = 32
            in_chans = cfg.NET.SWIN.IN_CHANS
            patch_size = cfg.NET.SWIN.PATCH_SIZE
            net = SimMIM(config=cfg.NET.SIMMIM,
                         encoder=encoder, 
                         encoder_stride=encoder_stride, 
                         in_chans=in_chans, 
                         patch_size=patch_size)
        elif net_name == 'simmimv2':
            encoder = SwinTransformerV2ForSimMIM(img_size=cfg.AUG.CROP_SIZE,
                                                patch_size=cfg.NET.SWINV2.PATCH_SIZE,
                                                in_chans=cfg.NET.SWINV2.IN_CHANS,
                                                num_classes=0,
                                                embed_dim=cfg.NET.SWINV2.EMBED_DIM,
                                                depths=cfg.NET.SWINV2.DEPTHS,
                                                num_heads=cfg.NET.SWINV2.NUM_HEADS,
                                                window_size=cfg.NET.SWINV2.WINDOW_SIZE,
                                                mlp_ratio=cfg.NET.SWINV2.MLP_RATIO,
                                                qkv_bias=cfg.NET.SWINV2.QKV_BIAS,
                                                drop_rate=cfg.NET.DROP_RATE,
                                                drop_path_rate=cfg.NET.DROP_PATH_RATE,
                                                ape=cfg.NET.SWINV2.APE,
                                                patch_norm=cfg.NET.SWINV2.PATCH_NORM,
                                                use_checkpoint=cfg.TRAIN.USE_CHECKPOINT)
            encoder_stride = 32
            in_chans = cfg.NET.SWINV2.IN_CHANS
            patch_size = cfg.NET.SWINV2.PATCH_SIZE
            net = SimMIM(config=cfg.NET.SIMMIM,
                         encoder=encoder, 
                         encoder_stride=encoder_stride, 
                         in_chans=in_chans, 
                         patch_size=patch_size)
    else:
        raise NotImplementedError(f"Unkown net: {net_type}")
    return net


def build_loss(cfg):
    if cfg.TRAIN.LOSS.NAME == 'dinov1':
        loss_func = DinoV1Loss(size_average=cfg.TRAIN.LOSS.IS_AVERAGE, 
                               ignore_index=cfg.TRAIN.LOSS.IGNORE_INDEX)
    elif cfg.TRAIN.LOSS.NAME == 'dinov2':
        loss_func = DinoV2Loss(size_average=cfg.TRAIN.LOSS.IS_AVERAGE, 
                               ignore_index=cfg.TRAIN.LOSS.IGNORE_INDEX)
    else:
        raise NotImplementedError(f"Unkown loss: {cfg.TRAIN.LOSS.NAME}")
    return loss_func


def build_optimizer(cfg, net):
    if cfg.TRAIN.OPTIMIZER.NAME == 'adam':
        optimizer = torch.optim.Adam([{'params': net.parameters(), 'initial_lr': cfg.TRAIN.LEARNING_RATE}], 
                                     lr=cfg.TRAIN.LEARNING_RATE, 
                                     betas=cfg.TRAIN.OPTIMIZER.BETAS, 
                                     weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY, 
                                     eps=cfg.TRAIN.OPTIMIZER.EPS)
    elif cfg.TRAIN.OPTIMIZER.NAME == 'sgd':
        optimizer = torch.optim.SGD([{'params': net.parameters(), 'initial_lr': cfg.TRAIN.LEARNING_RATE}], 
                                     lr=cfg.TRAIN.LEARNING_RATE, 
                                     weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
                                     momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM)
    elif cfg.TRAIN.OPTIMIZER.NAME == 'adamw':
        optimizer = torch.optim.AdamW([{'params': net.parameters(), 'initial_lr': cfg.TRAIN.LEARNING_RATE}], 
                                      lr=cfg.TRAIN.LEARNING_RATE, 
                                      betas=cfg.TRAIN.OPTIMIZER.BETAS, 
                                      weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
                                      eps=cfg.TRAIN.OPTIMIZER.EPS)
    elif cfg.TRAIN.OPTIMIZER.NAME == 'rmsprop':
        optimizer = torch.optim.RMSprop([{'params': net.parameters(), 'initial_lr': cfg.TRAIN.LEARNING_RATE}], 
                                        lr=cfg.TRAIN.LEARNING_RATE, 
                                        momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM, 
                                        weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
                                        eps=cfg.TRAIN.OPTIMIZER.EPS)
    else:
        raise NotImplementedError(f"Unkown optimizer: {cfg.TRAIN.OPTIMIZER.NAME}")
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


def build_transform(cfgs):
    if cfgs.AUG.IS_AUG == False:
        trans = transforms.ToTensor()
    else:
        if cfgs.NET.NAME == 'dinov1':
            trans = DinoV1Augmentation(cfgs)
    return trans


def test_CosineScheduler():
    # test CosineScheduler class
    import matplotlib.pyplot as plt
    import numpy as np

    start_value = 0
    final_value = 1
    total_iters = 100
    warmup_iters = 10
    start_warmup_value = 0
    freeze_iters = 0
    is_restart = False
    T_0 = 5
    T_mult = 2
    scheduler = CosineScheduler(start_value, final_value, total_iters, warmup_iters, start_warmup_value, freeze_iters, is_restart, T_0, T_mult)
    x = np.arange(total_iters)
    y = np.zeros(total_iters)
    for i in range(total_iters):
        y[i] = scheduler[i]
    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    test_CosineScheduler()