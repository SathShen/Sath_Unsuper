# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
import time
from yacs.config import CfgNode as CN
import argparse

# <editor-fold desc="base config setting">
base_cfg = CN()
# Base config files
base_cfg.BASE = ['']

# -----------------------------------------------------------------------------
# training misc
# -----------------------------------------------------------------------------
base_cfg.IS_CMD = False
base_cfg.CFG_PATH = None
base_cfg.CFG_NOTE = 'default'
base_cfg.PRETRAIN_PATH = None

base_cfg.DEVICE_IDS = [0]
base_cfg.IS_FP16 = False
base_cfg.SAVE_FREQ = 10
base_cfg.NUM_EPOCHS = 100
base_cfg.NUM_ITERS_PER_EPOCH = 0
base_cfg.IS_USE_CHECKPOINT = False
base_cfg.CLIP_GRAD = 3.0
base_cfg.FREEZE_LAST_LAYER_EPOCHS = 1

# evaluate
base_cfg.IS_EVAL = False
base_cfg.IS_SAVE_PRED = False

# do not set
base_cfg.CFG_DIR = './Configs'
base_cfg.LOG_DIR = './Logs'
base_cfg.OUTPUT = './Configs'
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
base_cfg.DATA = CN()
# Path to dataset, could be overwritten by command line argument
base_cfg.DATA.TRAIN_DATA_PATH = ''
base_cfg.DATA.VALID_DATA_PATH = ''
base_cfg.DATA.TEST_DATA_PATH = ''
base_cfg.DATA.CLASS_LIST = []
base_cfg.DATA.NUM_WORKERS = 4
base_cfg.DATA.BATCH_SIZE = 4

base_cfg.DATA.NUM_CLASSES = 0
# Interpolation to resize image (random, bilinear, bicubic)
base_cfg.DATA.INTERPOLATION = 'bilinear'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
base_cfg.DATA.PIN_MEMORY = True
# [SimMIM] Mask patch size for MaskGenerator
base_cfg.DATA.MASK_PATCH_SIZE = 32
# [SimMIM] Mask ratio for MaskGenerator
base_cfg.DATA.MASK_RATIO = 0.6

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
base_cfg.AUG = CN()

base_cfg.AUG.IS_AUG = False
# Random crop resize
base_cfg.AUG.CROP_PER = 0.4
base_cfg.AUG.RESIZE_RATIO = 0.3
base_cfg.AUG.CROP_SIZE = 512
# Color jitter factor
base_cfg.AUG.INTENSITY = 0.4
base_cfg.AUG.HUE = 0.2
base_cfg.AUG.SATURATION = 0.3
base_cfg.AUG.CONTRAST = 0.3
# Multi-crop for unsupervised learning
base_cfg.AUG.GLOBAL_SCALE = (0.4, 1.)
base_cfg.AUG.LOCAL_NUMBER = 8
base_cfg.AUG.LOCAL_SCALE = (0.05, 0.4)

# -----------------------------------------------------------------------------
# Net settings
# -----------------------------------------------------------------------------
base_cfg.NET = CN()
# Net name
base_cfg.NET.NAME = 'swin'
# Dropout rate
base_cfg.NET.DROP_RATE = 0.0
# Drop path rate
base_cfg.NET.DROP_PATH_RATE = 0.1
base_cfg.NET.PATCH_SIZE = 4
base_cfg.NET.OUT_DIM = 65536


# Swin Transformer parameters
# img_size要被patch_size整除, 除出来的patch_solution要被window_size整除
base_cfg.NET.SWIN = CN()
base_cfg.NET.SWIN.IN_CHANS = 3                  # num of input channel
base_cfg.NET.SWIN.EMBED_DIM = 96                # embedding dimension
base_cfg.NET.SWIN.DEPTHS = [2, 2, 6, 2]         # num of blocks in stages
base_cfg.NET.SWIN.NUM_HEADS = [3, 6, 12, 24]    # multi-heads
base_cfg.NET.SWIN.WINDOW_SIZE = 7               # size of window
base_cfg.NET.SWIN.MLP_RATIO = 4.                # mlp hidden layers ratio
base_cfg.NET.SWIN.QKV_BIAS = True
base_cfg.NET.SWIN.QK_SCALE = None
base_cfg.NET.SWIN.APE = False
base_cfg.NET.SWIN.PATCH_NORM = True

# Swin Transformer V2 parameters
base_cfg.NET.SWINV2 = CN()
base_cfg.NET.SWINV2.PATCH_SIZE = 4
base_cfg.NET.SWINV2.IN_CHANS = 3
base_cfg.NET.SWINV2.EMBED_DIM = 96
base_cfg.NET.SWINV2.DEPTHS = [2, 2, 6, 2]
base_cfg.NET.SWINV2.NUM_HEADS = [3, 6, 12, 24]
base_cfg.NET.SWINV2.WINDOW_SIZE = 7
base_cfg.NET.SWINV2.MLP_RATIO = 4.
base_cfg.NET.SWINV2.QKV_BIAS = True
base_cfg.NET.SWINV2.APE = False
base_cfg.NET.SWINV2.PATCH_NORM = True
base_cfg.NET.SWINV2.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]

# SwinV2UNet parameters
base_cfg.NET.SWINV2UNET = CN()
base_cfg.NET.SWINV2UNET.PATCH_SIZE = 4
base_cfg.NET.SWINV2UNET.IN_CHANS = 3
base_cfg.NET.SWINV2UNET.EMBED_DIM = 96
base_cfg.NET.SWINV2UNET.DEPTHS = [2, 2, 6, 2]
base_cfg.NET.SWINV2UNET.NUM_HEADS = [3, 6, 12, 24]
base_cfg.NET.SWINV2UNET.WINDOW_SIZE = 7
base_cfg.NET.SWINV2UNET.MLP_RATIO = 4.
base_cfg.NET.SWINV2UNET.QKV_BIAS = True
base_cfg.NET.SWINV2UNET.APE = False
base_cfg.NET.SWINV2UNET.PATCH_NORM = True
base_cfg.NET.SWINV2UNET.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]

# Swin Transformer MoE parameters
base_cfg.NET.SWIN_MOE = CN()
base_cfg.NET.SWIN_MOE.PATCH_SIZE = 4
base_cfg.NET.SWIN_MOE.IN_CHANS = 3
base_cfg.NET.SWIN_MOE.EMBED_DIM = 96
base_cfg.NET.SWIN_MOE.DEPTHS = [2, 2, 6, 2]
base_cfg.NET.SWIN_MOE.NUM_HEADS = [3, 6, 12, 24]
base_cfg.NET.SWIN_MOE.WINDOW_SIZE = 7
base_cfg.NET.SWIN_MOE.MLP_RATIO = 4.
base_cfg.NET.SWIN_MOE.QKV_BIAS = True
base_cfg.NET.SWIN_MOE.QK_SCALE = None
base_cfg.NET.SWIN_MOE.APE = False
base_cfg.NET.SWIN_MOE.PATCH_NORM = True
base_cfg.NET.SWIN_MOE.MLP_FC2_BIAS = True
base_cfg.NET.SWIN_MOE.INIT_STD = 0.02
base_cfg.NET.SWIN_MOE.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
base_cfg.NET.SWIN_MOE.MOE_BLOCKS = [[-1], [-1], [-1], [-1]]
base_cfg.NET.SWIN_MOE.NUM_LOCAL_EXPERTS = 1
base_cfg.NET.SWIN_MOE.TOP_VALUE = 1
base_cfg.NET.SWIN_MOE.CAPACITY_FACTOR = 1.25
base_cfg.NET.SWIN_MOE.COSINE_ROUTER = False
base_cfg.NET.SWIN_MOE.NORMALIZE_GATE = False
base_cfg.NET.SWIN_MOE.USE_BPR = True
base_cfg.NET.SWIN_MOE.IS_GSHARD_LOSS = False
base_cfg.NET.SWIN_MOE.GATE_NOISE = 1.0
base_cfg.NET.SWIN_MOE.COSINE_ROUTER_DIM = 256
base_cfg.NET.SWIN_MOE.COSINE_ROUTER_INIT_T = 0.5
base_cfg.NET.SWIN_MOE.MOE_DROP = 0.0
base_cfg.NET.SWIN_MOE.AUX_LOSS_WEIGHT = 0.01

# Swin MLP parameters
base_cfg.NET.SWIN_MLP = CN()
base_cfg.NET.SWIN_MLP.PATCH_SIZE = 4
base_cfg.NET.SWIN_MLP.IN_CHANS = 3
base_cfg.NET.SWIN_MLP.EMBED_DIM = 96
base_cfg.NET.SWIN_MLP.DEPTHS = [2, 2, 6, 2]
base_cfg.NET.SWIN_MLP.NUM_HEADS = [3, 6, 12, 24]
base_cfg.NET.SWIN_MLP.WINDOW_SIZE = 7
base_cfg.NET.SWIN_MLP.MLP_RATIO = 4.
base_cfg.NET.SWIN_MLP.APE = False
base_cfg.NET.SWIN_MLP.PATCH_NORM = True


# Net type 
base_cfg.NET.TYPE = 'swin'
# [SimMIM] Norm target during training
base_cfg.NET.SIMMIM = CN()
base_cfg.NET.SIMMIM.NORM_TARGET = CN()
base_cfg.NET.SIMMIM.NORM_TARGET.ENABLE = False
base_cfg.NET.SIMMIM.NORM_TARGET.PATCH_SIZE = 47

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
base_cfg.TRAIN = CN()
base_cfg.TRAIN.START_EPOCH = 0

base_cfg.TRAIN.LEARNING_RATE = 2e-3
base_cfg.TRAIN.USE_CHECKPOINT = True

base_cfg.TRAIN.LOSS = CN()
base_cfg.TRAIN.LOSS.NAME = 'focal'
base_cfg.TRAIN.LOSS.IS_AVERAGE = True
base_cfg.TRAIN.LOSS.IGNORE_INDEX = -100
# [DiceBCE] a, b
base_cfg.TRAIN.LOSS.A = 0.8
base_cfg.TRAIN.LOSS.B = 0.2
# [Focal] alpha, gamma
base_cfg.TRAIN.LOSS.ALPHA = 0.25
base_cfg.TRAIN.LOSS.GAMMA = 2.


# Optimizer
base_cfg.TRAIN.OPTIMIZER = CN()
base_cfg.TRAIN.OPTIMIZER.NAME = 'adamw'
base_cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY = 0.0005
# Optimizer Epsilon
base_cfg.TRAIN.OPTIMIZER.EPS = 2e-5
# Adam Optimizer Betas
base_cfg.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
base_cfg.TRAIN.OPTIMIZER.MOMENTUM = 0.9


# LR scheduler
base_cfg.TRAIN.LR_SCHEDULER = CN()
base_cfg.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
base_cfg.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
base_cfg.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.5
# warmup_prefix used in CosineLRScheduler
base_cfg.TRAIN.LR_SCHEDULER.T_MAX = 30
base_cfg.TRAIN.LR_SCHEDULER.T_MULT = 2
base_cfg.TRAIN.LR_SCHEDULER.ETA_MIN = 2e-5
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
base_cfg.TRAIN.LR_SCHEDULER.GAMMA = 0.1   # ???
base_cfg.TRAIN.LR_SCHEDULER.MULTISTEPS = []



# [SimMIM] Layer decay for fine-tuning
base_cfg.TRAIN.LAYER_DECAY = 1.0
# MoE
base_cfg.TRAIN.MOE = CN()
# Only save model on master device
base_cfg.TRAIN.MOE.SAVE_MASTER = False

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
base_cfg.TEST = CN()
base_cfg.TEST.SHUFFLE = False
# </editor-fold>
def get_nettype(netname):
    if (netname == 'swin' or netname == 'swinv2' or netname =='swinmoe' or netname == 'swinmlp' or 
        netname == 'swinmlp' or netname == 'swinv2unet'):
        return 'swin'
    if netname == 'simmim' or netname == 'simmimv2':
        return 'simmim'
    if netname == 'unet':
        return 'unet'
    if netname == 'deeplab':
        return 'deeplab'
    if netname == 'dlinknet34' or netname == 'dlinknet50' or netname == 'dlinknet101':
        return 'dlinknet'


def bool_flag(str):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if str.lower() in FALSY_STRINGS:
        return False
    elif str.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(f'=> merge config from {cfg_file}')
    config.merge_from_file(cfg_file)   # 从文件合并到config中
    config.freeze()


def update_config(config, args):
    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False
    
    if _check_args('cfg_path'):
        if args.cfg_path != 'None':
            _update_config_from_file(config, args.cfg_path)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if _check_args('cfg_note'):
        config.CFG_NOTE = args.cfg_note
    if _check_args('pretrain_path'):
        if args.pretrain_path == 'None':
            config.NET.PRETRAIN_PATH = None
            config.TRAIN.START_EPOCH = 0
        else:
            config.NET.PRETRAIN_PATH = args.pretrain_path
            config.TRAIN.START_EPOCH = int(os.path.split(config.NET.PRETRAIN_PATH)[1].split('_')[2][2:]) + 1
            config.NET.NAME = os.path.split(config.NET.PRETRAIN_PATH)[1].split('_')[0].lower()
            config.NET.TYPE = get_nettype(config.NET.NAME)
    if _check_args('device_ids'):
        config.DEVICE_IDS = args.device_ids
    if _check_args('is_eval'):
        config.IS_EVAL = bool_flag(args.is_eval)
    if _check_args('is_save_pred'):
        config.IS_SAVE_PRED = bool_flag(args.is_save_pred)

    if _check_args('train_data_path'):
        config.DATA.TRAIN_DATA_PATH = args.train_data_path
        assert os.path.exists(config.DATA.TRAIN_DATA_PATH), "数据路径不存在，退出程序！"
    if _check_args('valid_data_path'):
        config.DATA.VALID_DATA_PATH = args.valid_data_path
        assert os.path.exists(config.DATA.VALID_DATA_PATH), "数据路径不存在，退出程序！"
    if _check_args('test_data_path'):
        config.DATA.TEST_DATA_PATH = args.test_data_path
        assert os.path.exists(config.DATA.TEST_DATA_PATH), "数据路径不存在，退出程序！"
    if _check_args('class_list'):
        config.DATA.CLASS_LIST = args.class_list
        config.DATA.NUM_CLASSES = len(config.DATA.CLASS_LIST)
    if _check_args('num_workers'):
        config.DATA.NUM_WORKERS = args.num_workers
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size

    if _check_args('is_aug'):
        config.AUG.IS_AUG = bool_flag(args.is_aug)
    if _check_args('aug_size'):
        config.AUG.CROP_SIZE = args.aug_size
    if _check_args('aug_scale'):
        config.AUG.CROP_PER = args.aug_scale
    if _check_args('aug_ratio'):
        config.AUG.RESIZE_RATIO = args.aug_ratio
    if _check_args('aug_intensity'):
        config.AUG.INTENSITY = args.aug_intensity
    if _check_args('aug_hue'):
        config.AUG.HUE = args.aug_hue
    if _check_args('aug_saturation'):
        config.AUG.SATURATION = args.aug_saturation
    if _check_args('aug_contrast'):
        config.AUG.CONTRAST = args.aug_contrast
    
    if _check_args('learning_rate'):
        config.TRAIN.LEARNING_RATE = args.learning_rate
    if _check_args('num_epochs'):
        config.TRAIN.NUM_EPOCHS = args.num_epochs
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = args.use_checkpoint

    if _check_args('net_name'):
        config.NET.NAME = args.net_name.lower()
        config.NET.TYPE = get_nettype(args.net_name.lower())
    if _check_args('net_dropout_rate'):
        config.NET.DROP_RATE = args.net_dropout_rate
    if _check_args('net_dropout_path_rate'):
        config.NET.DROP_PATH_RATE = args.net_dropout_path_rate

    if _check_args('net_swinv2unet_patch_size'):
        config.NET.SWINV2UNET.PATCH_SIZE = args.net_swinv2unet_patch_size
    if _check_args('net_swinv2unet_in_chans'):
        config.NET.SWINV2UNET.IN_CHANS = args.net_swinv2unet_in_chans
    if _check_args('net_swinv2unet_embed_dim'):
        config.NET.SWINV2UNET.EMBED_DIM = args.net_swinv2unet_embed_dim
    if _check_args('net_swinv2unet_depths'):
        config.NET.SWINV2UNET.DEPTHS = args.net_swinv2unet_depths
    if _check_args('net_swinv2unet_num_heads'):
        config.NET.SWINV2UNET.NUM_HEADS = args.net_swinv2unet_num_heads
    if _check_args('net_swinv2unet_window_size'):
        config.NET.SWINV2UNET.WINDOW_SIZE = args.net_swinv2unet_window_size
    if _check_args('net_swinv2unet_mlp_ratio'):
        config.NET.SWINV2UNET.MLP_RATIO = args.net_swinv2unet_mlp_ratio
    if _check_args('net_swinv2unet_qkv_bias'):
        config.NET.SWINV2UNET.QKV_BIAS = args.net_swinv2unet_qkv_bias
    if _check_args('net_swinv2unet_ape'):
        config.NET.SWINV2UNET.APE = args.net_swinv2unet_ape
    if _check_args('net_swinv2unet_patch_norm'):
        config.NET.SWINV2UNET.PATCH_NORM = args.net_swinv2unet_patch_norm
    if _check_args('net_swinv2unet_pretrained_window_sizes'):
        config.NET.SWINV2UNET.PRETRAINED_WINDOW_SIZES = args.net_swinv2unet_pretrained_window_sizes
    
        
    if _check_args('loss_name'):
        config.TRAIN.LOSS.NAME = args.loss_name.lower()
    if _check_args('loss_dicebce_a'):
        config.TRAIN.LOSS.A = args.loss_dicebce_a
    if _check_args('loss_dicebce_b'):
        config.TRAIN.LOSS.B = args.loss_dicebce_b
    if _check_args('loss_focal_alpha'):
        config.TRAIN.LOSS.ALPHA = args.loss_focal_alpha
    if _check_args('loss_focal_gamma'):
        config.TRAIN.LOSS.GAMMA = args.loss_focal_gamma

    if _check_args('optim_name'):
        config.TRAIN.OPTIMIZER.NAME = args.optim_name.lower()
    if _check_args('optim_momentum'):
        config.TRAIN.OPTIMIZER.MOMENTUM = args.optim_momentum
    if _check_args('optim_weight_decay'):
        config.TRAIN.OPTIMIZER.WEIGHT_DECAY = args.optim_weight_decay
    if _check_args('optim_eps'):
        config.TRAIN.OPTIMIZER.EPS = args.optim_eps
    if _check_args('optim_betas'):
        config.TRAIN.OPTIMIZER.BETAS = tuple(args.optim_betas)

    if _check_args('sche_name'):
        config.TRAIN.LR_SCHEDULER.NAME  = args.sche_name.lower()
    if _check_args('sche_step_size'):
        config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS  = args.sche_step_size
    if _check_args('sche_decay_rate'):
        config.TRAIN.LR_SCHEDULER.DECAY_RATE  = args.sche_decay_rate
    if _check_args('sche_T_max'):
        config.TRAIN.LR_SCHEDULER.T_MAX  = args.sche_T_max
    if _check_args('sche_T_mult'):
        config.TRAIN.LR_SCHEDULER.T_MULT  = args.sche_T_mult
    if _check_args('sche_eta_min'):
        config.TRAIN.LR_SCHEDULER.ETA_MIN  = args.sche_eta_min
    if _check_args('sche_milestones'):
        config.TRAIN.LR_SCHEDULER.MULTISTEPS  = args.sche_milestones

    # output folder
    config.OUTPUT = os.path.join(config.CFG_DIR, config.NET.NAME)
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = base_cfg.clone()
    if args:
        update_config(config, args)
    return config


def save_config(config):
    if not os.path.exists(config.OUTPUT):
        os.makedirs(config.OUTPUT)
    if config.IS_EVAL:
        path = f"{config.OUTPUT}/{config.NET.NAME}_eval_{config.CFG_NOTE}_{time.strftime('%y%m%d')}_{time.strftime('%H%M%S')}.yaml"
    else:  
        path = f"{config.OUTPUT}/{config.NET.NAME}_{config.CFG_NOTE}_{time.strftime('%y%m%d')}_{time.strftime('%H%M%S')}.yaml"
    with open(path, "w") as f:
        f.write(config.dump())


def test():
    args = None
    cfg = get_config(args)
    save_config(cfg)


if __name__ == "__main__":
    test()

