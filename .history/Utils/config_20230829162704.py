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
base_cfg.CLIP_GRAD = 3.0
base_cfg.FREEZE_LAST_LAYER_EPOCHS = 1

# evaluate
base_cfg.IS_EVAL = False
base_cfg.IS_SAVE_PRED = False

# no need to set
base_cfg.START_EPOCH = 0
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
base_cfg.AUG.CROP_SIZE = 448  # for the input size = 512

# Color jitter factor
base_cfg.AUG.INTENSITY = 0.4
base_cfg.AUG.HUE = 0.2
base_cfg.AUG.SATURATION = 0.3
base_cfg.AUG.CONTRAST = 0.3
# Multi-crop for unsupervised learning
base_cfg.AUG.GLOBAL_SCALE = (0.4, 1.)
base_cfg.AUG.NUM_LOCAL = 8
base_cfg.AUG.LOCAL_SCALE = (0.05, 0.4)
base_cfg.AUG.LOCAL_CROP_SIZE = 192

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
base_cfg.NET.PATCH_SIZE = 16
base_cfg.NET.EMBED_DIM = 192
base_cfg.NET.OUT_DIM = 65536


# Swin Transformer parameters
# img_size要被patch_size整除, 除出来的patch_solution要被window_size整除
base_cfg.NET.SWIN = CN()
base_cfg.NET.SWIN.IN_CHANS = 3                  # num of input channel
base_cfg.NET.SWIN.DEPTHS = [2, 2, 6, 2]         # num of blocks in stages
base_cfg.NET.SWIN.NUM_HEADS = [3, 6, 12, 24]    # multi-heads
base_cfg.NET.SWIN.WINDOW_SIZE = 7               # size of window
base_cfg.NET.SWIN.MLP_RATIO = 4.                # mlp hidden layers ratio
base_cfg.NET.SWIN.IS_QKV_BIAS = True
base_cfg.NET.SWIN.IS_APE = False
base_cfg.NET.SWIN.IS_PATCH_NORM = True

# Dino parameters
base_cfg.NET.DINO = CN()
base_cfg.NET.DINO.IS_NORM_LAST_LAYER = True
base_cfg.NET.DINO.IS_BN_IN_HEAD = False


base_cfg.NET.SWIN.QK_SCALE = None
# Net type 
base_cfg.NET.TYPE = ''
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
# [SimMIM] Norm target during training
base_cfg.NET.SIMMIM = CN()
base_cfg.NET.SIMMIM.NORM_TARGET = CN()
base_cfg.NET.SIMMIM.NORM_TARGET.ENABLE = False
base_cfg.NET.SIMMIM.NORM_TARGET.PATCH_SIZE = 47

# -----------------------------------------------------------------------------
# Optimization settings
# -----------------------------------------------------------------------------
base_cfg.LOSS = CN()
base_cfg.LOSS.NAME = 'focal'
base_cfg.LOSS.IS_AVERAGE = True
base_cfg.LOSS.IGNORE_INDEX = -100
# [DiceBCE] a, b
base_cfg.LOSS.A = 0.8
base_cfg.LOSS.B = 0.2
# [Focal] alpha, gamma
base_cfg.LOSS.ALPHA = 0.25
base_cfg.LOSS.GAMMA = 2.


# Optimizer
base_cfg.OPTIMIZER = CN()
base_cfg.OPTIMIZER.NAME = 'adamw'
# SGD momentum
base_cfg.OPTIMIZER.MOMENTUM = 0.9
# Optimizer Epsilon
base_cfg.OPTIMIZER.EPS = 2e-5
# Adam Optimizer Betas
base_cfg.OPTIMIZER.BETAS = (0.9, 0.999)



# lr scheduler setting
base_cfg.LR_SCHEDULER = CN()
base_cfg.LR_SCHEDULER.LEARNING_RATE = 2e-3
base_cfg.LR_SCHEDULER.FINAL_VALUE = 1e-7
base_cfg.LR_SCHEDULER.WARMUP_ITERS = 0
base_cfg.LR_SCHEDULER.WARMUP_VALUE = 1e-6
base_cfg.LR_SCHEDULER.FREEZE_ITERS = 0
base_cfg.LR_SCHEDULER.IS_RESTART = True
base_cfg.LR_SCHEDULER.T_0 = 10
base_cfg.LR_SCHEDULER.T_MULT = 1

# weight decay scheduler setting
base_cfg.WD_SCHEDULER = CN()
base_cfg.WD_SCHEDULER.WEIGHT_DECAY = 0.04
base_cfg.WD_SCHEDULER.FINAL_VALUE = 0.4
base_cfg.WD_SCHEDULER.WARMUP_ITERS = 0
base_cfg.WD_SCHEDULER.WARMUP_VALUE = 0.
base_cfg.WD_SCHEDULER.FREEZE_ITERS = 0
base_cfg.WD_SCHEDULER.IS_RESTART = False
base_cfg.WD_SCHEDULER.T_0 = 0
base_cfg.WD_SCHEDULER.T_MULT = 0

# teacher momentum scheduler setting
base_cfg.TM_SCHEDULER = CN()
base_cfg.TM_SCHEDULER.TEACHER_MOMENTUM = 0.996
base_cfg.TM_SCHEDULER.FINAL_VALUE = 1.
base_cfg.TM_SCHEDULER.WARMUP_ITERS = 0
base_cfg.TM_SCHEDULER.WARMUP_VALUE = 0.
base_cfg.TM_SCHEDULER.FREEZE_ITERS = 0
base_cfg.TM_SCHEDULER.IS_RESTART = False
base_cfg.TM_SCHEDULER.T_0 = 0
base_cfg.TM_SCHEDULER.T_MULT = 0

base_cfg.STUDENT_TEMP = 0.1
# teacher temperature scheduler setting\
base_cfg.TT_SCHEDULER = CN()
base_cfg.TT_SCHEDULER.TEACHER_TEMP = 0.04
base_cfg.TT_SCHEDULER.FINAL_VALUE = 0.04
base_cfg.TT_SCHEDULER.WARMUP_ITERS = 30
base_cfg.TT_SCHEDULER.WARMUP_VALUE = 0.
base_cfg.TT_SCHEDULER.FREEZE_ITERS = 0
base_cfg.TT_SCHEDULER.IS_RESTART = False
base_cfg.TT_SCHEDULER.T_0 = 0
base_cfg.TT_SCHEDULER.T_MULT = 0


# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
base_cfg.LR_SCHEDULER.GAMMA = 0.1   # ???
base_cfg.LR_SCHEDULER.MULTISTEPS = []
# [SimMIM] Layer decay for fine-tuning
base_cfg.LAYER_DECAY = 1.0
# MoE
base_cfg.MOE = CN()
# Only save model on master device
base_cfg.MOE.SAVE_MASTER = False

# </editor-fold>
def get_net_type(netname):
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
    print(f'Merge config from {cfg_file}')
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
    # config setting
    if _check_args('is_cmd'):
        config.IS_CMD = bool_flag(args.is_cmd)
    if _check_args('cfg_note'):
        config.CFG_NOTE = args.cfg_note
    if _check_args('pretrain_path'):
        if args.pretrain_path == 'None':
            config.PRETRAIN_PATH = None
            config.START_EPOCH = 0
        else:
            config.PRETRAIN_PATH = args.pretrain_path
            config.START_EPOCH = int(os.path.split(config.PRETRAIN_PATH)[1].split('_')[2][2:]) + 1
            config.NET.NAME = os.path.split(config.PRETRAIN_PATH)[1].split('_')[0].lower()
            config.NET.TYPE = get_net_type(config.NET.NAME)
    # parser.add_argument('--output_path', '-op', type=str, help='output dir to save log, model, cfg...')
    if _check_args('output_path'):
        config.OUTPUT = args.output_path

    # train setting
    if _check_args('device_ids'):
        config.DEVICE_IDS = args.device_ids
    if _check_args('is_fp16'):
        config.IS_FP16 = bool_flag(args.is_fp16)
    if _check_args('save_freq'):
        config.SAVE_FREQ = args.save_freq
    if _check_args('num_epochs'):
        config.NUM_EPOCHS = args.num_epochs
    if _check_args('clip_grad'):
        config.CLIP_GRAD = args.clip_grad
    if _check_args('freeze_last_layer_epochs'):
        config.FREEZE_LAST_LAYER_EPOCHS = args.freeze_last_layer_epochs


    # evaluate setting
    if _check_args('is_eval'):
        config.IS_EVAL = bool_flag(args.is_eval)
    if _check_args('is_save_pred'):
        config.IS_SAVE_PRED = bool_flag(args.is_save_pred)

    # data setting
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

    # Multi-crop setting
    if _check_args('aug_global_scale'):
        config.AUG.GLOBAL_SCALE = args.aug_global_scale
    if _check_args('aug_num_local'):
        config.AUG.NUM_LOCAL = args.aug_num_local
    if _check_args('aug_local_scale'):
        config.AUG.LOCAL_SCALE = args.aug_local_scale
    
    # net setting
    if _check_args('net_name'):
        config.NET.NAME = args.net_name.lower()
        config.NET.TYPE = get_net_type(args.net_name.lower())
    if _check_args('net_dropout_rate'):
        config.NET.DROP_RATE = args.net_dropout_rate
    if _check_args('net_dropout_path_rate'):
        config.NET.DROP_PATH_RATE = args.net_dropout_path_rate
    if _check_args('net_patch_size'):
        config.NET.PATCH_SIZE = args.net_patch_size
    if _check_args('net_embed_dim'):
        config.NET.EMBED_DIM = args.net_embed_dim
    if _check_args('net_out_dim'):
        config.NET.OUT_DIM = args.net_out_dim
    if _check_args('net_num_heads'):
        config.NET.NUM_HEADS = args.net_num_heads
    

     # swin
    if _check_args('net_swin_in_chans'):
        config.NET.SWIN.IN_CHANS = args.net_swin_in_chans
    if _check_args('net_swin_embed_dim'):
        config.NET.SWIN.EMBED_DIM = args.net_swin_embed_dim
    if _check_args('net_swin_depths'):
        config.NET.SWIN.DEPTHS = args.net_swin_depths
    if _check_args('net_swin_num_heads'):
        config.NET.SWIN.NUM_HEADS = args.net_swin_num_heads
    if _check_args('net_swin_window_size'):
        config.NET.SWIN.WINDOW_SIZE = args.net_swin_window_size
    if _check_args('net_swin_mlp_ratio'):
        config.NET.SWIN.MLP_RATIO = args.net_swin_mlp_ratio
    if _check_args('net_swin_is_qkv_bias'):
        config.NET.SWIN.IS_QKV_BIAS = bool_flag(args.net_swin_is_qkv_bias)
    if _check_args('net_swin_is_ape'):
        config.NET.SWIN.IS_APE = bool_flag(args.net_swin_is_ape)
    if _check_args('net_swin_is_patch_norm'):
        config.NET.SWIN.IS_PATCH_NORM = bool_flag(args.net_swin_is_patch_norm)
    
    # dino
    if _check_args('net_dino_is_norm_last_layer'):
        config.NET.DINO.IS_NORM_LAST_LAYER = bool_flag(args.net_dino_is_norm_last_layer)
    if _check_args('net_dino_is_bn_in_head'):
        config.NET.DINO.IS_BN_IN_HEAD = bool_flag(args.net_dino_is_bn_in_head)
    
    # loss setting
    if _check_args('loss_name'):
        config.LOSS.NAME = args.loss_name.lower()
    elif config.NET.NAME == 'dinov1' or config.NET.NAME == 'dinov2':
            config.LOSS.NAME = config.NET.NAME
    if _check_args('loss_is_average'):
        config.LOSS.IS_AVERAGE = bool_flag(args.loss_is_average)
    if _check_args('loss_ignore_index'):
        config.LOSS.IGNORE_INDEX = args.loss_ignore_index
    if _check_args('loss_dicebce_a'):
        config.LOSS.A = args.loss_dicebce_a
    if _check_args('loss_dicebce_b'):
        config.LOSS.B = args.loss_dicebce_b
    if _check_args('loss_focal_alpha'):
        config.LOSS.ALPHA = args.loss_focal_alpha
    if _check_args('loss_focal_gamma'):
        config.LOSS.GAMMA = args.loss_focal_gamma

    # optimizer setting
    if _check_args('optim_name'):
        config.OPTIMIZER.NAME = args.optim_name.lower()
    if _check_args('optim_momentum'):
        config.OPTIMIZER.MOMENTUM = args.optim_momentum
    if _check_args('optim_eps'):
        config.OPTIMIZER.EPS = args.optim_eps
    if _check_args('optim_betas'):
        config.OPTIMIZER.BETAS = args.optim_betas

    # lr scheduler setting
    if _check_args('learning_rate'):
        config.LR_SCHEDULER.LEARNING_RATE = args.learning_rate
    if _check_args('lrs_final_value'):
        config.LR_SCHEDULER.FINAL_VALUE = args.lrs_final_value
    if _check_args('lrs_warmup_iters'):
        config.LR_SCHEDULER.WARMUP_ITERS = args.lrs_warmup_iters
    if _check_args('lrs_warmup_value'):
        config.LR_SCHEDULER.WARMUP_VALUE = args.lrs_warmup_value
    if _check_args('lrs_freeze_iters'):
        config.LR_SCHEDULER.FREEZE_ITERS = args.lrs_freeze_iters
    if _check_args('lrs_is_restart'):
        config.LR_SCHEDULER.IS_RESTART = bool_flag(args.lrs_is_restart)
    if _check_args('lrs_T_0'):
        config.LR_SCHEDULER.T_0 = args.lrs_T_0
    if _check_args('lrs_T_mult'):
        config.LR_SCHEDULER.T_MULT = args.lrs_T_mult
    
    # weight decay scheduler setting
    if _check_args('weight_decay'):
        config.WD_SCHEDULER.WEIGHT_DECAY = args.weight_decay
    if _check_args('wds_final_value'):
        config.WD_SCHEDULER.FINAL_VALUE = args.wds_final_value
    if _check_args('wds_warmup_iters'):
        config.WD_SCHEDULER.WARMUP_ITERS = args.wds_warmup_iters
    if _check_args('wds_warmup_value'):
        config.WD_SCHEDULER.WARMUP_VALUE = args.wds_warmup_value
    if _check_args('wds_freeze_iters'):
        config.WD_SCHEDULER.FREEZE_ITERS = args.wds_freeze_iters
    if _check_args('wds_is_restart'):
        config.WD_SCHEDULER.IS_RESTART = bool_flag(args.wds_is_restart)
    if _check_args('wds_T_0'):
        config.WD_SCHEDULER.T_0 = args.wds_T_0
    if _check_args('wds_T_mult'):
        config.WD_SCHEDULER.T_MULT = args.wds_T_mult

    # teacher momentum scheduler setting
    if _check_args('teacher_momentum'):
        config.TM_SCHEDULER.TEACHER_MOMENTUM = args.teacher_momentum
    if _check_args('tms_final_value'):
        config.TM_SCHEDULER.FINAL_VALUE = args.tms_final_value
    if _check_args('tms_warmup_iters'):
        config.TM_SCHEDULER.WARMUP_ITERS = args.tms_warmup_iters
    if _check_args('tms_warmup_value'):
        config.TM_SCHEDULER.WARMUP_VALUE = args.tms_warmup_value
    if _check_args('tms_freeze_iters'):
        config.TM_SCHEDULER.FREEZE_ITERS = args.tms_freeze_iters
    if _check_args('tms_is_restart'):
        config.TM_SCHEDULER.IS_RESTART = bool_flag(args.tms_is_restart)
    if _check_args('tms_T_0'):
        config.TM_SCHEDULER.T_0 = args.tms_T_0
    if _check_args('tms_T_mult'):
        config.TM_SCHEDULER.T_MULT = args.tms_T_mult

    # student temperature setting
    if _check_args('student_temp'):
        config.STUDENT_TEMP = args.student_temp
    # teacher temperature scheduler setting
    if _check_args('teacher_temp'):
        config.TT_SCHEDULER.TEACHER_TEMP = args.teacher_temp
    if _check_args('tts_final_value'):
        config.TT_SCHEDULER.FINAL_VALUE = args.tts_final_value
    if _check_args('tts_warmup_iters'):
        config.TT_SCHEDULER.WARMUP_ITERS = args.tts_warmup_iters
    if _check_args('tts_warmup_value'):
        config.TT_SCHEDULER.WARMUP_VALUE = args.tts_warmup_value
    if _check_args('tts_freeze_iters'):
        config.TT_SCHEDULER.FREEZE_ITERS = args.tts_freeze_iters
    if _check_args('tts_is_restart'):
        config.TT_SCHEDULER.IS_RESTART = bool_flag(args.tts_is_restart)
    if _check_args('tts_T_0'):
        config.TT_SCHEDULER.T_0 = args.tts_T_0
    if _check_args('tts_T_mult'):
        config.TT_SCHEDULER.T_MULT = args.tts_T_mult

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

