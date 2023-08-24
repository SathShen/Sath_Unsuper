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
base_cfg.NET.SWIN.IS_QKV_BIAS = True
base_cfg.NET.SWIN.IS_APE = False
base_cfg.NET.SWIN.IS_PATCH_NORM = True

# Dino parameters
base_cfg.NET.DINO = CN()
base_cfg.NET.DINO.IS_NORM_LAST_LAYER = True
base_cfg.NET.DINO.IS_BN_IN_HEAD = False


base_cfg.NET.SWIN.QK_SCALE = None
# Net type 
base_cfg.NET.TYPE = 'swin'
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
base_cfg.OPTIMIZER.WEIGHT_DECAY = 0.0005
# Optimizer Epsilon
base_cfg.OPTIMIZER.EPS = 2e-5
# Adam Optimizer Betas
base_cfg.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
base_cfg.OPTIMIZER.MOMENTUM = 0.9


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