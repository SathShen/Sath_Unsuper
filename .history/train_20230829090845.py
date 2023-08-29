import argparse
import torch
from tqdm import tqdm
import torch.utils.data
from pretrain_frame import PretrainFrame
from Utils import Timer, Logger, LocalDatasetBuilder
from Utils.config import get_config, save_config


def train(cfgs, frame, dataset):
    total_timer = Timer()
    epoch_timer = Timer()
    total_timer.start()

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfgs.DATA.BATCH_SIZE,
                                                     shuffle=True, num_workers=cfgs.DATA.NUM_WORKERS)

    logger = Logger(cfgs.NET.NAME, cfgs.CFG_NOTE)
    logger.log_in(f'Start on device_ids: {frame.device_ids}!', f'{train_dataset.__len__()} examples in training set')
    best_loss = 999

    it = cfgs.start_iteration
    for epoch in range(cfgs.TRAIN.START_EPOCH, cfgs.TRAIN.NUM_EPOCHS):
        epoch_timer.start()
        train_data_loader_iter = iter(train_data_loader)
        train_epoch_loss = 0
        for auged_crops_list in tqdm(train_data_loader_iter, ncols=80):   # imgs: batch_size, channels, H, W
            frame.set_input(auged_crops_list)                 
            l = frame.optimize(it)
            train_epoch_loss += l
            it += 1
        train_epoch_loss /= len(train_data_loader_iter)

        # EMA update for the teacher

        if epoch == cfgs.TRAIN.START_EPOCH:
            best_loss = train_epoch_loss
            frame.save_weights(cfgs.DATA.TRAIN_DATA_PATH, cfgs.NET.NAME, cfgs.CFG_NOTE, epoch, train_epoch_loss)
        else:
            if train_epoch_loss < best_loss:
                best_loss = train_epoch_loss
                frame.save_weights(cfgs.DATA.TRAIN_DATA_PATH, cfgs.NET.NAME, cfgs.CFG_NOTE, epoch, train_epoch_loss)
        epoch_timer.stop()
        logger.log_in(f'epoch: {epoch}, epoch_time: {epoch_timer.get_epochtime()}, '
                      f'train_loss: {train_epoch_loss:.3f}, best_loss: {best_loss:.3f}, '
                      f'lr: {frame.lr:.2e}, ')   # FIXME schedulers...wd...momentum...
            
        frame.update_lr()
        logger.flush()

    total_timer.stop()
    logger.log_in(f'train_time: {epoch_timer.get_sumtime()}, '
                  f'{train_dataset.__len__() * (cfgs.TRAIN.NUM_EPOCHS - cfgs.TRAIN.START_EPOCH) / epoch_timer.sum():.2f}examples/sec, '
                  f'total_time: {total_timer.get_sumtime()}, best_loss: {best_loss:.3f}', 'Finish!')

def get_parserargs():
    parser = argparse.ArgumentParser(description='Train the network on images using Pytorch')

    # ==========training misc setting==========
    # config setting
    parser.add_argument('--is_cmd', '-cmd', default=False, help='If training on cmd, set on_cmd True')
    parser.add_argument('--cfg_path', '-cfg', type=str, default=None, metavar="CFG", help='path to load a local config file')
    parser.add_argument('--cfg_note', '-cn', metavar='CN', type=str, help='note which will be saved in config name')
    parser.add_argument('--pretrain_path', '-pp', metavar='PP', type=str, default=None, help='pretrain model abspath')

    # train setting
    parser.add_argument('--device_ids', '-d', metavar='D', type=int, nargs='+', help='device ids which is training on, first is the major')
    parser.add_argument('--is_fp16', '-fp16', metavar='FP16', default=True, help=""" Improves time and memory requirements, but can provoke instability and decay of performance. 
                        We recommend disabling mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--save_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--num_epochs', '-e', metavar='E', type=int, help='Number of training epochs')
    parser.add_argument('--num_iters_per_epoch', '-nit', metavar='NIT', type=int, help="""Number of iterations per epoch, 
                        it equal to len(dataset) / batch_size.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter gradient norm if using gradient clipping. 
                        Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--freeze_last_layer_epochs', '-flle', default=1, type=int, help="""Number of epochs during which we keep the output layer fixed. 
                        Typically doing so during the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
    # ==============data setting=============
    parser.add_argument('--train_data_path', '-tp', metavar='TP', type=str, help='Training dataset abspath')
    parser.add_argument('--num_workers', '-nw', metavar='NW', type=int, help='number of workers in dataloader')
    parser.add_argument('--batch_size', '-b', metavar='B', type=int, help='Batch size per gpu')

    # ==========augmentation setting=========
    parser.add_argument('--is_aug', '-a', metavar='AUG', help='Use augmentations or not')
    # Random crop resize setting
    parser.add_argument('--aug_size', '-asi', metavar='ASI', type=int, help='Output size in cropresize')
    parser.add_argument('--aug_scale', '-asc', metavar='ASC', type=float, help='Minimum cropping percentage in one image')
    parser.add_argument('--aug_ratio', '-ar', metavar='AR', type=float, help='Maximum height to aspect ratio of crop')
    # Color jitter factor setting
    parser.add_argument('--aug_intensity', '-ai', metavar='AI', type=float, help='intensity of color jitter')
    parser.add_argument('--aug_hue', '-ah', metavar='AH', type=float, help='hue of color jitter')
    parser.add_argument('--aug_saturation', '-asa', metavar='ASA', type=float, help='saturation of color jitter')
    parser.add_argument('--aug_contrast', '-ac', metavar='AC', type=float, help='contrast of color jitter')
    # Multi-crop setting
    parser.add_argument('--aug_global_scale', '-ags', type=float, nargs='+', default=(0.4, 1.), help="""Scale range of the cropped image before resizing, 
                        relatively to the origin image. Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we 
                        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--aug_num_local', '-anl', type=int, default=8, help="""Number of small local views to generate. Set this parameter to 0 to disable
                         multi-crop training. When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--aug_local_scale', 'als', type=float, nargs='+', default=(0.05, 0.4), help="""Scale range of the cropped image before resizing, 
                        relatively to the origin image. Used for small local view cropping of multi-crop.""")

    # ==========net setting==========
    parser.add_argument('--net_name', '-n', metavar='N', type=str, help='Network name, DlinkNet34, DlinkNet50, Unet, swin...')
    parser.add_argument('--net_dropout_rate', '-ndr', metavar='NDR', type=float, help='dropout rate')
    parser.add_argument('--net_dropout_path_rate', '-ndpr', metavar='NDPR', type=float, help='stochastic depth dropout path rate')
    parser.add_argument('--net_patch_size', '-nps', metavar='NPS', default=16, type=int, help="""Using smaller values leads to better performance but requires more memory. 
                        If <16, we recommend disabling mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--net_out_dim', default=65536, type=int, help="""Output dim. For complex and large datasets large values (like 65k) work well.""")
    # swin
    parser.add_argument('--net_swin_in_chans', '-nsuic', metavar='NSUIC', type=int, help='in channels of swin')
    parser.add_argument('--net_swin_embed_dim', '-nsued', metavar='NSUED', type=int, help='embed dim of swin')
    parser.add_argument('--net_swin_depths', '-nsudp', metavar='NSUDP', type=int, nargs='+', help='depths of swin')
    parser.add_argument('--net_swin_num_heads', '-nsunh', metavar='NSUNH', type=int, nargs='+', help='num heads of swin')
    parser.add_argument('--net_swin_window_size', '-nsuws', metavar='NSUWS', type=int, help='window size of swin')
    parser.add_argument('--net_swin_mlp_ratio', '-nsumr', metavar='NSUMR', type=float, help='mlp ratio of swin')
    parser.add_argument('--net_swin_is_qkv_bias', '-nsuiqb', metavar='NSUQKB', help='qkv bias of swin')
    parser.add_argument('--net_swin_is_ape', '-nsuia', metavar='NSUAP', help='ape of swin')
    parser.add_argument('--net_swin_is_patch_norm', '-nsuipn', metavar='NSUPN', help='patch norm of swin')
    parser.add_argument('--net_swin_pretrained_window_sizes', '-nsupws', metavar='NSUPWS', type=int, nargs='+', help='pretrained window sizes of swin')
    # (dino)
    parser.add_argument('--net_dino_is_norm_last_layer', 'ndinll', default=True, help="""Not normalizing leads to better performance but can make the training unstable. 
                        we typically set False with small and True with base.""")
    parser.add_argument('--net_dino_is_bn_in_head', 'ndibin', default=False, help="Whether to use batch normalizations in projection head (Default: False)")
    
    # ==========optimization setting==========
    # loss setting
    parser.add_argument('--loss_name', '-l', metavar='L', type=str,
                        help='Loss function name, Dice, DiceBCE, SoftIoU, Focal, CE, BCE...')
    
    # optimizer setting
    parser.add_argument('--optim_name', '-o', metavar='O', type=str, help='Optimizer name, sgd, adam, rmsprop, adamw...')
    parser.add_argument('--optim_momentum', '- omm', metavar='OMM', type=float, help='beta of optimize momentum vt, some optimizer do not need')
    parser.add_argument('--optim_eps', '-oe', metavar='OE', type=float, help='eps of adam, rmsprop, adamw...')
    parser.add_argument('--optim_betas', '-obt', metavar='OBT', type=float, nargs=2, help='betas of adam, adamw...')

    # lr scheduler setting
    parser.add_argument('--learning_rate', '-lr', metavar='LR', type=float, help='Learning rate (start with max learning rate)')
    parser.add_argument('--lrs_final_value', '-lrsfv', metavar='LRSMV', type=float, help='final value of cosinewarm scheduler(min in cycle)')
    parser.add_argument('--lrs_warmup_iters', '-lrswi', metavar='LRSWI', type=int, help='warmup iters of cosinewarm scheduler')
    parser.add_argument('--lrs_warmup_value', '-lrswv', metavar='LRSSWV', type=float, help='start warmup value of cosinewarm scheduler')
    parser.add_argument('--lrs_freeze_iters', '-lrsfi', metavar='LRSFI', default=0, type=int, help='freeze iters of cosinewarm scheduler')
    parser.add_argument('--lrs_is_restart', '-lrsir', metavar='LRSIR', default=True, help='is restart of cosinewarm scheduler')
    parser.add_argument('--lrs_T_0', '-lrst0', metavar='LRSTMA', type=int, help='T_0 of cosinewarm scheduler if restart')
    parser.add_argument('--lrs_T_mult', '-lrstmu', metavar='LRSTMU', type=int, help='T_mult of cosinewarm scheduler if restart')

    # weight decay scheduler setting
    parser.add_argument('--weight_decay', '-wd', metavar='WD', type=float, default=0.04, help="""Initial value of the weight decay. With ViT, a 
                        smaller value at the beginning of training works well.""")
    parser.add_argument('--wds_final_value', '-wdsfv', metavar='WDSMV', type=float, help='final value of cosinewarm scheduler(max in cycle)')

    # teacher momentum scheduler setting
    parser.add_argument('--teacher_momentum', '-tm', metavar='TM', type=float, default=0.996, type=float, help="""Base EMA parameter for teacher update. 
                        The value is increased to 1 during training with cosine schedule. We recommend setting a higher value with small batches: 
                        for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--tms_final_value', '-tmsfv', metavar='TMSFV', type=float, default=1., help='Final value for the teacher momentum.')
    
    # teacher temperature scheduler setting
    parser.add_argument('--student_temp', '-st', default=0.1, type=float, help="Initial value for the student temperature: 0.1 works well in most cases.")
    parser.add_argument('--teacher_temp', '-tt', default=0.04, type=float, help="""Initial value for the teacher temperature: 0.04 works well in most cases.
                        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--tts_final_value', "-ttsfv", metavar='TTSFV', default=0.04, type=float, help="""Final value (after linear warmup) of the teacher temperature. 
                        For most experiments, anything above 0.07 is unstable. We recommend starting with the default value of 0.04 and 
                        increase this slightly if needed.""")
    parser.add_argument('--tts_warmup_iters', '-ttswi', metavar='TTSWI', type=int, help='warmup iters of cosinewarm scheduler')


    args, unknown = parser.parse_known_args()

    if not args.on_cmd:
        print('init_args in IDE')
        init_args(args)
    config = get_config(args)
    save_config(config)
    return config


def init_args(args):
    # args.cfg_path = r'E:\Sht\Projects\Python\Sath_Super\Configs\dlinknet34\dlinknet34_test_BJ.yaml'
    args.cfg_path = 'None'
    args.cfg_note = 'GIDwater'
    # args.pretrain_path = r'F:\Test_data\BJ\train\dlinknet34_ep6_train_077.params'
    args.pretrain_path = None
    args.device_ids = [0]
    args.is_eval = False

    args.train_data_path = r'F:\RSDL\GID\train'
    args.valid_data_path = r'F:\RSDL\GID\valid'
    # args.train_data_path = r'F:\Test_data\GID_water\train'
    # args.valid_data_path = r'F:\Test_data\GID_water\valid'
    args.class_list = [0, 255]
    args.num_workers = 4
    args.batch_size = 2
    
    args.is_aug = True
    args.aug_size = 512
    args.aug_scale = 0.4
    args.aug_ratio = 0.3
    args.aug_intensity = 0.4
    args.aug_hue = 0.2
    args.aug_saturation = 0.4
    args.aug_contrast = 0.4
    
    args.learning_rate = 2e-4
    args.num_epochs = 100
    args.net_name = 'Swinv2unet'
    args.net_dropout_rate = 0.
    args.net_dropout_path_rate = 0.

    args.net_swinv2unet_patch_size = 8   # img_size要被patch_size整除, 除出来的patch_solution要被window_size整除
    args.net_swinv2unet_in_chans = 3
    args.net_swinv2unet_embed_dim = 192
    args.net_swinv2unet_depths = [2, 2, 6, 2]
    args.net_swinv2unet_num_heads = [3, 6, 12, 24]
    args.net_swinv2unet_window_size = 8
    args.net_swinv2unet_mlp_ratio = 4.
    args.net_swinv2unet_qkv_bias = True
    args.net_swinv2unet_ape = False
    args.net_swinv2unet_patch_norm = True
    args.net_swinv2unet_pretrained_window_sizes = [0, 0, 0, 0]
    
    args.loss_name = 'focal'
    args.loss_dice_a = 0.8
    args.loss_dice_b = 0.2
    args.loss_focal_alpha = 0.25
    args.loss_focal_gamma = 2.
    
    args.optim_name = 'adamw'
    args.optim_momentum = 0.9
    args.optim_weight_decay = 0.0005
    args.optim_eps = 2e-5
    args.optim_betas = (0.9, 0.999)
    
    args.sche_name = 'cosinewarm'
    args.sche_step_size = 3
    args.sche_decay_rate = 0.6
    args.sche_T_max = 10        
    args.sche_T_mult = 2
    args.sche_eta_min = 2e-6
    args.sche_milestones = [10, 30, 70]

if __name__ == '__main__':
    cfgs = get_parserargs()
    frame = PretrainFrame(cfgs)
    train_dataset = LocalDatasetBuilder(cfgs)

    train(cfgs, frame, train_dataset)
