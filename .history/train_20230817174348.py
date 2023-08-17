import argparse
import torch
from tqdm import tqdm
import torch.utils.data
from frame import TrainFrame
from Utils import Timer, Logger, LocalDatasetBuilder
from Utils.config import get_config, save_config


def train(frame, cfgs):
    total_timer = Timer()
    epoch_timer = Timer()
    total_timer.start()

    train_dataset = LocalDatasetBuilder(cfgs)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfgs.DATA.BATCH_SIZE,
                                                     shuffle=True, num_workers=cfgs.DATA.NUM_WORKERS)

    logger = Logger(cfgs.NET.NAME, cfgs.CFG_NOTE)
    logger.log_in(f'Start on device_ids: {frame.device_ids}!', f'{train_dataset.__len__()} examples in training set')
    best_loss = 999

    for epoch in range(cfgs.TRAIN.START_EPOCH, cfgs.TRAIN.NUM_EPOCHS):
        epoch_timer.start()
        train_data_loader_iter = iter(train_data_loader)
        train_epoch_loss = 0
        for imgs, labs in tqdm(train_data_loader_iter, ncols=80):   # imgs: batch_size, channels, H, W
            frame.set_input(imgs, labs)                   # labs: batch_size, num_classes, H, W
            l = frame.optimize()
            train_epoch_loss += l
        train_epoch_loss /= len(train_data_loader_iter)
        frame.validiate(iter(valid_data_loader))

        if epoch == cfgs.TRAIN.START_EPOCH:
            best_loss = train_epoch_loss
            best_train_mIoU = frame.train_metrics.macro_IoU()
            best_valid_mIoU = frame.valid_metrics.macro_IoU()
            frame.save_weights(cfgs.DATA.TRAIN_DATA_PATH, cfgs.NET.NAME, cfgs.CFG_NOTE, epoch, 
                               best_train_mIoU, train_epoch_loss, dataset='train')
            frame.save_weights(cfgs.DATA.TRAIN_DATA_PATH, cfgs.NET.NAME, cfgs.CFG_NOTE, epoch, 
                               best_valid_mIoU, train_epoch_loss, dataset='valid')
        else:
            if train_epoch_loss < best_loss:
                best_loss = train_epoch_loss
            if frame.train_metrics.macro_IoU() > best_train_mIoU:
                best_train_mIoU = frame.train_metrics.macro_IoU()
                frame.save_weights(cfgs.DATA.TRAIN_DATA_PATH, cfgs.NET.NAME, cfgs.CFG_NOTE, epoch, 
                                   best_train_mIoU, train_epoch_loss, dataset='train')
            if frame.valid_metrics.macro_IoU() > best_valid_mIoU:
                best_valid_mIoU = frame.valid_metrics.macro_IoU()
                frame.save_weights(cfgs.DATA.TRAIN_DATA_PATH, cfgs.NET.NAME, cfgs.CFG_NOTE, epoch, 
                                   best_valid_mIoU, train_epoch_loss, dataset='valid')
        epoch_timer.stop()
        logger.log_in(f'epoch: {epoch}, epoch_time: {epoch_timer.get_epochtime()}, '
                      f'train_loss: {train_epoch_loss:.3f}, best_loss: {best_loss:.3f}, lr: {frame.lr:.2e}, '
                      f'train_mIoU: {frame.train_metrics.macro_IoU():.3f}, best_train_mIoU: {best_train_mIoU:.3f}, '
                      f'valid_mIoU: {frame.valid_metrics.macro_IoU():.3f}, best_valid_mIoU: {best_valid_mIoU:.3f}')
        frame.train_metrics.reset()
        frame.valid_metrics.reset()
        frame.update_lr()
        logger.flush()

    total_timer.stop()
    logger.log_in(f'train_time: {epoch_timer.get_sumtime()}, '
                  f'{train_dataset.__len__() * (cfgs.TRAIN.NUM_EPOCHS - cfgs.TRAIN.START_EPOCH) / epoch_timer.sum():.2f}examples/sec, '
                  f'total_time: {total_timer.get_sumtime()}, '
                  f'best_train_mIoU: {best_train_mIoU:.3f}, best_valid_mIoU: {best_valid_mIoU:.3f}', 'Finish!')

def get_parserargs():
    parser = argparse.ArgumentParser(description='Train the network on images using Pytorch')

    # ==========training setting==========
    parser.add_argument('--on_cmd', '-cmd', type=bool, default=False, help='If training on cmd, set on_cmd True')
    parser.add_argument('--cfg_path', '-cfg', type=str, default=None, metavar="CFG", help='path to load a local config file')
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
    parser.add_argument('--cfg_note', '-cn', metavar='CN', type=str, help='note which will be saved in config name')
    parser.add_argument('--pretrain_path', '-pp', metavar='PP', type=str, default=None, help='pretrain model abspath')
    parser.add_argument('--device_ids', '-d', metavar='D', type=int, nargs='+', help='device ids which is training on, first is the major')
    parser.add_argument('--is_eval', '-ev', metavar='EV', type=bool, help='the eval mode or not')

    # ==========data setting==========
    parser.add_argument('--train_data_path', '-tp', metavar='TP', type=str, help='Training dataset abspath')
    parser.add_argument('--valid_data_path', '-vp', metavar='VP', type=str, help='Validation dataset abspath')
    parser.add_argument('--class_list', '-cl', metavar='CL', type=int, nargs='+', help='label classes index list')
    parser.add_argument('--num_workers', '-nw', metavar='NW', type=int, help='number of workers in dataloader')
    parser.add_argument('--batch_size', '-b', metavar='B', type=int, help='Batch size')
    # augmentation setting
    parser.add_argument('--is_aug', '-a', metavar='AUG', type= bool, help='Use augmentations or not')
    parser.add_argument('--aug_size', '-asi', metavar='ASI', type=int, help='Image size that input network, if aug, it will be crop size')
    parser.add_argument('--aug_scale', '-asc', metavar='ASC', type=float, help='min crop scale percentage in one image')
    parser.add_argument('--aug_ratio', '-ar', metavar='AR', type=float, help='max crop scale percentage in one image')
    parser.add_argument('--aug_intensity', '-ai', metavar='AI', type=float, help='intensity of color jitter')
    parser.add_argument('--aug_hue', '-ah', metavar='AH', type=float, help='hue of color jitter')
    parser.add_argument('--aug_saturation', '-asa', metavar='ASA', type=float, help='saturation of color jitter')
    parser.add_argument('--aug_contrast', '-ac', metavar='AC', type=float, help='contrast of color jitter')

    # ==========hyperparams setting==========
    parser.add_argument('--learning_rate', '-lr', metavar='LR', type=float, help='Learning rate')
    parser.add_argument('--num_epochs', '-e', metavar='E', type=int, help='Number of training epochs')
    parser.add_argument('--use_checkpoint', '-uc', metavar='UC', type=bool, help='use checkpoint or not')

    # net setting
    parser.add_argument('--net_name', '-n', metavar='N', type=str, help='Network name, DlinkNet34, DlinkNet50, Unet, swin...')
    parser.add_argument('--net_dropout_rate', '-ndr', metavar='NDR', type=float, help='dropout rate')
    parser.add_argument('--net_dropout_path_rate', '-ndpr', metavar='NDPR', type=float, help='stochastic depth dropout path rate')
    # (swinv2unet)
    parser.add_argument('--net_swinv2unet_patch_size', '-nsups', metavar='NSUPS', type=int, help='patch size of swin')
    parser.add_argument('--net_swinv2unet_in_chans', '-nsuic', metavar='NSUIC', type=int, help='in channels of swin')
    parser.add_argument('--net_swinv2unet_embed_dim', '-nsued', metavar='NSUED', type=int, help='embed dim of swin')
    parser.add_argument('--net_swinv2unet_depths', '-nsudp', metavar='NSUDP', type=int, nargs='+', help='depths of swin')
    parser.add_argument('--net_swinv2unet_num_heads', '-nsunh', metavar='NSUNH', type=int, nargs='+', help='num heads of swin')
    parser.add_argument('--net_swinv2unet_window_size', '-nsuws', metavar='NSUWS', type=int, help='window size of swin')
    parser.add_argument('--net_swinv2unet_mlp_ratio', '-nsumr', metavar='NSUMR', type=float, help='mlp ratio of swin')
    parser.add_argument('--net_swinv2unet_qkv_bias', '-nsuqkb', metavar='NSUQKB', type=bool, help='qkv bias of swin')
    parser.add_argument('--net_swinv2unet_ape', '-nsuap', metavar='NSUAP', type=bool, help='ape of swin')
    parser.add_argument('--net_swinv2unet_patch_norm', '-nsupn', metavar='NSUPN', type=bool, help='patch norm of swin')
    parser.add_argument('--net_swinv2unet_pretrained_window_sizes', '-nsupws', metavar='NSUPWS', type=int, nargs='+', help='pretrained window sizes of swin')

    
    # loss setting
    parser.add_argument('--loss_name', '-l', metavar='L', type=str,
                        help='Loss function name, Dice, DiceBCE, SoftIoU, Focal, CE, BCE...')
    parser.add_argument('--loss_dicebce_a', '-ldba', metavar='LDBA', type=float, help='a of dicebce loss function')
    parser.add_argument('--loss_dicebce_b', '-ldbb', metavar='LDBB', type=float, help='b of dicebce loss function')
    parser.add_argument('--loss_focal_alpha', '-lfa', metavar='LFA', type=float, help='alpha of focal loss function')
    parser.add_argument('--loss_focal_gamma', '-lfg', metavar='LFG', type=float, help='gamma of focal loss function')
    # optimizer setting
    parser.add_argument('--optim_name', '-o', metavar='O', type=str, help='Optimizer name, sgd, adam, rmsprop, adamw...')
    parser.add_argument('--optim_momentum', '- omm', metavar='OMM', type=float, help='beta of momentum vt, some do not need')
    parser.add_argument('--optim_weight_decay', '-owd', metavar='OWD', type=float, help='lambda of weight decay')
    parser.add_argument('--optim_eps', '-oe', metavar='OE', type=float, help='eps of adam, rmsprop, adamw...')
    parser.add_argument('--optim_betas', '-obt', metavar='OBT', type=float, nargs=2, help='betas of adam, adamw...')
    # lr scheduler setting
    parser.add_argument('--sche_name', '-s', metavar='S', type=str,
                        help='Learning rate scheduler name, step, multistep, cosine, cosinewarm...')
    parser.add_argument('--sche_step_size', '-sss', metavar='SSS', type=int, help='step size of step scheduler')
    parser.add_argument('--sche_decay_rate', '-sdc', metavar='SDC', type=float, help='decay rate of step scheduler, etc gamma of stepLR')
    parser.add_argument('--sche_T_max', '-stma', metavar='STMA', type=int, help='T_max of cosinewarm scheduler')
    parser.add_argument('--sche_T_mult', '-stmu', metavar='STMU', type=int, help='T_mult of cosinewarm scheduler')
    parser.add_argument('--sche_eta_min', '-setm', metavar='SETM', type=float, help='eta_min of cosinewarm scheduler')
    parser.add_argument('--sche_milestones', '-sml', metavar='SML', type=list, help='milestones of multistep scheduler')



    # Model parameters
    parser.add_argument('--backbone', 'bb', default='vit_small', type=str, help='Name of architecture to train.')
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_ema', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')


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
    args.use_checkpoint = True
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

    frame = TrainFrame(cfgs)
    train(frame, cfgs)
