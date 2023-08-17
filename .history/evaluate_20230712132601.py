import torch
import os
from Utils import try_gpu, Timer, Logger, SegmentationMetrics, build_net, build_transform, one_hot_encode
from Utils.dataset import RS_loader, Class2Idx, Idx2Class
from tqdm import tqdm
import argparse
from Utils.config import get_config, save_config
import cv2
import numpy as np


def evaluate(cfgs):
    timer = Timer()
    timer.start()
    if cfgs.IS_SAVE_PRED:
        save_path = f'{cfgs.DATA.TEST_DATA_PATH}/predict'
        os.makedirs(save_path, exist_ok=True)
    mdevice = try_gpu(cfgs.DEVICE_IDX[0])
    net = build_net(cfgs).eval()
    net.load_state_dict(torch.load(cfgs.NET.PRETRAIN_PATH)['model_state_dict'])

    if len(cfgs.DEVICE_IDS) > 1:
        net = torch.nn.DataParallel(net, device_ids=cfgs.DEVICE_IDS)
    net = net.to(mdevice)
    trans = build_transform(cfgs, is_aug=False)
    metrics = SegmentationMetrics(num_classes=cfgs.DATA.NUM_CLASSES)
    logger = Logger(f'{cfgs.NET.NAME}_eval', cfgs.CFG_NOTE)
    
    img_path = f"{cfgs.DATA.TEST_DATA_PATH}/image"
    lab_path = f"{cfgs.DATA.TEST_DATA_PATH}/label"
    img_fullname_list = os.listdir(img_path)
    lab_fullname_list = os.listdir(lab_path)
    img_ext = img_fullname_list[0].split('.')[-1]  # 图片后缀
    lab_ext = lab_fullname_list[0].split('.')[-1]
    img_name_list = list(map(lambda x: x[:-(len(img_ext) + 1)], img_fullname_list))

    logger.log_in(f'Evaluate on {mdevice}!, save predict: {cfgs.IS_SAVE_PRED}', 
                f'{img_fullname_list.__len__()} examples in test set.')

    with torch.no_grad():
        for img_name in tqdm(img_name_list, ncols=80):
            img, lab = RS_loader(img_name, img_path, lab_path, img_ext, lab_ext)
            img_auged, lab_auged = trans(img, lab)    
            img_auged = img_auged.unsqueeze(0).to(mdevice)   # 1 channels h w
            lab_auged = lab_auged.to(mdevice)   # 1 h w
            preds = net(img_auged)             # 1 classes h w
            
            metrics_preds = preds.argmax(dim=1)  # 1 h w 012
            metrics_labs = Class2Idx(lab_auged, cfgs.DATA.CLASS_LIST, mdevice)  # 1 h w 012
            metrics.add_imgs(metrics_preds, metrics_labs)

            if cfgs.IS_SAVE_PRED:
                cv2.imwrite(f'{save_path}/{img_name}.tif', 
                            Idx2Class(metrics_preds.squeeze(0), 
                                      cfgs.DATA.CLASS_LIST, mdevice).cpu().numpy().astype(np.uint8),
                            (int(cv2.IMWRITE_TIFF_COMPRESSION), 1))
    timer.stop()
    logger.log_in(f'evaluate_time: {timer.sum() // 60:.0f}m{timer.sum() % 60:.0f}s, '
                  f'{img_fullname_list.__len__() / timer.sum():.2f}examples/sec',
                  f'===>PA: {metrics.pixel_accuracy():.3f}',
                  f'===>mPrecision: {metrics.macro_precision():.3f}\tmRecall: {metrics.macro_recall():.3f}',
                  f'===>mIoU: {metrics.macro_IoU():.3f}\t\tmF1-score: {metrics.macro_F1score():.3f}')


def get_parserargs():
    parser = argparse.ArgumentParser(description='Train the network on images and target masks')

    # ==========evaluate setting==========
    parser.add_argument('--on_cmd', '-cmd', type=bool, default=False, help='If training on cmd, set on_cmd True')
    parser.add_argument('--cfg_path', '-cfg', type=str, default=None, metavar="CFG", help='path to load a local config file')
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
    parser.add_argument('--cfg_note', '-cn', metavar='CN', type=str, help='note which will be saved in config name')
    parser.add_argument('--pretrain_path', '-pp', metavar='PP', type=str, default=None, help='pretrain model abspath')
    parser.add_argument('--device_idx', '-d', metavar='D', type=int, help='device which is training on')
    parser.add_argument('--is_eval', '-ev', metavar='EV', type=bool, default=True, help='the eval mode or not')
    parser.add_argument('--is_save_pred', '-sp', metavar='SP', type=bool, default=True, help='save predict or not')

    # ==========data setting==========
    parser.add_argument('--test_data_path', '-tep', metavar='TEP', type=str, help='Test dataset abspath')
    parser.add_argument('--class_list', '-cl', metavar='CL', type=int, nargs='+', help='label classes index list')

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
    args.cfg_note = 'test_BJ'
    args.pretrain_path = r'F:\RSDL\GID\train\Unet_ep51_valid_645.params'
    args.device_idx = 0
    args.is_eval = True
    args.is_save_pred = True

    args.test_data_path = r'F:\RSDL\GID\valid'
    args.class_list = [0, 255]


if __name__ == '__main__':
    cfgs = get_parserargs()
    evaluate(cfgs)
