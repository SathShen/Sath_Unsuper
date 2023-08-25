from torchvision import transforms
import torch.utils.data as data
import cv2
import numpy as np
import os
import sys
sys.path.append('./')
import random
from osgeo import gdal
from Utils.builder import build_transform
from Utils.plot import show_examples, show_augs
from yacs.config import CfgNode as CN
import torchvision.transforms.functional as F
from Utils.augmentation import HazeSimulation

def read_img_GDAL(path, data_type=np.uint8):
    dataset = gdal.Open(path)
    nXSize = dataset.RasterXSize
    nYSize = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, nXSize, nYSize).astype(data_type) # chw
    return data.transpose((1, 2, 0)) # hwc


def read_img_cv(path, data_type=np.uint8):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(data_type) # hwc


def default_loader(img_path):
    img_ext = img_path.split('.')[-1]
    if img_ext == 'tif':
        img = read_img_GDAL(img_path, np.uint8)
    elif img_ext == 'png' or img_ext == 'jpg':
        img = read_img_cv(img_path, np.uint8)
    else:
        print("Error: Wrong image format!")
        sys.exit()
    return img # img: hwc


class LocalDatasetBuilder(data.Dataset):
    def __init__(self, cfgs, img_loader=None):
        self.data_path = cfgs.DATA.TRAIN_DATA_PATH
        self.img_path_list = []
        self.get_img_path_list_from_dir(self.data_path)

        if img_loader is None:
            self.loader = default_loader
        else:
            self.loader = img_loader
        
        self.trans = build_transform(cfgs, is_aug=cfgs.AUG.IS_AUG)

    def get_img_path_list_from_dir(self, dir_path):
        img_fullname_list = os.listdir(dir_path)
        for img_fullname in img_fullname_list:
            if len(img_fullname.split('.')) == 1:
                self.get_img_path_list_from_dir(dir_path + '/' + img_fullname)
            else:
                self.img_path_list.append(dir_path + '/' + img_fullname)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = self.loader(img_path)
        img_auged = self.trans(img)    # chw
        return img_auged

    def __len__(self):
        return len(self.img_path_list)


def data_test(cfgs):
    dataset1 = LocalDatasetBuilder(cfgs)
    num_rows = 2
    num_cols = 4
    rint = random.randint(0, dataset1.__len__() - num_cols)
    imgs = []
    for i in range(num_cols * num_rows):
        img = dataset1[rint + i]
        imgs.append(img)
    show_examples(imgs, num_rows, num_cols)


def aug_test(cfgs):
    dataset1 = LocalDatasetBuilder(cfgs)
    num_rows = 2
    num_cols = 4
    rint = random.randint(0, dataset1.__len__())
    img = dataset1[rint].numpy().transpose((1, 2, 0))           # 因为要做增强，所以要先转numpy
    trans = transforms.Compose([    # 输入:rgbn... hwc uint8 ndarry -> rgbn... chw float32 tensor
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),  # 随机水平反转
        # transforms.RandomVerticalFlip(),    # 随机垂直反转
        # transforms.RandomResizedCrop(size=(cfgs.AUG.CROP_SIZE, cfgs.AUG.CROP_SIZE), scale=(cfgs.AUG.CROP_PER, 1), 
        #                              ratio=(1 - cfgs.AUG.RESIZE_RATIO, 1 + cfgs.AUG.RESIZE_RATIO)),
        # transforms.ColorJitter(brightness=cfgs.AUG.INTENSITY, contrast=cfgs.AUG.CONTRAST,
        #                        saturation=cfgs.AUG.SATURATION, hue=cfgs.AUG.HUE),
        HazeSimulation(p=0.9)
                               ])
    show_augs(img, trans, num_rows, num_cols)


if __name__ == "__main__":
    test_cfg = CN()
    test_cfg.DATA = CN()
    test_cfg.DATA.TRAIN_DATA_PATH = r'F:\Backup\Not_RS\classification\stl10\labeled'
    test_cfg.AUG = CN()
    test_cfg.AUG.IS_AUG = False
    test_cfg.AUG.INTENSITY = 0.4
    test_cfg.AUG.HUE = 0.2
    test_cfg.AUG.SATURATION = 0.3
    test_cfg.AUG.CONTRAST = 0.3
    test_cfg.AUG.CROP_PER = 0.4
    test_cfg.AUG.RESIZE_RATIO = 0.3
    test_cfg.AUG.CROP_SIZE = 512

    data_test(test_cfg)
    aug_test(test_cfg)

