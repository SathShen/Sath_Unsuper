import torch
import torch.utils.data as data
import cv2
import numpy as np
import os
import sys
import random
from osgeo import gdal
from Utils.builder import build_transform
from Utils.plot import show_examples, show_augs
from Utils.augmentation import *
from yacs.config import CfgNode as CN


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
        img_fullname_list = os.listdir(self.img_path)
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


def data_test():
    dataset1 = LocalDatasetBuilder(data_path=r'E:\Sht\DATA\Test_data\GID_water', label_idx_list=label_idx_list)
    num_rows = 2
    num_cols = 4
    rint = random.randint(0, dataset1.__len__() - num_cols)
    imgs = []
    for i in range(num_cols):
        img = dataset1[rint + i][0]
        imgs.append(img)
    for i in range(num_cols):
        lab = one_hot_decode(dataset1[rint + i][1], label_idx_list)
        imgs.append(lab)
    show_examples(imgs, num_rows, num_cols)


def aug_test():
    label_idx_list = [0, 255]
    dataset1 = LocalDatasetBuilder(data_path=r'E:\Sht\DATA\Test_data\GID_water', label_idx_list=label_idx_list)
    num_rows = 2
    num_cols = 4
    rint = random.randint(0, dataset1.__len__())
    img = dataset1[rint][0].numpy().transpose((1, 2, 0))           # 因为要做增强，所以要先转numpy
    lab = one_hot_decode(dataset1[rint][1], label_idx_list).numpy().transpose((1, 2, 0))
    trans = ExtCompose([    # 输入:rgbn... hwc uint8 ndarry -> rgbn... chw float32 tensor
        ExtToTensor(),
        ExtRandomHorizontalFlip(),  # 随机水平反转
        ExtRandomVerticalFlip(),    # 随机垂直反转
        ExtRandomResizedCrop(size=(512, 512), scale=(0.3, 1), ratio=(3. / 4, 4. / 3)),
        ExtColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.2)])
    show_augs(img, lab, trans, num_rows, num_cols)


if __name__ == "__main__":
    base_cfg = CN()
    base_cfg.DATA = CN()
    base_cfg.DATA.TRAIN_DATA_PATH = r'E:\Sht\DATA\Test_data\GID_water'
    data_test()
    aug_test()
    # one_hot_test()

