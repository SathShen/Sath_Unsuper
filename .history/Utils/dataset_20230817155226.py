import torch
import torch.utils.data as data
import numpy as np
import os
import sys
from osgeo import gdal
from Utils.builder import build_transform
from Utils.plot import show_examples, show_augs
import random
from Utils.augmentation import *


def Idx2Class(label, class_list, device):
    """
    label: C,H,W or H,W

    """
    new_label = torch.zeros_like(label, dtype=torch.uint8, device=device)

    for i, label_idx in enumerate(class_list):
        label_idx = torch.tensor(label_idx, dtype=torch.uint8, device=device)
        new_label = torch.where(label == i, label_idx, new_label)
    return new_label


def Class2Idx(label, class_list, device):
    """
    label: C,H,W or H,W

    """
    new_label = torch.zeros_like(label, dtype=torch.uint8, device=device)

    for i, label_idx in enumerate(class_list):
        i = torch.tensor(i, dtype=torch.uint8, device=device)
        new_label = torch.where(label == label_idx, i, new_label)
    return new_label


def one_hot_encode(label, label_idx_list):
    """创建独热编码以实现多类loss计算"""
    # 索引为labels(1,H,W),分配1到num_classes维度(0)  zeros[index[0][i][j]] [i] [j] = 1
    num_classes = len(label_idx_list)
    for i, label_idx in enumerate(label_idx_list):
        i = torch.tensor(i, dtype=torch.uint8)
        isIdx = (label == label_idx)
        label = torch.where(isIdx, i, label)
    _, H, W = label.shape
    output = torch.zeros((num_classes, H, W), dtype=torch.uint8).scatter(0, label.long(), 1)
    return output


def one_hot_decode(one_hot_label, label_idx_list):
    """OneHotlabels: C,H,W"""
    num_classes, height, width = one_hot_label.shape
    label = torch.zeros((1, height, width), dtype=torch.uint8)

    for i, label_idx in enumerate(label_idx_list):
        one_class_label = one_hot_label[i, :, :].unsqueeze(0)
        label_idx = torch.tensor(label_idx, dtype=torch.uint8)
        isOne = (one_class_label == 1)       # bool map
        label = torch.where(isOne, label_idx, label)
    return label


def batch_one_hot_decode(one_hot_label_batch, label_idx_list, device):
    """one_hot_label_batch: N,C,H,W"""
    batch_size, num_classes, height, width = one_hot_label_batch.shape
    label_batch = torch.zeros((batch_size, height, width), dtype=torch.uint8, device=device)
    for i, label_idx in enumerate(label_idx_list):
        one_class_label_batch = one_hot_label_batch[:, i, :, :]
        label_idx = torch.tensor(label_idx, dtype=torch.uint8, device=device)
        isOne = (one_class_label_batch == 1)       # bool map
        label_batch = torch.where(isOne, label_idx, label_batch)
    return label_batch


def read_RS_image(path, data_type):
    dataset = gdal.Open(path)
    nXSize = dataset.RasterXSize
    nYSize = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, nXSize, nYSize).astype(data_type)
    return data


def RS_loader(img_name, img_path, lab_path, img_ext, lab_ext):
    if img_ext == 'tif':
        img = read_RS_image(f'{img_path}/{img_name}.{img_ext}', np.uint8).transpose((1, 2, 0))
    else:
        print("Error: Wrong image format!")
        sys.exit()
    if lab_ext == 'tif':
        mask = read_RS_image(f'{lab_path}/{img_name}.{lab_ext}', np.uint8)
        if len(mask.shape) == 2:
            mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
        elif len(mask.shape) == 3:
            mask = mask.transpose((1, 2, 0))
        else:
            print("Error: Wrong mask dimension!")
            sys.exit()
    else:
        print("Error: Wrong image format!")
        sys.exit()
    return img, mask     # img, mask : hwc


class LocalDatasetBuilder(data.Dataset):
    def __init__(self, cfgs, data_path, is_aug=False, img_loader=None):
        self.data_path = data_path
        self.img_path_list = []
        self.get_img_path_list_from_dir(self.data_path)
        self.img_fullname_list = os.listdir(self.img_path)
        self.lab_fullname_list = os.listdir(self.lab_path)
        self.img_ext = self.img_fullname_list[0].split('.')[-1]  # 图片后缀
        self.lab_ext = self.lab_fullname_list[0].split('.')[-1]
        self.img_name_list = list(map(lambda x: x[:-(len(self.img_ext) + 1)], self.img_fullname_list))
        self.class_list = cfgs.DATA.CLASS_LIST

        if img_loader is None:
            self.loader = RS_loader
        else:
            self.loader = img_loader
        
        self.trans = build_transform(cfgs, is_aug=is_aug)

    def get_img_path_list_from_dir(self, dir_path):
        img_fullname_list = os.listdir(self.img_path)
        for img_fullname in img_fullname_list:
            if len(img_fullname.split('.')) == 1:
                self.get_img_path_list_from_dir(dir_path + '/' + img_fullname)
            else:
                self.img_path_list.append(dir_path + '/' + img_fullname)
    
    def get_img_path_list(self):
        self.img_path_list = 

    def __getitem__(self, idx):
        img_name = self.img_name_list[idx]
        img, lab = self.loader(img_name, self.img_path, self.lab_path, self.img_ext, self.lab_ext)
        img_auged, lab_auged = self.trans(img, lab)    # chw
        onehot_lab_auged = one_hot_encode(lab_auged, self.class_list)
        return img_auged, onehot_lab_auged

    def __len__(self):
        return len(self.img_name_list)


def data_test():
    label_idx_list = [0, 255]
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


def one_hot_test():
    lab_list = [0, 255]
    lab = torch.randint(0, 2, (1, 50, 50))
    lab = torch.where(lab == 1, 255, 0)
    one_hot_lab = one_hot_encode(lab, lab_list)
    lab_2 = one_hot_decode(one_hot_lab, lab_list)
    show_examples([lab, lab_2], 1, 2)


if __name__ == "__main__":
    data_test()
    aug_test()
    # one_hot_test()

