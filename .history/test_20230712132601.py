import torch
import numpy as np
import os
from tqdm import tqdm
import cv2
from osgeo import gdal
import matplotlib.pyplot as plt
import argparse


def crop_2_blks(img_path, save_path, in_extention='tiff', out_extension='tif'):  # step为步长
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    imgs_list = list(filter(lambda x: x.find(f'{in_extention}') != -1, os.listdir(img_path)))
    for img_full_name in tqdm(imgs_list):
        img_ext = img_full_name.split('.')[-1]  # 源文件后缀
        img_name = img_full_name[:-(len(img_ext) + 1)]  # 源文件名称

        # ===========gdal read===========
        dataset = gdal.Open(f'{img_path}/{img_full_name}')
        nXSize = dataset.RasterXSize
        nYSize = dataset.RasterYSize
        adf_GeoTransform = dataset.GetGeoTransform()
        im_Proj = dataset.GetProjection()
        img = dataset.ReadAsArray(0, 0, nXSize, nYSize)  # chw 
        # ============中间处理=============
        maxV = img.max()
        minV = img.min()
        img = (255 * (img.astype(np.float32) - minV) / (maxV - minV)).astype(np.uint8)
        
        # ===========gdal write===========
        B, G, R = img[0, :, :], img[1, :, :], img[2, :, :]  # chw

        driver = gdal.GetDriverByName("GTiff")
        datasetnew = driver.Create(f"{save_path}/{img_name}.{out_extension}",
                                   nXSize, nYSize, bands=3, eType=gdal.GDT_Byte)
        datasetnew.SetGeoTransform(adf_GeoTransform)
        datasetnew.SetProjection(im_Proj)
        bandB = datasetnew.GetRasterBand(1)
        bandG = datasetnew.GetRasterBand(2)
        bandR = datasetnew.GetRasterBand(3)
        bandB.WriteArray(B);
        bandG.WriteArray(G);
        bandR.WriteArray(R)
        datasetnew.FlushCache()  # Write to disk.必须有清除缓存
              
    print("已成功执行！")


def get_parserargs():
    parser = argparse.ArgumentParser(description='Train the network on images and target masks')
    parser.add_argument('--ip', '-ip')
    parser.add_argument('--op', '-op')

    args, unknown = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = get_parserargs()
    args.ip = r'F:\Backup\GF1_WHU_cloud\image'
    args.op = r'F:\GID_WHU_CLOUD_8BIT'
    crop_2_blks(args.ip, args.op)