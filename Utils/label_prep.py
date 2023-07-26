import os
import sys
from tqdm import tqdm
import cv2
from osgeo import gdal
import numpy as np


def label_1C_Binarization(label_path, save_path, mix_list, out_extension='tif'):
    """二值化label，输入单通道，得到单通道0-255二值label"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    imgs_list = os.listdir(label_path)
    for img_full_name in tqdm(imgs_list):
        img_ext = img_full_name.split('.')[-1]  # 源文件后缀
        img_name = img_full_name[:-(len(img_ext)+1)]  # 源文件名称
        if (img_ext == 'tif') | (img_ext == 'jpg') | (img_ext == 'png'):
            img = cv2.imread(f'{label_path}/{img_full_name}', flags=cv2.IMREAD_UNCHANGED)  # 读取图片
            is255 = np.full(img.shape, False, dtype=bool)  # 全 False 初始化
            for i, idx in enumerate(mix_list):
                if i == 0:
                    is255 = (img == idx)
                else:
                    is255 = is255 | (img == idx)
            img_new = np.where(is255, 255, 0).astype(np.uint8)  # 设为255 其他设为0
            if out_extension == 'tif':
                cv2.imwrite(f'{save_path}/{img_name}.{out_extension}', img_new, (int(cv2.IMWRITE_TIFF_COMPRESSION), 1))
            if out_extension == 'png':
                cv2.imwrite(f'{save_path}/{img_name}.{out_extension}', img_new, (int(cv2.IMWRITE_PNG_COMPRESSION), 0))
        else:
            print('文件夹内有不支持的格式。')


def label_RGB_Binarization(label_path, save_path, palette, mix_list, out_extension='tif'):
    """二值化label，输入多通道（RBGlabel），得到单通道0-255二值label"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    imgs_list = os.listdir(label_path)
    for img_full_name in tqdm(imgs_list):
        img_ext = img_full_name.split('.')[-1]  # 源文件后缀
        img_name = img_full_name[:-(len(img_ext) + 1)]  # 源文件名称
        if (img_ext == 'tif') | (img_ext == 'jpg') | (img_ext == 'png'):
            img = cv2.imread(f'{label_path}/{img_full_name}', flags=cv2.IMREAD_UNCHANGED)  # 读取图片
            img_b, img_g, img_r = cv2.split(img)
            is255 = np.full(img.shape, False, dtype=bool)  # 全 False 初始化
            for i, idx in enumerate(mix_list):
                if i == 0:
                    is255 = (img_r == palette[idx][0]) & (img_g == palette[idx][1]) & (img_b == palette[idx][2])
                else:
                    is255 = is255 | ((img_r == palette[idx][0]) & (img_g == palette[idx][1]) & (img_b == palette[idx][2]))
            img_new = np.where(is255, 255, 0).astype(np.uint8)
            if out_extension == 'tif':
                cv2.imwrite(f'{save_path}/{img_name}.{out_extension}', img_new, (int(cv2.IMWRITE_TIFF_COMPRESSION), 1))
            if out_extension == 'png':
                cv2.imwrite(f'{save_path}/{img_name}.{out_extension}', img_new, (int(cv2.IMWRITE_PNG_COMPRESSION), 0))
        else:
            print('文件夹内有不支持的格式。')


def label_RGB2_1C(label_path, save_path, palette, out_extension='tif'):
    """输入多通道label、得到单通道label"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    imgs_list = os.listdir(label_path)
    for img_full_name in tqdm(imgs_list):
        img_ext = img_full_name.split('.')[-1]  # 源文件后缀
        img_name = img_full_name[:-(len(img_ext) + 1)]  # 源文件名称
        if (img_ext == 'tif') | (img_ext == 'jpg') | (img_ext == 'png'):
            img = cv2.imread(f'{label_path}/{img_full_name}', flags=cv2.IMREAD_UNCHANGED)  # 读取图片
            img_b, img_g, img_r = cv2.split(img)
            img_new = np.full(img_r.shape, 99, dtype=np.uint8)  # 全99初始化
            for i, color in enumerate(palette.values()):
                isColor = (img_r == color[0]) & (img_g == color[1]) & (img_b == color[2])
                img_new = np.where(isColor, i, img_new)
            if out_extension == 'tif':
                cv2.imwrite(f'{save_path}/{img_name}.{out_extension}', img_new, (int(cv2.IMWRITE_TIFF_COMPRESSION), 1))
            if out_extension == 'png':
                cv2.imwrite(f'{save_path}/{img_name}.{out_extension}', img_new, (int(cv2.IMWRITE_PNG_COMPRESSION), 0))
        else:
            print('文件夹内有不支持的格式。')


def crop_2_blks(img_path, save_path, crop_shape, crop_step, in_extention='tif', out_extension='tif'):  # step为步长
    """ 使用opencv裁剪，支持无黑边影像 """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    imgs_list = list(filter(lambda x: x.find(f'{in_extention}') != -1, os.listdir(img_path)))
    for img_full_name in tqdm(imgs_list):
        img_ext = img_full_name.split('.')[-1]  # 源文件后缀
        img_name = img_full_name[:-(len(img_ext) + 1)]  # 源文件名称
        img = cv2.imread(f'{img_path}/{img_full_name}')
        img_shape = img.shape

        for h in range(0, img_shape[0], crop_step[0]):
            start_h = h  # star_h表示起始高度，从0以步长step=256开始循环
            for w in range(0, img_shape[1], crop_step[1]):
                start_w = w   # star_w表示起始宽度，从0以步长step=256开始循环

                end_h = start_h + crop_shape[0]  # end_h是终止高度
                if end_h > img_shape[0]:  # 如果边缘位置不够512的列
                    # 以倒数512形成裁剪区域
                    start_h = img_shape[0] - crop_shape[0]
                    end_h = start_h + crop_shape[0]

                end_w = start_w + crop_shape[1]  # end_w是中止宽度
                if end_w > img_shape[1]:  # 如果边缘位置不够512的行
                    # 以倒数512形成裁剪区域
                    start_w = img_shape[1] - crop_shape[1]
                    end_w = start_w + crop_shape[1]

                cropped = img[start_h:end_h, start_w:end_w]
                save_name = f'{img_name}_{start_h}_{start_w}'  # 用起始坐标来命名切割得到的图像
                if out_extension == 'tif':
                    cv2.imwrite(f'{save_path}/{save_name}.{out_extension}', cropped, (int(cv2.IMWRITE_TIFF_COMPRESSION), 1))
                if out_extension == 'png':
                    cv2.imwrite(f'{save_path}/{save_name}.{out_extension}', cropped, (int(cv2.IMWRITE_PNG_COMPRESSION), 0))
    print("已成功执行！")


def crop_RS_2_RGBblks(img_path, save_path, crop_shape, crop_step):
    """多波段遥感影像裁剪，输出RGB, 过滤掉0值大于阈值和有nan的影像"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    imgs_list = list(filter(lambda x: (x[-3:] == 'tif') | (x[-4:] == 'tiff'), os.listdir(img_path)))
    for img_full_name in tqdm(imgs_list):
        img_ext = img_full_name.split('.')[-1]  # 源文件后缀
        img_name = img_full_name[:-(len(img_ext) + 1)]  # 源文件名称
        dataset = gdal.Open(f'{img_path}/{img_full_name}')
        img = dataset.ReadAsArray()

        img = img.transpose(1, 2, 0)[:, :, ::-1]  # chw转hwc，通道RGBN倒序排列位NBGR
        if img.shape[2] > 3:
            img = img[:, :, -3:]
        maxV = img.max()
        minV = img.min()
        img = (255 * (img.astype('float32') - minV) / (maxV - minV)).astype('uint8')
        img_shape = img.shape

        for h in range(0, img_shape[0], crop_step[0]):
            start_h = h  # star_h表示起始高度，从0以步长step=256开始循环
            for w in range(0, img_shape[1], crop_step[1]):
                start_w = w   # star_w表示起始宽度，从0以步长step=256开始循环

                end_h = start_h + crop_shape[0]  # end_h是终止高度
                if end_h > img_shape[0]:  # 如果边缘位置不够512的列
                    # 以倒数512形成裁剪区域
                    start_h = img_shape[0] - crop_shape[0]
                    end_h = start_h + crop_shape[0]

                end_w = start_w + crop_shape[1]  # end_w是中止宽度
                if end_w > img_shape[1]:  # 如果边缘位置不够512的行
                    # 以倒数512形成裁剪区域
                    start_w = img_shape[1] - crop_shape[1]
                    end_w = start_w + crop_shape[1]

                cropped = img[start_h:end_h, start_w:end_w, :]
                img_b, img_g, img_r = cv2.split(cropped)
                isZero = (img_b == 0) & (img_g == 0) & (img_r == 0)
                if (np.isnan(cropped).sum() != 0) or isZero.sum() >= 50:
                    continue
                save_name = f'{img_name}_{start_h}_{start_w}'  # 用起始坐标来命名切割得到的图像
                cv2.imwrite(f'{save_path}/{save_name}.tif', cropped, (int(cv2.IMWRITE_TIFF_COMPRESSION), 1))
    print("已成功执行！")


def crop_RSlabel_2_blks(img_path, save_path, crop_shape, crop_step):
    """单波段label裁剪, 输出RGB, 过滤掉0值大于阈值和有nan的影像"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    imgs_list = list(filter(lambda x: (x[-3:] == 'tif') | (x[-4:] == 'tiff'), os.listdir(img_path)))
    for img_full_name in tqdm(imgs_list):
        img_ext = img_full_name.split('.')[-1]  # 源文件后缀
        img_name = img_full_name[:-(len(img_ext) + 1)]  # 源文件名称
        dataset = gdal.Open(f'{img_path}/{img_full_name}')
        img = dataset.ReadAsArray().astype('uint8')
        img_shape = img.shape

        for h in range(0, img_shape[0], crop_step[0]):
            start_h = h  # star_h表示起始高度，从0以步长step=256开始循环
            for w in range(0, img_shape[1], crop_step[1]):
                start_w = w  # star_w表示起始宽度，从0以步长step=256开始循环

                end_h = start_h + crop_shape[0]  # end_h是终止高度
                if end_h > img_shape[0]:  # 如果边缘位置不够512的列
                    # 以倒数512形成裁剪区域
                    start_h = img_shape[0] - crop_shape[0]
                    end_h = start_h + crop_shape[0]

                end_w = start_w + crop_shape[1]  # end_w是中止宽度
                if end_w > img_shape[1]:  # 如果边缘位置不够512的行
                    # 以倒数512形成裁剪区域
                    start_w = img_shape[1] - crop_shape[1]
                    end_w = start_w + crop_shape[1]

                cropped = img[start_h:end_h, start_w:end_w]
                isZero = (cropped == 0)
                if (np.isnan(cropped).sum() != 0) or isZero.sum() >= 50:
                    continue
                newname = img_name.replace('_ReferenceMask', "")
                save_name = f'{newname}_{start_h}_{start_w}'  # 用起始坐标来命名切割得到的图像
                cv2.imwrite(f'{save_path}/{save_name}.tif', cropped, (int(cv2.IMWRITE_TIFF_COMPRESSION), 1))
    print("已成功执行！")


def rename(img_path, save_path):
    """使用rename函数重命名，rename会将源文件删除"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    imgs_list = os.listdir(img_path)
    for i, filename in tqdm(enumerate(imgs_list)):
        img_ext = filename.split('.')[-1]  # 源文件后缀
        newname = f'{100 + i}.{img_ext}'
        # newname = filename.replace('_label_', f"_{i}_")
        os.rename(f'{img_path}/{filename}',
                  f'{save_path}/{newname}')


def RemoveUncorrespond(img_path, lab_path):
    """去除image和label文件夹中名称不对应的影像对"""
    img_fullname_list = os.listdir(img_path)
    img_ext = img_fullname_list[0].split('.')[-1]
    img_name_list = list(map(lambda x: x[:-(len(img_ext) + 1)], img_fullname_list))
    lab_fullname_list = os.listdir(lab_path)
    lab_ext = lab_fullname_list[0].split('.')[-1]
    lab_name_list = list(map(lambda x: x[:-(len(lab_ext) + 1)], lab_fullname_list))

    img_del_name_list = list(set(img_name_list).difference(set(lab_name_list)))  # img有lab没有的
    lab_del_name_list = list(set(lab_name_list).difference(set(img_name_list)))  # lab有img没有的

    print('img中非对应影像：', img_del_name_list)
    print('lab中非对应影像：', lab_del_name_list)
    choice = input('确认删除，[Y / N]')
    if choice.upper() == 'Y':
        for img_del_name in tqdm(img_del_name_list):
            del_path = f'{img_path}/{img_del_name}.{img_ext}'
            os.remove(del_path)
        for lab_del_name in tqdm(lab_del_name_list):
            del_path = f'{lab_path}/{lab_del_name}.{lab_ext}'
            os.remove(del_path)
        print("去除非对应样本成功。")
    elif choice.upper() == 'N':
        print("取消删除，退出程序")
        sys.exit()
    else:
        print('输入错误，推出程序')
        sys.exit()


def RemoveNonetype(img_path, lab_path):
    """去除打不开的损坏影像(为None)"""
    img_fullname_list = os.listdir(img_path)
    img_ext = img_fullname_list[0].split('.')[-1]
    img_name_list = list(map(lambda x: x[:-(len(img_ext) + 1)], img_fullname_list))
    lab_fullname_list = os.listdir(lab_path)
    lab_ext = lab_fullname_list[0].split('.')[-1]

    count = 0
    for img_name in tqdm(img_name_list):
        img = cv2.imread(f'{img_path}/{img_name}.{img_ext}')
        if img is None:
            os.remove(f'{img_path}/{img_name}.{img_ext}')
            os.remove(f'{lab_path}/{img_name}.{lab_ext}')
            count += 1
    print(f'Removed {count} imgs in total.')


# 遥感影像处理模板
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
                                   nXSize, nYSize, bands=3, eType=gdal.GDT_Byte, options=["COMPRESS=LZW"])
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


if __name__ == '__main__':
    # ================crop==================
    palette = {
        # RGB
        0: (255, 251, 177),  # Farmland
        1: (214, 167, 201),  # Field
        2: (49, 173, 105),  # Woodland
        3: (131, 194, 56),  # Grass
        4: (229, 103, 102),  # Building
        5: (209, 201, 211),  # Road
        6: (235, 137, 126),  # Structures
        7: (197, 154, 140),  # Artificial piling
        8: (215, 200, 185),  # Bare land
        9: (163, 214, 245),  # Water
    }
    Deepglobe_palette = {
        # RGB
        0: (0, 255, 255),  # urban_land
        1: (255, 255, 0),  # agriculture_land
        2: (255, 0, 255),  # rangeland
        3: (0, 255, 0),  # forest_land
        4: (0, 0, 255),  # water
        5: (255, 255, 255),  # barren_land
        6: (0, 0, 0),  # unknown
    }
    mix_list = [1, 2, 3]

    
    img_path = r'E:\Sht\DATA\Test_data\BJ\valid\image'
    lab_path = r'E:\Sht\DATA\Test_data\BJ\valid\label'
    # crop_img_path = r'E:\GF1_WHU\img_crop'
    # crop_lab_path = r'E:\GF1_WHU\lab_crop'

    RemoveNonetype(img_path, lab_path)


    # crop_2_blks(img_path, save_path, (512, 512), (512, 512), 'jpg', 'png')
    # crop_RS_2_RGBblks(img_path, crop_img_path, (512, 512), (512, 512))
    # crop_RSlabel_2_blks(lab_path, crop_lab_path, (512, 512), (512, 512))
    # rename(crop_lab_path, crop_lab_path)
    # RemoveUncorrespond(crop_img_path, crop_lab_path)
    # label_1C_Binarization(label_path, save_path, mix_list, 'tif')
    # label_RGB_Binarization(label_path, save_path, Deepglobe_palette, mix_list, 'png')
    # label_RGB2_1C(label_path, save_path, palette, 'tif')





