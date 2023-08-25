import matplotlib.pyplot as plt
import torch
import numpy as np


def show_examples(imgs, num_rows, num_cols, title_list=None, scale=1.5):
    """input C,H,W tensor or H,W,C ndarray"""
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    mode = None
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            if img.shape[0] == 1:
                mode = 'gray'
            ax.imshow(img.numpy().transpose((1, 2, 0)), cmap=mode)
        else:
            if img.shape[2] == 1:
                mode = 'gray'
            ax.imshow(img, cmap=mode)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if title_list:
            ax.set_title(title_list[i])
    plt.show()
    return axes


def show_augs(img, augs, num_rows, num_cols, title_list=None, scale=1.5):
    """input HWC ndarray"""
    imgs_auged = [0 for i in range(num_rows * num_cols)]
    for i in range(num_cols * num_rows):
        if i == 0:
            imgs_auged[i] = img
        img_auged = augs(img)
        imgs_auged[i] = img_auged
    show_examples(imgs_auged, num_rows, num_cols, title_list, scale)


def im_show(img_array):
    plt.figure()
    plt.imshow(img_array)
    plt.show()

