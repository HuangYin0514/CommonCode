#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :catimg.py
@说明        : 将文件夹的图像合并成一个图像 cat操作
@时间        :2020/10/26 21:46:26
@作者        :HuangYin
@版本        :1.0
'''


import numpy as np
import glob
import os
import cv2
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import torch
from PIL import Image 


if __name__ == "__main__":

    trans = transforms.Compose([
        # transforms.Resize(64),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    WSI_MASK_PATH = 'experiments/catimg'
    paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.jpg'))
    paths.sort()
    img_list = []
    for path in paths:
        img = Image.open(path).convert('RGB') # 读取图像 
        img = trans(img)
        print(img.shape)
        img_list.append(img)
    print(len(img_list))
    ext = 'jpg'
    save_dir_path = 'experiments/catimg/dcganimg'
    # torch.cat(img_list, 0)
    torchvision.utils.save_image(img_list, str(Path(save_dir_path) / f'-dcgan.{ext}'), normalize=True)
    print('ok')
