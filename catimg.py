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


class CatImg():

    def __init__(self, old_path, save_path):
        self.old_path = old_path
        self.save_path = save_path
        self.save_ext = 'jpg'

        self.trans = transforms.Compose([
            # transforms.Resize(64),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def cat(self):
        paths = glob.glob(os.path.join(self.old_path, '*.jpg'))
        paths.sort()
        img_list = []
        for path in paths:
            img = Image.open(path).convert('RGB')  # 读取图像
            img = self.trans(img)
            # print(img.shape)
            img_list.append(img)
        print(len(img_list))
        # torch.cat(img_list, 0)
        torchvision.utils.save_image(img_list, str(Path(self.save_path) / f'-dcgan.{self.save_ext}'),
                                     nrow=5, normalize=True)


if __name__ == "__main__":

    old_path = 'delFolder/sstegan'
    save_path = 'delFolder/catSSteGAN'
    catImg = CatImg(old_path, save_path)
    catImg.cat()

    old_path = 'delFolder/biggan'
    save_path = 'delFolder/catBigGAN'
    catImg2 = CatImg(old_path, save_path)
    catImg2.cat()

    print('ok')
