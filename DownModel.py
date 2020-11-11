#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :downmodel.py
@说明        :下载model
@时间        :2020/10/27 11:46:36
@作者        :HuangYin
@版本        :1.0
'''
from torchvision import models

if __name__ == "__main__":
    resnet = models.resnet101(pretrained=True)
    print(resnet)