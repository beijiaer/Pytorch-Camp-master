"""
===============================
  -*- coding:utf-8 -*-
  Author    :hanjinyue
  Time      :2020/7/31 21:24
  File      :transforms-methods_1-08.py
================================
"""

# 参数设置
import os
import sys
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from lesson_han import cfg
from lesson_han.my_dataset import RMBDatasetHan
from lesson_han.my_transforms_han import AddPepperNoise
from tools.common_tools import transform_invert
from tools.my_dataset import RMBDataset
from lesson_han import my_transforms_han

MAX_EPOCH = 10
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(),'../..','data'))
print("BASE_DIR=",BASE_DIR)
sys.path.append(BASE_DIR)
#============================step 1/5 加载数据和数据预处理=========================
split_dir = os.path.join(BASE_DIR,'rmb_split')
train_dir = os.path.join(split_dir,'train')
valid_dir = os.path.join(split_dir,'valid')


#处理数据
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform=transforms.Compose([
     transforms.Resize((224, 224)),
    AddPepperNoise(snr=0.9,p=0.9),
    # transforms.CenterCrop(196)
    # transforms.RandomCrop(32,padding=4),
    # 1 Pad
    # transforms.Pad(padding=32, fill=(255, 0, 0), padding_mode='constant'),
    # transforms.Pad(padding=(8, 64), fill=(255, 0, 0), padding_mode='constant'),
    # transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='constant'),
    # transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='symmetric'),

    # 2 ColorJitter
    # transforms.ColorJitter(brightness=0.5),
    # transforms.ColorJitter(contrast=0.5),
    # transforms.ColorJitter(saturation=0.5),
    # transforms.ColorJitter(hue=0.3),

    # 3 Grayscale
    # transforms.Grayscale(num_output_channels=3),

    # 4 Affine
    # transforms.RandomAffine(degrees=30),
    # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), fillcolor=(255, 0, 0)),
    # transforms.RandomAffine(degrees=0, scale=(0.7, 0.7)),
    # transforms.RandomAffine(degrees=0, shear=(0, 0, 0, 45)),
    # transforms.RandomAffine(degrees=0, shear=90, fillcolor=(255, 0, 0)),

    # 5 Erasing
    # transforms.ToTensor(),
    # transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(254/255, 0, 0)),
    # transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='1234'),

    # 1 RandomChoice
    # transforms.RandomChoice([transforms.RandomVerticalFlip(p=1), transforms.RandomHorizontalFlip(p=1)]),

    # 2 RandomApply
    # transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=45, fillcolor=(255, 0, 0)),
    #                         transforms.Grayscale(num_output_channels=3)], p=0.5),
    # 3 RandomOrder
    # transforms.RandomOrder([transforms.RandomRotation(15),
    #                         transforms.Pad(padding=32),
    #                         transforms.RandomAffine(degrees=0, translate=(0.01, 0.1), scale=(0.9, 1.1))]),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean,norm_std),

    ])
valid_transform = transforms.Compose([
    transforms.Resize(224,224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean,norm_std)
])

#和构建myDataSet实例
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=train_transform)
print('train_data',train_data.__class__)
#加载数据
# train_loader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
# valid_loader = DataLoader(valid_data,batch_size=BATCH_SIZE)
# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
print('train_loader=',train_data.__class__)

for epoch in range(MAX_EPOCH):
    for i,data in enumerate(train_loader):
        inputs,labels = data
        img_tensor = inputs[0,...]
        img = transform_invert(img_tensor,train_transform)
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()
