"""
===============================
  -*- coding:utf-8 -*-
  Author    :hanjinyue
  Time      :2020/7/31 15:28
  File      :train_letnet-06-han.py
================================
"""
import sys
import os
import torch
from torch import nn

from lesson_han import cfg
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from lesson_han.my_dataset import RMBDatasetHan
from model.lenet import LeNet
import torch.optim as optim
import matplotlib as plt

# 参数设置
from tools.common_tools import transform_invert

MAX_EPOCH = 10
BATCH_SIZE = 16
LR = 0.01
log_interval = 10
val_interval = 1

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(),'../..','data'))
print("BASE_DIR=",BASE_DIR)
sys.path.append(BASE_DIR)
#============================step 1/5 加载数据和数据预处理=========================
split_dir = os.path.join(BASE_DIR,'rmb_split')
if not os.path.exists(split_dir):
    raise Exception(r"数据 {} 不存在, 回到lesson-06\1_split_dataset.py生成数据".format(split_dir))

train_dir = os.path.join(split_dir,'train')
valid_dir = os.path.join(split_dir,'valid')


#处理数据
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform=transforms.Compose([transforms.Resize(32,32),
    transforms.RandomCrop(32,padding=4),
    transforms.Normalize(norm_mean,norm_std),
    transforms.ToTensor()])
valid_transform = transforms.Compose([
    transforms.Resize(32,32),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean,norm_std)
])

#和构建myDataSet实例
train_data = RMBDatasetHan(train_dir,transform=train_transform)
valid_data = RMBDatasetHan(valid_dir,transform=valid_transform)
print('train_data',train_data.__class__)
#加载数据
train_loader = DataLoader(train_data,batch_size=cfg.batch_size,shuffle=True)
valid_loader = DataLoader(valid_data,batch_size=cfg.batch_size)
print('train_loader=',train_data.__class__)

# ============================ step 2/5 模型 ============================

net = LeNet(classes=2)
net.initialize_weights()

# ============================ step 3/5 损失函数 ============================
criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数

# ============================ step 4/5 优化器 ============================
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)                        # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)     # 设置学习率下降策略

# ============================ step 5/5 训练 ============================
train_curve = list()
valid_curve = list()

for epoch in range(MAX_EPOCH):
    for i,data in enumerate(train_loader):
        inputs,labels = data
        img_tensor = inputs[0,...]
        img = transform_invert(img_tensor,train_transform)
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()









