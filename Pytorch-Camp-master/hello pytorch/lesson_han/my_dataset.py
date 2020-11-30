"""
===============================
  -*- coding:utf-8 -*-
  Author    :hanjinyue
  Time      :2020/7/31 15:30
  File      :my_dataset.py
================================
"""
import os

from PIL import Image
from torch.utils.data import Dataset
import random
random.seed(1)
rmb_label = {"1": 0, "100": 1}
class RMBDatasetHan(Dataset):
    def __init__(self,data_dir,transform=None):
        """
        RMB面额分类任务
        :param data_dir: 数据集所在路径
        :param transform: 是否对数据集进行预处理
        """
        self.label_name = {"1":0,"100":1}
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = rmb_label[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info
    def __getitem__(self, index):
        """
        获取索引为index的图片和标签
        :param index: 根据图片路径列表和图片索引，获取索引为index的图片和标签
        :return:
        """
        path_img,label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None :
            img = self.transform(img)

        return img,label


    def __len__(self):
        return len(self.data_info)

