"""
===============================
  -*- coding:utf-8 -*-
  Author    :hanjinyue
  Time      :2020/8/2 15:35
  File      :my_transforms_han.py
================================
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
class AddPepperNoise(object):

    def __init__(self,snr,p=0.8):
        assert isinstance(snr,float) and (isinstance(p,float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """

        :param img: PLT img
        :return: PLT img
        """
        if random.uniform(0,1) < self.p:
            img_ = np.array(img).copy()
            h,w,c = img_.shape
            signal = self.snr
            noise = 1. - signal
            mask = np.random.choice((0,1,2), size=(h,w,1),p=(signal, noise/2.0, noise/2.))
            mask = np.repeat(mask,c,axis=2)
            img_[mask == 1] = 255
            img_[mask == 2] = 0
            return Image.fromarray(img_.astype('uint8')).convert('RGB')

        else:
            return img
