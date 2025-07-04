import os
from utils import  read_image
suffixes = ['/*.png', '/*.jpg', '/*.bmp', '/*.tif']

import matplotlib.pyplot as plt

import random
import torch
import numpy as np
import glob
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import torchvision.transforms.functional as TF
class TrainDataSet(Dataset):
    def __init__(self, lr_roots, gt_roots, patch_size):
        super().__init__()

        self.file_client = None
        self.patch_size = patch_size
        self.suffixes = suffixes

        # 支持多个路径
        self.lr_data = []
        self.gt_data = []

        for lr_root, gt_root in zip(lr_roots, gt_roots):
            print(lr_root, gt_root)
            for suffix in self.suffixes:
                self.lr_data.extend(glob.glob(os.path.join(lr_root + suffix)))
                self.gt_data.extend(glob.glob(os.path.join(gt_root + suffix)))

        self.lr_data = sorted(self.lr_data)
        self.gt_data = sorted(self.gt_data)

        assert len(self.lr_data) == len(self.gt_data), "the length of lrs and gts is not equal!"

    def __getitem__(self, index):

        # 获取低清图和高清图的路径
        lr_name = self.lr_data[index]
        gt_name = self.gt_data[index]

        # 读取低清图和高清图
        lr_img = Image.open(lr_name).convert('RGB')
        gt_img = Image.open(gt_name).convert('RGB')

        # 获取图像尺寸
        width, height = lr_img.size

        # 如果图像尺寸小于 patch_size，进行 resize 操作
        if width < self.patch_size and height < self.patch_size:
            lr_img = lr_img.resize((self.patch_size, self.patch_size), Image.ANTIALIAS)
            gt_img = gt_img.resize((self.patch_size, self.patch_size), Image.ANTIALIAS)
        elif width < self.patch_size:
            lr_img = lr_img.resize((self.patch_size, height), Image.ANTIALIAS)
            gt_img = gt_img.resize((self.patch_size, height), Image.ANTIALIAS)
        elif height < self.patch_size:
            lr_img = lr_img.resize((width, self.patch_size), Image.ANTIALIAS)
            gt_img = gt_img.resize((width, self.patch_size), Image.ANTIALIAS)

        # 获取新的图像尺寸
        width, height = lr_img.size

        aug = random.randint(0, 2)
        if aug == 1:
            lr_img = TF.adjust_gamma(lr_img, 1)
            gt_img = TF.adjust_gamma(gt_img, 1)

        aug = random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            lr_img = TF.adjust_saturation(lr_img, sat_factor)
            gt_img = TF.adjust_saturation(gt_img, sat_factor)

        # 随机裁剪
        x = random.randint(0, width - self.patch_size)
        y = random.randint(0, height - self.patch_size)

        # 裁剪图像
        lr_crop_img = lr_img.crop((x, y, x + self.patch_size, y + self.patch_size))
        gt_crop_img = gt_img.crop((x, y, x + self.patch_size, y + self.patch_size))

        # --- 转换为张量并进行归一化 ---
        transform_lr = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])

        lr_tensor = transform_lr(lr_crop_img)
        gt_tensor = transform_gt(gt_crop_img)


        # Data Augmentations
        aug = random.randint(0, 8)
        if aug==1:
            lr_tensor = lr_tensor.flip(1)
            gt_tensor = gt_tensor.flip(1)
        elif aug==2:
            lr_tensor = lr_tensor.flip(2)
            gt_tensor = gt_tensor.flip(2)
        elif aug==3:
            lr_tensor = torch.rot90(lr_tensor,dims=(1,2))
            gt_tensor = torch.rot90(gt_tensor,dims=(1,2))
        elif aug==4:
            lr_tensor = torch.rot90(lr_tensor,dims=(1,2), k=2)
            gt_tensor = torch.rot90(gt_tensor,dims=(1,2), k=2)
        elif aug==5:
            lr_tensor = torch.rot90(lr_tensor,dims=(1,2), k=3)
            gt_tensor = torch.rot90(gt_tensor,dims=(1,2), k=3)
        elif aug==6:
            lr_tensor = torch.rot90(lr_tensor.flip(1),dims=(1,2))
            gt_tensor = torch.rot90(gt_tensor.flip(1),dims=(1,2))
        elif aug==7:
            lr_tensor = torch.rot90(lr_tensor.flip(2),dims=(1,2))
            gt_tensor = torch.rot90(gt_tensor.flip(2),dims=(1,2))

        # --- 检查图像通道是否为 3 ---
        if lr_tensor.shape[0] != 3 or gt_tensor.shape[0] != 3:
            raise Exception(f"Bad image channel: {gt_name}")

        return lr_tensor, gt_tensor


    def __len__(self):
        return len(self.lr_data)




class TestDataSet(Dataset):
    def __init__(self, lr_root, gt_root):
        super().__init__()

        self.lr_data = []
        self.gt_data = []
        for suffix in suffixes:
            self.lr_data.extend(glob.glob(lr_root + suffix))
            self.gt_data.extend(glob.glob(gt_root + suffix))
        self.lr_data = sorted(self.lr_data)
        self.gt_data = sorted(self.gt_data)

        assert len(self.lr_data) == len(self.gt_data), "the length of lrs and gts is not equal!"

    def __getitem__(self, index):
        lr_path = self.lr_data[index]
        gt_path = self.gt_data[index]

        lr_img = Image.open(lr_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        # --- 转换为张量并进行归一化 ---
        transform_lr = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])

        lr_tensor = transform_lr(lr_img)
        gt_tensor = transform_gt(gt_img)

        # 提取文件名作为字符串（例如 "0001.png"）
        filename = os.path.basename(lr_path)
        return lr_tensor, gt_tensor,filename

    def __len__(self):
        return len(self.lr_data)