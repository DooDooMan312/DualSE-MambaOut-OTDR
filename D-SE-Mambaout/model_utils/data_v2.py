# -*- coding: utf-8 -*-
# @Time    : 2025/5/15 20:18
# @Author  : XXX
# @Site    : 
# @File    : data_v2.py
# @Software: PyCharm 
# @Comment :

import cv2
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image

HW = 224


def readjpgfile(listpath, label, rate=None):
    """读取图像文件并预处理"""
    image_dir = sorted(os.listdir(listpath))
    total_images = len(image_dir)
    if rate:
        total_images = int(total_images * rate)

    # 初始化存储数组（添加通道维度）
    x = np.zeros((total_images, HW, HW, 1), dtype=np.uint8)
    y = np.zeros(total_images, dtype=np.uint8)

    for i, file in enumerate(image_dir[:total_images]):
        img_path = os.path.join(listpath, file)
        img = cv2.imread(img_path)

        # 转换为灰度图并添加通道维度
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = np.expand_dims(img_gray, axis=2)  # 形状变为 (H, W, 1)

        # 调整大小并存储
        resized = cv2.resize(img_gray, (HW, HW), interpolation=cv2.INTER_LINEAR)
        x[i] = resized
        y[i] = label

    return x, y


# 弱增强配置（已包含灰度转换）
weak_transform = transforms.Compose([
    transforms.ToPILImage(),  # 将NumPy数组转换为PIL图像
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 灰度图专用归一化
])

# 强增强配置（已包含灰度转换）
strong_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None, is_train=True):
        self.x = x
        self.y = y
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # 获取图像和标签
        image = self.x[index].squeeze(2)  # 移除冗余通道维度 (H, W, 1) -> (H, W)
        image = Image.fromarray(image)  # 转换为PIL图像

        if self.transform:
            image = self.transform(image)

        if self.y is not None:
            return image, self.y[index]
        return image


def getDateset(dir_class1, dir_class2, dir_class3, testSize=0.3, rate=None):
    # 读取所有类别的数据
    x1, y1 = readjpgfile(dir_class1, 0, rate)
    x2, y2 = readjpgfile(dir_class2, 1, rate)
    x3, y3 = readjpgfile(dir_class3, 2, rate)

    # 合并数据
    X = np.concatenate([x1, x2, x3], axis=0)
    Y = np.concatenate([y1, y2, y3], axis=0)

    # 划分训练集/验证集
    train_indices, val_indices = train_test_split(
        np.arange(len(X)),
        test_size=testSize,
        random_state=42,
        stratify=Y
    )

    # 创建数据集
    train_dataset = ImgDataset(
        X[train_indices],
        Y[train_indices],
        transform=strong_transform if testSize > 0 else weak_transform
    )

    val_dataset = ImgDataset(
        X[val_indices],
        Y[val_indices],
        transform=weak_transform
    )

    return train_dataset, val_dataset


def getDataLoader(class1path, class2path, class3path, batch_size, mode='train'):
    assert mode in ['train', 'val', 'test']

    if mode == 'train':
        train_set, val_set = getDateset(
            class1path,
            class2path,
            class3path,
            testSize=0.2
        )
        return (
            DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4),
            DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
        )
    elif mode == 'test':
        test_set, _ = getDateset(
            class1path,
            class2path,
            class3path,
            testSize=0  # 使用全部数据作为测试集
        )
        return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)