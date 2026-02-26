import cv2
import os
import numpy as np
import torchvision.transforms as transforms
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
import torch
import random
from imblearn.over_sampling import SMOTE
from collections import Counter

HW = 224

def readjpgfile(listpath, label, rate=None):
    assert rate == None or rate // 1 == rate
    # label 是一个布尔值，代表需不需要返回 y 值
    image_dir = sorted(os.listdir(listpath))
    n = len(image_dir)  # 图像的数量
    if rate:
        n = n * rate
    # x存储图片，每张灰色图片都是1800(高)*1800(宽)*1(灰色单通道)
    x = np.zeros((n, HW, HW, 1), dtype=np.uint8)  # 初始化(20, 1800, 1800, 3)
    # y存储标签，每个y大小为1
    y = np.zeros(n, dtype=np.uint8)

    if not rate:
        for i, file in enumerate(image_dir):
            # file_name_seq[i] = file
            img = cv2.imread(os.path.join(listpath, file))  # (1800, 1800, 3)
            # xshape = img.shape
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img_gray_reshaped = img_gray.resize(HW, HW, 1)
            img_gray_reshaped = np.expand_dims(img_gray, axis=2)

            # xshape = img.shape
            # Xmid = img.shape[1]//2
            # 利用cv2.resize()函数将不同大小的图片统一为128(高)*128(宽) os.path.join作用是将两个路径拼接起来。路径+文件名
            x[i, :, :, 0] = cv2.resize(img_gray_reshaped, (HW, HW))
            y[i] = label

    else:
        for i, file in enumerate(image_dir):
            img = cv2.imread(os.path.join(listpath, file))
            # xshape = img.shape
            # Xmid = img.shape[1]//2
            # 利用cv2.resize()函数将不同大小的图片统一为128(高)*128(宽) os.path.join作用是将两个路径拼接起来。路径+文件名
            for j in range(rate):
                x[rate * i + j, :, :] = cv2.resize(img, (HW, HW))
                y[rate * i + j] = label
    return x, y


# training 时，通过来进行数据增强（data_abnor augmentation）
train_transform_weak = transforms.Compose([
    # transforms.RandomResizedCrop(150),
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    # transforms.ColorJitter(contrast=100),
    transforms.ToTensor()
    # transforms.Normalize(mean=[33.9580],
    #                      std=[68.0186])
])

train_transform_strong = transforms.Compose([
    # transforms.RandomResizedCrop(150),
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ColorJitter(contrast=100),
    transforms.ToTensor()
    # transforms.Normalize(mean=[33.9580],
    #                      std=[68.0186])
])

# testing 时，不需要进行数据增强（data_abnor augmentation）
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ColorJitter(contrast=100),
    transforms.ToTensor()
])

class ImgDataset(Dataset):

    def __init__(self, x, y=None, transform=None, lessTran=False):
        self.x = x  # x = none
        # label 需要是 LongTensor 型
        self.y = y  # y = none
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform  # transform = None
        self.lessTran = lessTran  # lessTran = False

        # 定义了五种不同的transform
        # 强制水平翻转
        self.trans0 = torchvision.transforms.Compose([
            transforms.ToPILImage(),
            torchvision.transforms.Resize(256),
            torchvision.transforms.RandomCrop(224),
            torchvision.transforms.RandomHorizontalFlip(p=1),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize([33.9580],
            #                                  [68.0186])
        ])
        # 强制垂直翻转
        self.trans1 = torchvision.transforms.Compose([
            transforms.ToPILImage(),
            torchvision.transforms.Resize(256),
            torchvision.transforms.RandomCrop(224),
            torchvision.transforms.RandomVerticalFlip(p=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([33.9580],
                                             [68.0186])
        ])
        # 旋转-90~90
        self.trans2 = torchvision.transforms.Compose([
            transforms.ToPILImage(), torchvision.transforms.Resize(256),
            torchvision.transforms.RandomCrop(224),
            torchvision.transforms.RandomRotation(90),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([33.9580],
                                             [68.0186])
        ])

        # 亮度在0-2之间增强，0是原图
        self.trans3 = torchvision.transforms.Compose([
            transforms.ToPILImage(), torchvision.transforms.Resize(256),
            torchvision.transforms.RandomCrop(224),
            torchvision.transforms.ColorJitter(brightness=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([33.9580],
                                             [68.0186])
        ])
        # 修改对比度，0-2之间增强，0是原图
        self.trans4 = torchvision.transforms.Compose([
            transforms.ToPILImage(), torchvision.transforms.Resize(256),
            torchvision.transforms.RandomCrop(224),
            torchvision.transforms.ColorJitter(contrast=2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([33.9580],
                                             [68.0186])
        ])
        # 颜色变化
        self.trans5 = torchvision.transforms.Compose([
            transforms.ToPILImage(), torchvision.transforms.Resize(256),
            torchvision.transforms.RandomCrop(224),
            torchvision.transforms.ColorJitter(hue=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([33.9580],
                                             [68.0186])
        ])
        # 混合
        self.trans6 = torchvision.transforms.Compose([
            transforms.ToPILImage(), torchvision.transforms.Resize(256),
            torchvision.transforms.RandomCrop(224),
            torchvision.transforms.ColorJitter(brightness=1, contrast=2, hue=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([33.9580],
                                             [68.0186])
        ])
        self.trans_list = [self.trans0, self.trans1, self.trans2, self.trans3, self.trans4, self.trans5, self.trans6]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]  # 获得第 index 的 x并进行赋值

        if self.y is not None:
            if self.lessTran:
                num = random.randint(0, 6)  # num --> one of (0, 1, 2, 3, 4, 5, 6)
                X = self.trans_list[num](X)  # 对X进行 增强 enhance
            else:
                if self.transform is not None:
                    X = self.transform(X)
            Y = self.y[index]
            return X, Y
        else:
            return X

    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
            image, label = self.__getitem__(index)  # 通过index 对 image和label赋值
            images.append(image)  # 将image 添加到 image_list中
            labels.append(label)  # 将label 添加到 label_list中
        return torch.stack(images), torch.tensor(labels)  # image 进行stack化、labels tensor


def getDateset(dir_class1, dir_class2, dir_class3, testSize=0.3, rate=None, testNum=None, lessTran=False):
    '''
    :param dir_class1:   这个是参数较少的那个
    :param dir_class2:
    :param testSize:
    :param rate:
    :param testNum:
    :return:
    '''
    x1, y1 = readjpgfile(dir_class1, 0, rate=rate)  # 类0是extra_acid (n1, 224, 224, 3)
    x2, y2= readjpgfile(dir_class2, 1)  # 类1是acid (n2, 224, 224, 3)
    x3, y3= readjpgfile(dir_class3, 2)  # 类2是fine (n3, 224, 224, 3)
    if testNum == -1:

        X = np.concatenate((x1, x2, x3))  # (n1+n2+n3, 224, 224, 3)
        Y = np.concatenate((y1, y2, y3))  # (n1+n2+n3)
        # todo: 我这里将图像增强给关掉了，不然图像由于特征增强反而丢失特性
        dataset_weak = ImgDataset(X, Y, transform=train_transform_weak, lessTran=False)
        dataset_strong = ImgDataset(X, Y, transform=train_transform_strong, lessTran=False)
        return dataset_weak, dataset_strong

    if not testNum:
        X = np.concatenate((x1, x2))
        Y = np.concatenate((y1, y2))
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=testSize, random_state=0)

    else:
        train_x1, test_x1, train_y1, test_y1 = train_test_split(x1, y1, test_size=testNum / len(y1), random_state=0)
        train_x2, test_x2, train_y2, test_y2 = train_test_split(x2, y2, test_size=testNum / len(y2), random_state=0)
        print(len(test_y2), len(test_y1))
        train_x = np.concatenate((train_x1, train_x2))
        test_x = np.concatenate((test_x1, test_x2))
        train_y = np.concatenate((train_y1, train_y2))
        test_y = np.concatenate((test_y1, test_y2))

# todo: 我这里将图像增强给关掉了，不然图像由于特征增强反而丢失特性
    train_dataset = ImgDataset(train_x, train_y, transform=None, lessTran=False)
    test_dataset = ImgDataset(test_x, test_y, transform=None, lessTran=False)

    # test_x1,test_y1 = readjpgfile(r'F:\li_XIANGMU\pycharm\deeplearning\cat_dog\catsdogs\test\Cat',0)  #猫是0
    # test_x2,test_y2 = readjpgfile(r'F:\li_XIANGMU\pycharm\deeplearning\cat_dog\catsdogs\test\Dog',1)
    # test_x = np.concatenate((test_x1,test_x2))
    # test_y = np.concatenate((test_y1,test_y2))

    return train_dataset, test_dataset


def smote(X_train, y_train):
    oversampler = SMOTE(sampling_strategy='auto', random_state=np.random.randint(100), k_neighbors=5, n_jobs=-1)
    os_X_train, os_y_train = oversampler.fit_resample(X_train, y_train)
    print('Resampled dataset shape {}'.format(Counter(os_y_train)))
    return os_X_train, os_y_train


def getDataLoader(class1path, class2path, class3path, batchSize, mode='train'):
    assert mode in ['train', 'val', 'test']
    if mode == 'train':
        train_set_weak, train_set_strong = getDateset(class1path, class2path, class3path, testNum=-1)  # return Dataset-- 七种不同的transform，X，Y
        # fixme
        """
        当 mode = 'train'时，会发生无法读取的问题
        """
        #todo 我这里保留两个dataloader 弱增强和强增强
        trainloader_weak = DataLoader(train_set_weak, batch_size=batchSize, shuffle=False)  # Dataset( batch_size =32 + train_set )
        trainloader_strong = DataLoader(train_set_strong, batch_size=batchSize, shuffle=False)  # Dataset( batch_size =32 + train_set )

        return trainloader_weak, trainloader_strong


    elif mode == 'test':
        testset_weak, test_strong = getDateset(class1path, class2path, class3path, testNum=-1)
        testLoader_weak = DataLoader(testset_weak, batch_size=batchSize, shuffle=False)
        testLoader_strong = DataLoader(test_strong, batch_size=batchSize, shuffle=False)
        return testLoader_weak, testLoader_strong



