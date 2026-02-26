"""代码三合一"""
import torch.nn as nn
import random

import argparse

import torch

import timm

# assert timm.__version__ == "0.6.11"  # version check

from mambaout import MambaOut as SE_Mambaout
from mambaout_withoutAttention import MambaOut
from mambaout_weak_strong import MambaOut as D_SE_Mambaout


from model_utils.data import getDataLoader
from model_utils.train_v2_nomatrix import train_VAL as train_VAL_SE
from model_utils.train_v2_without_attention_nomatrix import train_VAL as train_VAL_M
from model_utils.train_v3_nomatrix import train_VAL as train_VAL_DSE
from torch.amp import autocast, GradScaler
from model_utils.data_weak_strong import getDataLoader as getDataLoader_D
import os


def init_mamba_class(model_name, pretrained=False, weights_path=None, **kwargs):  # 新增weights_path参数

    # 权重初始化（保持不变）
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model = model_name(
        depths=[3, 3, 9, 3],
        dims=[48, 64, 192, 288],
        **kwargs
    ).to(device)

    # 加载预训练权重（支持本地文件）
    if pretrained:
        if weights_path:
            model.load_state_dict(torch.load(weights_path))
        else:
            model.apply(init_weights)
    print("Model weights initialized.")
    return model



##################################################################
# 1是<7.05 2<7.15 3其他



##########################################
# dataset_paths = {
#         'train': {
#             'class1': r'GAF/0wind/train',
#             'class2': r'GAF/1man/train',
#             'class3': r'GAF/2excavation/train',
#         },
#         'val': {
#             'class1': r'GAF/0wind/val',
#             'class2': r'GAF/1man/val',
#             'class3': r'GAF/2excavation/val',
#         },
#         'test': {
#             'class1': r'GAF/0wind/test',
#             'class2': r'GAF/1man/test',
#             'class3': r'GAF/2excavation/test',
#         }}
# dataset_paths = {
#         'train': {
#             'class1': r'MTF/0wind/train',
#             'class2': r'MTF/1man/train',
#             'class3': r'MTF/2excavation/train',
#         },
#         'val': {
#             'class1': r'MTF/0wind/val',
#             'class2': r'MTF/1man/val',
#             'class3': r'MTF/2excavation/val',
#         },
#         'test': {
#             'class1': r'MTF/0wind/test',
#             'class2': r'MTF/1man/test',
#             'class3': r'MTF/2excavation/test',
#         }}
##########################################
dataset_paths = {
        'train': {
            'class1': r'../Recurrent_map_DAS/0wind/train',
            'class2': r'../Recurrent_map_DAS/1manual/train',
            'class3': r'../Recurrent_map_DAS/2digger/train',
        },
        'val': {
            'class1': r'../Recurrent_map_DAS/0wind/val',
            'class2': r'../Recurrent_map_DAS/1manual/val',
            'class3': r'../Recurrent_map_DAS/2digger/val',
        },
        'test': {
            'class1': r'../Recurrent_map_DAS/0wind/test',
            'class2': r'../Recurrent_map_DAS/1manual/test',
            'class3': r'../Recurrent_map_DAS/2digger/test',
        }}


# 读数据这里按照自己的写法就行 。
#################################################################
random.seed(1)
learning_rate = 1e-5

# Assuming model is already defined and set to device
loss_fn = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification with class indices
# 之前使用soft，效果很差，现在model_head也没有用softmax


scaler = GradScaler()  # Mixed precision training

batch_size = 10

# w = 0.00001
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##################################################################

train_FHR = dataset_paths['train']
# train_UC = dataset_paths['UC']['train']

val_FHR = dataset_paths['val']
# val_UC = dataset_paths['UC']['val']

test_FHR = dataset_paths['test']
# test_UC = dataset_paths['UC']['test']



trainloader_FHR = getDataLoader(train_FHR['class1'], train_FHR['class2'], train_FHR['class3'], batchSize=batch_size)  # Dataset12（12为内存地址）: (batch_size, sample, X, Y)
# trainloader_UC = getDataLoader(train_UC['class1'], train_UC['class2'], train_UC['class3'], batchSize=batch_size)  # Dataset12: (batch_size, sample, X, Y)

valloader_FHR = getDataLoader(val_FHR['class1'], val_FHR['class2'], val_FHR['class3'], batchSize=batch_size, mode='test')  # Dataset3（12为内存地址）: (batch_size, sample, X, Y)
# valloader_UC = getDataLoader(val_UC['class1'], val_UC['class2'], val_UC['class3'], batchSize=batch_size, mode='test')  # Dataset3（12为内存地址）: (batch_size, sample, X, Y)

testloader_FHR = getDataLoader(test_FHR['class1'], test_FHR['class2'], test_FHR['class3'], batchSize=batch_size, mode='test')  # Dataset3（12为内存地址）: (batch_size, sample, X, Y)
# testloader_UC = getDataLoader(test_UC['class1'], test_UC['class2'], test_UC['class3'], batchSize=batch_size, mode='test')  # Dataset3（12为内存地址）: (batch_size, sample, X, Y)



savePath_Mamba = r'../models_4.25/Mambaout'
savePath_SEMamba = r'../models_4.25/SE-Mambaout'
savePath_DSEMamba = r'../models_4.25/D-SE-Mambaout'
#
# savePath_Mamba = r'model_save/Mambaout/GAF'
# savePath_SEMamba = r'model_save/SE-Mambaout/GAF'
# savePath_DSEMamba = r'model_save/D-SE-Mambaout/GAF'

trainloader_FHR_weak, trainloader_FHR_strong = getDataLoader_D(train_FHR['class1'], train_FHR['class2'], train_FHR['class3'], batchSize=batch_size)  # Dataset12（12为内存地址）: (batch_size, sample, X, Y)
# trainloader_UC = getDataLoader(train_UC['class1'], train_UC['class2'], train_UC['class3'], batchSize=batch_size)  # Dataset12: (batch_size, sample, X, Y)

valloader_FHR_weak, valloader_FHR_strong = getDataLoader_D(val_FHR['class1'], val_FHR['class2'], val_FHR['class3'], batchSize=batch_size, mode='test')  # Dataset3（12为内存地址）: (batch_size, sample, X, Y)
# valloader_UC = getDataLoader(val_UC['class1'], val_UC['class2'], val_UC['class3'], batchSize=batch_size, mode='test')  # Dataset3（12为内存地址）: (batch_size, sample, X, Y)

testloader_FHR_weak, testloader_strong = getDataLoader_D(test_FHR['class1'], test_FHR['class2'], test_FHR['class3'], batchSize=batch_size, mode='test')  # Dataset3（12为内存地址）: (batch_size, sample, X, Y)
# testloader_UC = getDataLoader(test_UC['class1'], test_UC['class2'], test_UC['class3'], batchSize=batch_size, mode='test')  # Dataset3（12为内存地址）: (batch_size, sample, X, Y)


epoch = 1

if __name__ == '__main__':
    print(f'\n 当前是Mambaout')
    # model = init_mamba_class(pretrained=True, weights_path=r'D:\lnc_project\Mamba\MambaOut-main\DAS\model_save\epoch_with_attention\4\DASFinefinal.pth')
    model_Mambaout = init_mamba_class(MambaOut,pretrained=False)
    model_Mambaout.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model_Mambaout.parameters(), lr=1e-5, weight_decay=1e-4)  # Add weight decay

    train_VAL_M(model_Mambaout, trainloader_FHR, valloader_FHR, testloader_FHR,
              optimizer, loss_fn, batch_size,
              num_epoch=epoch, device=device, save_=savePath_Mamba)

    print(f'\n 当前是SE_Mambaout')
    model_SEMambaout = init_mamba_class(SE_Mambaout,pretrained=False)
    model_SEMambaout.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model_SEMambaout.parameters(), lr=1e-5, weight_decay=1e-4)  # Add weight decay

    train_VAL_SE(model_SEMambaout, trainloader_FHR, valloader_FHR, testloader_FHR,
              optimizer, loss_fn, batch_size,
              num_epoch=epoch, device=device, save_=savePath_SEMamba)

    print(f'\n 当前是D_SE_Mambaout')
    model_DSEMambaout = init_mamba_class(D_SE_Mambaout,pretrained=False)
    model_DSEMambaout.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model_DSEMambaout.parameters(), lr=1e-5, weight_decay=1e-4)  # Add weight decay

    train_VAL_DSE(model_DSEMambaout, trainloader_FHR_weak, trainloader_FHR_strong, valloader_FHR_weak, valloader_FHR_strong, testloader_FHR_weak, testloader_strong,
              optimizer, loss_fn, batch_size,
              num_epoch=epoch, device=device, save_=savePath_DSEMamba)
    # modelpath1 = savePath+'max'
    # model1 = torch.load(modelpath1)

    # test(model1, test_set=test_dataset)

    # modelpath2 = savePath+'final'

    # model2 = torch.load(modelpath2)
    # test(model2, test_set=test_dataset)

