import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset

from collections import OrderedDict
from torch.amp import autocast, GradScaler

scaler = GradScaler()  # 混合精度训练


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)

    # model
    parser.add_argument('--nb_classes', default=3, type=int,
                        help='number of the classfication types')
    parser.add_argument('--drop_path', default=0.1, type=float, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)

    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    # path
    parser.add_argument('--predModelPath', default='model_save/mae_pretrain_vit_base.pth',
                        help='finetune from checkpoint')

    return parser


args = get_args_parser()
args = args.parse_args()



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 3, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.225, 0.225, 0.225), 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'mambaout_femto': _cfg(
        url='https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_femto.pth'),
    'mambaout_kobe': _cfg(
        url='https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_kobe.pth'),
    'mambaout_tiny': _cfg(
        url='https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_tiny.pth'),
    'mambaout_small': _cfg(
        url='https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_small.pth'),
    'mambaout_base': _cfg(
        url='https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_base.pth'),
}

#
# def initMambaClass(args, pretrained=False):  # 初始化 Mamba
#     model = MambaOut.__dict__[args.model](
#         num_classes=args.nb_classes,
#         drop_path_rate=args.drop_path,
#     )
#     model.default_cfg = default_cfgs['mambaout_base']
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(
#             url=model.default_cfg['url'], map_location="cpu", check_hash=True)
#         model.load_state_dict(state_dict)
#         print(state_dict)
#     return model
# #     model = MambaOut(
# #         depths=[3, 4, 27, 3],
# #         dims=[128, 256, 512, 768],
# #         **kwargs)
# #     model.default_cfg = default_cfgs['mambaout_base']
# #     if pretrained:
# #         state_dict = torch.hub.load_state_dict_from_url(
# #             url=model.default_cfg['url'], map_location="cpu", check_hash=True)
# #         model.load_state_dict(state_dict)
# #     return model
#
#
# # 更新学习率

#fixme 3.25 20:53 -- 研究如何实现双通道输入
def train_VAL(model, train_set1, train_set2, val_set1, val_set2, optimizer, loss, batch_size, w, num_epoch, device, save_):
    # train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0)
    # val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=True,num_workers=0)

    ############        双通道输入train，FHR-=-UC       ##########
    train_loader1 = train_set1
    train_loader2 = train_set2

    val_loader1 = val_set1
    val_loader2 = val_set2
    ###########################################
    # 用测试集训练模型model(),用验证集作为测试集来验证
    plt_train_loss = []
    plt_val_loss = []
    plt_train_acc = []
    plt_val_acc = []
    maxacc = 0
    mixup_fn = None

    ###############################################

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc, train_loss, val_acc, val_loss = 0.0, 0.0, 0.0, 0.0

        model.train()

        for (data1, target1), (data2, target2) in zip(train_loader1, train_loader2):
            input1, input2 = data1.to(device), data2.to(device)
            targets = target1.to(device)  # 只用 target1 作为标签

            # # 处理 mixup
            # if mixup_fn is not None:
            #     inputs1, targets = mixup_fn(inputs1, targets)

            optimizer.zero_grad()  # 用 optimizer 将模型参数的梯度 gradient 归零


            train_pred = model(input1, input2)  # 利用 model_utils 得到预测的概率分布，这边实际上是调用模型的 forward 函数
            # batch_loss = loss(train_pred, data[1].cuda(), w, model) # 计算 loss （注意 prediction 跟 label 必须同时在 CPU 或是 GPU 上）
            batch_loss = loss(train_pred, targets)

            batch_loss.backward()  # 利用 back propagation 算出每个参数的 gradient
            optimizer.step()  # 以 optimizer 用 gradient 更新参数


            train_acc += (train_pred.argmax(dim=1) == targets).sum().item()
            train_loss += batch_loss.item()

        # 计算训练集的平均损失和准确率
        train_loss /= len(train_loader1)
        train_acc /= len(train_set1)

        # 验证集
        loss_fn = nn.CrossEntropyLoss()  # 适用于分类任务
        model.eval()
        with torch.no_grad():
            for (data1, target1), (data2, target2) in zip(val_loader1, val_loader2):
                input1, input2 = data1.to(device), data2.to(device)
                targets = target1.to(device)

                val_pred = model(input1, input2)
                batch_loss = loss_fn(val_pred, targets)

                val_acc += (val_pred.argmax(dim=1) == targets).sum().item()
                val_loss += batch_loss.item()

            # 计算验证集的平均损失和准确率
            val_loss /= len(val_loader1)
            val_acc /= len(val_set1)
            # 反向传播

        scaler.scale(batch_loss).backward()

        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # 计算训练准确率
        _, predicted = torch.max(outputs, 1)
        if mixup_fn is None:
            train_acc += (predicted == targets.argmax(dim=1)).sum().item()  # 适应 one-hot
        train_loss += batch_loss.item()

        # 记录训练结果
        plt_train_acc.append(train_acc)
        plt_train_loss.append(train_loss)
        plt_val_acc.append(val_acc)
        plt_val_loss.append(val_loss)

        # 保存最优模型
        if val_acc > maxacc:
            torch.save(model.state_dict(), save_ + 'best.pth')
            maxacc = val_acc

        elapsed_time = time.time() - epoch_start_time
        print(f'[{epoch + 1:03d}/{num_epoch}] {elapsed_time:.2f}s '
              f'Train Acc: {train_acc:.6f}, Loss: {train_loss:.6f} | '
              f'Val Acc: {val_acc:.6f}, Loss: {val_loss:.6f}')

    # 保存最终模型
    torch.save(model.state_dict(), save_ + 'final.pth')

    # 画训练曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(plt_train_acc, label='Train Acc')
    plt.plot(plt_val_acc, label='Val Acc')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(plt_train_loss, label='Train Loss')
    plt.plot(plt_val_loss, label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

    ###############################################
    #
    # for epoch in range(num_epoch):
    #     # update_lr(optimizer,epoch)
    #     epoch_start_time = time.time()
    #     train_acc = 0.0
    #     train_loss = 0.0
    #     val_acc = 0.0
    #     val_loss = 0.0
    #
    #     model.train()  # 确保 model_utils 是在 训练 model_utils (开启 Dropout 等...)
    #
    #     for i, data in enumerate(train_loader):
    #         optimizer.zero_grad()  # 用 optimizer 将模型参数的梯度 gradient 归零
    #
    #         train_pred = model(data[0].to(device))  # 利用 model_utils 得到预测的概率分布，这边实际上是调用模型的 forward 函数
    #         # batch_loss = loss(train_pred, data[1].cuda(), w, model) # 计算 loss （注意 prediction 跟 label 必须同时在 CPU 或是 GPU 上）
    #         batch_loss = loss(train_pred, data[1].to(device))
    #         batch_loss.backward()  # 利用 back propagation 算出每个参数的 gradient
    #         optimizer.step()  # 以 optimizer 用 gradient 更新参数
    #
    #         train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
    #         train_loss += batch_loss.item()
    #
    #     # 验证集val
    #     model.eval()
    #
    #     with torch.no_grad():
    #         for i, data in enumerate(val_loader):
    #             val_pred = model(data[0].to(device))
    #             # batch_loss = loss(val_pred, data[1].cuda(),w, model)
    #             batch_loss = loss(val_pred, data[1].to(device))
    #             val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
    #             val_loss += batch_loss.item()
    #
    #         if val_acc > maxacc:
    #             torch.save(model, save_ + 'max')
    #             maxacc = val_acc
    #             # torch.save({'epoch': epoch + 1, 'state_dict': model_utils.state_dict(), 'best_loss': val_loss,
    #             #             'optimizer': optimizer.state_dict(),'alpha': loss.alpha, 'gamma': loss.gamma},
    #             #            'cat_dog_res18')
    #             # 保存用于画图
    #         plt_train_acc.append(train_acc / train_set.dataset.__len__())
    #         plt_train_loss.append(train_loss / train_set.dataset.__len__())
    #         plt_val_acc.append(val_acc / val_set.dataset.__len__())
    #         plt_val_loss.append(val_loss / val_set.dataset.__len__())
    #
    #         # 将结果 print 出來
    #         print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
    #               (epoch + 1, num_epoch, time.time() - epoch_start_time,
    #                plt_train_acc[-1], plt_train_loss[-1], plt_val_acc[-1], plt_val_loss[-1]))
    #
    #     if epoch == num_epoch - 1:
    #         torch.save(model, save_ + 'final')
    #
    # # Loss曲线
    # plt.plot(plt_train_loss)
    # plt.plot(plt_val_loss)
    # plt.title('Loss')
    # plt.legend(['train', 'val'])
    # plt.savefig('loss.png')
    # plt.show()
    #
    # # Accuracy曲线
    # plt.plot(plt_train_acc)
    # plt.plot(plt_val_acc)
    # plt.title('Accuracy')
    # plt.legend(['train', 'val'])
    # plt.savefig('acc.png')
    # plt.show()