import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
import models_vit


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


def initMaeClass(args):
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    checkpoint = torch.load(args.predModelPath, map_location='cpu')

    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    return model


# 更新学习率


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

    for epoch in range(num_epoch):
        # update_lr(optimizer,epoch)
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()  # 确保 model_utils 是在 训练 model_utils (开启 Dropout 等...)

        for (data1, target1), (data2, target2) in zip(train_loader1, train_loader2):
            input1, input2 = data1.to(device), data2.to(device)
            targets = target1.to(device)  # 只用 target1 作为标签


            optimizer.zero_grad()  # 用 optimizer 将模型参数的梯度 gradient 归零

            train_pred = model(input1, input2)  # 利用 model_utils 得到预测的概率分布，这边实际上是调用模型的 forward 函数
            # batch_loss = loss(train_pred, data[1].cuda(), w, model) # 计算 loss （注意 prediction 跟 label 必须同时在 CPU 或是 GPU 上）
            batch_loss = loss(train_pred, targets)

            batch_loss.backward()  # 利用 back propagation 算出每个参数的 gradient
            optimizer.step()  # 以 optimizer 用 gradient 更新参数

            # train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            # train_loss += batch_loss.item()


            train_acc += (train_pred.argmax(dim=1) == targets).sum().item()
            train_loss += batch_loss.item()

        # 计算训练集的平均损失和准确率
        train_loss /= len(train_loader1)
        train_acc /= len(train_set1)
        # 验证集val
        model.eval()
        loss_fn = nn.CrossEntropyLoss()  # 适用于分类任务


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

            # 保存最佳模型
            if val_acc > maxacc:
                torch.save(model, save_ + 'max')
                maxacc = val_acc

            # 记录损失和准确率
            plt_train_acc.append(train_acc)
            plt_train_loss.append(train_loss)
            plt_val_acc.append(val_acc)
            plt_val_loss.append(val_loss)

            # 打印训练进度
            print(f'[{epoch+1:03d}/{num_epoch}] {time.time() - epoch_start_time:.2f} sec(s) '
                  f'Train Acc: {train_acc:.6f} Loss: {train_loss:.6f} | '
                  f'Val Acc: {val_acc:.6f} Loss: {val_loss:.6f}')
        if epoch == num_epoch - 1:
            torch.save(model, save_ + 'final')

    # Loss曲线
    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title('Loss')
    plt.legend(['train', 'val'])
    plt.savefig('loss.png')
    plt.show()

    # Accuracy曲线
    plt.plot(plt_train_acc)
    plt.plot(plt_val_acc)
    plt.title('Accuracy')
    plt.legend(['train', 'val'])
    plt.savefig('acc.png')
    plt.show()