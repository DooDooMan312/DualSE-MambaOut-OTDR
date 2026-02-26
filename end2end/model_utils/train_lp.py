import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
import models_vit
from timm.models.layers import trunc_normal_

#fixme：暂停部分，进行一次step into，看看具体是哪里数据格式为 1， 769  每个 channel应该只有一个 value
#fixme: patch_embeded

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

    trunc_normal_(model.head.weight, std=0.01)

    return model


# 更新学习率


def train_VAL(model, train_set, val_set, optimizer, loss, batch_size, w, num_epoch, device, save_):
    # train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0)
    # val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=True,num_workers=0)
    train_loader = train_set
    val_loader = val_set
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
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()  # 用 optimizer 将模型参数的梯度 gradient 归零

############################################
            # # a = data[0]  #    将维度由(1, 3, 224, 224) -> (3, 224, 224)
            # data = data[0]
            # data_squeezed = data.squeeze(0)
############################################

            train_pred = model(data[0].to(device))  # 利用 model_utils 得到预测的概率分布，这边实际上是调用模型的 forward 函数
            # batch_loss = loss(train_pred, data[1].cuda(), w, model) # 计算 loss （注意 prediction 跟 label 必须同时在 CPU 或是 GPU 上）
            batch_loss = loss(train_pred, data[1].to(device))
            batch_loss.backward()  # 利用 back propagation 算出每个参数的 gradient
            optimizer.step()  # 以 optimizer 用 gradient 更新参数

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

        # 验证集val
        model.eval()

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred = model(data[0].to(device))
                # batch_loss = loss(val_pred, data[1].cuda(),w, model)
                batch_loss = loss(val_pred, data[1].to(device))
                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                val_loss += batch_loss.item()

            if val_acc > maxacc:
                torch.save(model, save_ + 'max')
                maxacc = val_acc
                # torch.save({'epoch': epoch + 1, 'state_dict': model_utils.state_dict(), 'best_loss': val_loss,
                #             'optimizer': optimizer.state_dict(),'alpha': loss.alpha, 'gamma': loss.gamma},
                #            'cat_dog_res18')
                # 保存用于画图
            plt_train_acc.append(train_acc / train_set.dataset.__len__())
            plt_train_loss.append(train_loss / train_set.dataset.__len__())
            plt_val_acc.append(val_acc / val_set.dataset.__len__())
            plt_val_loss.append(val_loss / val_set.dataset.__len__())

            # 将结果 print 出來
            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                  (epoch + 1, num_epoch, time.time() - epoch_start_time,
                   plt_train_acc[-1], plt_train_loss[-1], plt_val_acc[-1], plt_val_loss[-1]))

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