# -*- coding: utf-8 -*-
# @Time    : 2025/4/25 18:54
# @Author  : XXX
# @Site    : 
# @File    : train_resnet50.py
# @Software: PyCharm 
# @Comment :

"""
这是只有对比度增强的版本
"""

import os
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
import time
import numpy as np
import pandas as pd
import argparse
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


from collections import OrderedDict
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR

scaler = GradScaler()

# 在每个epoch结束时，除了现有的accuracy和loss外，还计算并记录其他评价指标
def compute_metrics(y_true, y_pred):
    """计算 FPR, Precision, Recall, F1 Score"""
    labels = np.unique(np.concatenate((y_true, y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FPR = FP / (FP + TN + 1e-8)  # 避免除 0
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {'FPR': FPR, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}


def save_loss_acc_to_csv(epoch, train_loss, train_acc, val_loss, val_acc, elapsed_time, num_epoch):
    # Check if the file exists to decide whether to append or create a new file
    file_path_loss = f'model_save/{num_epoch}_loss.csv'
    file_path_acc = f'model_save/{num_epoch}_acc.csv'
    file_exists_loss = os.path.exists(file_path_loss)
    file_exists_acc = os.path.exists(file_path_acc)

    # Save loss to CSV
    df_train_loss = pd.DataFrame({
        'epoch': [epoch],  # Wrap epoch in a list
        'elapsed_time': [elapsed_time],  # Wrap elapsed_time in a list
        'train_loss': [train_loss],  # Wrap train_loss in a list
        'val_loss': [val_loss]  # Wrap val_loss in a list
    })
    df_train_loss.to_csv(file_path_loss, mode='a', header=not file_exists_loss, index=False)

    # Save accuracy to CSV
    df_train_acc = pd.DataFrame({
        'epoch': [epoch],  # Wrap epoch in a list
        'elapsed_time': [elapsed_time],  # Wrap elapsed_time in a list
        'train_acc': [train_acc],  # Wrap train_acc in a list
        'val_acc': [val_acc]  # Wrap val_acc in a list
    })
    df_train_acc.to_csv(file_path_acc, mode='a', header=not file_exists_acc, index=False)

    print(f"Epoch {epoch} - Loss and Accuracy saved to CSV.")
# 可视化混淆矩阵
# def plot_confusion_matrix(cm, title, classes, normalize=False):
#     """绘制混淆矩阵"""
#     plt.figure(figsize=(8, 8))
#     plt.xlim(-0.5, cm.shape[1] - 0.5)
#     plt.ylim(cm.shape[0] - 0.5, -0.5)
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title(title, fontsize=22)
#     plt.colorbar()
#
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45, fontsize=18)
#     plt.yticks(tick_marks, classes, fontsize=18)
#
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     if cm.size > 0:
#         thresh = cm.max() / 2.
#     else:
#         print("Error: Confusion matrix is empty!")
#
#     for i, j in np.ndindex(cm.shape):
#         plt.text(j, i, format(cm[i, j], '.2f'),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black", fontsize=20)
#     # 设置坐标轴标签字体大小
#     plt.xlabel('Predicted label', fontsize=22)
#     plt.ylabel('True label', fontsize=22)
#     plt.show()

def plot_confusion_matrix(cm, title, classes, normalize=False):
    """绘制混淆矩阵（优化版V2）"""
    plt.figure(figsize=(8, 8))
    # plt.xlim(-0.5, cm.shape[1] - 0.5)
    # plt.ylim(cm.shape[0] - 0.5, -0.5)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=22, pad=20)  # 增加标题与图形的间距

    # 设置colorbar字体大小
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20, pad=5)  # 调整colorbar间距

    tick_marks = np.arange(len(classes))

    # 设置刻度参数（内侧刻度）
    plt.tick_params(
        axis='x',
        which='both',
        direction='in',  # 刻度线朝内
        labelrotation=45,
        labelsize=18,
        pad=12  # 刻度标签与轴的距离
    )
    plt.tick_params(
        axis='y',
        which='both',
        direction='in',
        labelsize=18,
        pad=12
    )

    # 设置刻度标签
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # 处理归一化并转换数据类型
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm * 100).astype(int)  # 转换为百分比整数
        print("Normalized confusion matrix (percentage)")
    else:
        cm = cm.astype(int)
        print('Confusion matrix, without normalization')

    if cm.size > 0:
        thresh = cm.max() / 2.
    else:
        print("Error: Confusion matrix is empty!")
        return

    # 在单元格中心添加文本（优化对齐方式）
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j, i,  # 单元格中心坐标
            format(cm[i, j], 'd'),
            ha='center',  # 水平居中
            va='center',  # 垂直居中
            color="white" if cm[i, j] > thresh else "black",
            fontsize=20,
            fontweight='bold'  # 加粗文本
        )

    plt.xlabel('Predicted label', fontsize=22, labelpad=15)  # 增加标签间距
    plt.ylabel('True label', fontsize=22, labelpad=15)
    plt.tight_layout()
    plt.show()

# Training loop with improvements
def train_VAL(model, train_loader1, val_loader1, test_loader1, optimizer, loss_fn, batch_size, num_epoch, device,
              save_):
    # Data loaders for dual-channel input
    # 但这里 train_set 和 val_set 已经在main程序中package到了dataloader中
    ################################
    # train_loader1 = DataLoader(train_set1, batch_size=batch_size, shuffle=True, num_workers=4)
    # train_loader2 = DataLoader(train_set2, batch_size=batch_size, shuffle=True, num_workers=4)

    # val_loader1 = DataLoader(val_set1, batch_size=batch_size, shuffle=False, num_workers=4)
    # val_loader2 = DataLoader(val_set2, batch_size=batch_size, shuffle=False, num_workers=4)
    ################################
    # Metrics and tracking lists
    plt_train_loss, plt_val_loss = [], []
    plt_train_acc, plt_val_acc = [], []

    # elapsed_time_list = []
    # train_loss_list, train_acc_list = [], []
    # val_loss_list, val_acc_list = [], []

    max_acc = 0

    best_val_loss = float('inf')
    patience = 5  # number of epochs to wait for improvement
    epochs_without_improvement = 0
    # Learning Rate Scheduler (optional)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    for epoch in range(num_epoch):
        model.train() # 启动训练模式，权重可调
        epoch_start_time = time.time()

        train_acc, train_loss = 0.0, 0.0
        val_acc, val_loss = 0.0, 0.0

        # 用于存储训练集和验证集的真实标签和预测标签
        y_true_train, y_pred_train = [], []
        y_true_val, y_pred_val = [], []


        # Train loop
        for data1, target1  in train_loader1:
            # fixme: 这里只是(1, 224, 224)，少一个维度，processed
            input1 = data1.to(device)  # input：tensor(batch_size, channels, 224, 224)  ## input:(1, 224, 224)
            # fixme: 2025.4.2 大问题——input1和input2不是同一类，只好把shuffle给关了
            targets = target1.to(device)  # input：tensor(channels,)

            optimizer.zero_grad()

            with autocast(device_type=device.type):  # Automatic Mixed Precision (AMP)
                train_pred = model(input1)  # Forward pass train_pred: tensor:(channels, 3), tensor([[1.1553, 2.75, -0.6338],\n [0.9, 1.9, -0.3]])
                batch_loss = loss_fn(train_pred, targets)  # input

                # # Check for NaN or Inf in loss
                # if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                #     print("NaN or Inf detected in loss")
                #     continue

            scaler.scale(batch_loss).backward()  # Backpropagation
            scaler.step(optimizer)  # Update model parameters
            scaler.update()  # Update the scaler

            #######################################
            # 收集训练集的真实标签和预测标签
            _, predicted = torch.max(train_pred, 1)  # torch.max(tensor, dim)；即提取tensor 第一个维度的max，返回 maximum和对应的index(类)
            # _: 为train_predict的最大值——1.4385；predicted:就是maximum对应的类
            y_true_train.extend(targets.cpu().numpy())
            y_pred_train.extend(predicted.cpu().numpy())

            # 计算准确率
            train_loss += batch_loss.item()
            train_acc += (predicted == targets).sum().item() / len(targets)

            #######################################
        ##############计算混淆矩阵和分类
        # 计算混淆矩阵及分类报告
        # Compute metrics and plot confusion matrix
        train_cm = confusion_matrix(y_true_train, y_pred_train)
        # train_metrics = compute_metrics(y_true_train, y_pred_train)

        train_loss /= len(train_loader1)
        train_acc /= len(train_loader1)

        # Validation loop
        model.eval()
        with torch.no_grad():
            for data1, target1 in val_loader1:
                input1 = data1.to(device)
                targets = target1.to(device)

                val_pred = model(input1)  # Forward pass
                batch_loss = loss_fn(val_pred, targets)

                val_loss += batch_loss.item()
                val_acc += (val_pred.argmax(dim=1) == targets).sum().item()

            _, predicted = torch.max(val_pred, 1)  # torch.max(tensor, dim)；即提取tensor 第一个维度的max，返回 maximum和对应的index(类)
            # _: 为val_pred的最大值——1.4385；predicted:就是maximum对应的类
            y_true_val.extend(targets.cpu().numpy())
            y_pred_val.extend(predicted.cpu().numpy())

            # Calculate average validation metrics
            val_loss /= len(val_loader1)
            val_acc /= len(val_loader1.dataset)  # 要除整体数据的数量

            ##############计算混淆矩阵和分类
            # 计算混淆矩阵及分类报告
            val_cm = confusion_matrix(y_true_val, y_pred_val)
            val_metrics = compute_metrics(y_true_val, y_pred_val)
            scheduler.step(val_loss)

            # Optional: print current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[Epoch {epoch}] val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, lr: {current_lr:.2e}")
            ##############计算混淆矩阵和分类

        ################
        classes = ['Wind', 'Man-made', 'Excavation']
        plot_confusion_matrix(train_cm, "Training Confusion Matrix", classes)
        plot_confusion_matrix(val_cm, "Validation Confusion Matrix", classes)

        # Log metrics and print
        plt_train_acc.append(train_acc)
        plt_train_loss.append(train_loss)
        plt_val_acc.append(val_acc)
        plt_val_loss.append(val_loss)

        # Save best model based on validation accuracy
        if val_acc > max_acc:
            torch.save(model.state_dict(), save_ + 'best.pth')
            max_acc = val_acc

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_model.pth")  # Save the best model
        # else:
        #     epochs_without_improvement += 1
        #     if epochs_without_improvement >= patience:
        #         print("Early stopping...")
        #         break  # Stop training early if no improvement in validation loss

        elapsed_time = time.time() - epoch_start_time
        print(f'[{epoch + 1:03d}/{num_epoch}] {elapsed_time:.2f}s '
              f'Train Acc: {train_acc:.6f}, Loss: {train_loss:.6f} | '
              f'Val Acc: {val_acc:.6f}, Loss: {val_loss:.6f}')

        save_loss_acc_to_csv(epoch + 1, train_loss, train_acc, val_loss, val_acc, elapsed_time, num_epoch)

        # # # 将每个 epoch 的损失和准确率保存到列表中
        # # train_loss_list.append(train_loss)
        # # train_acc_list.append(train_acc)
        # #
        # # val_loss_list.append(val_loss)
        # # val_acc_list.append(val_acc)
        # # elapsed_time_list.append(elapsed_time)
        # # Check if the file exists to decide whether to append or create a new file
        # file_path_loss = f'model_save/{num_epoch}_loss.csv'
        # file_path_acc = f'model_save/{num_epoch}_acc.csv'
        # file_exists_loss = os.path.exists(file_path_loss)
        # file_exists_acc = os.path.exists(file_path_acc)
        #
        # # Save loss to CSV
        # df_train_loss = pd.DataFrame({
        #     'epoch': epoch,
        #     'elapsed_time': elapsed_time,
        #     'train_loss': train_loss,
        #     'val_loss': val_loss
        # })
        # df_train_loss.to_csv(f'model_save/{num_epoch}_loss.csv', mode='a', header=not file_exists_loss, index=False)
        #
        # # Save accuracy to CSV
        # df_train_acc = pd.DataFrame({
        #     'epoch': epoch,
        #     'elapsed_time': elapsed_time,
        #     'train_acc': train_acc,
        #     'val_acc': val_acc
        # })
        # df_train_acc.to_csv(f'model_save/{num_epoch}_acc.csv', mode='a', header=not file_exists_acc, index=False)
        #
        # # save_loss_acc2csv(train_loss_list, train_acc_list, val_loss_list, val_acc_list, elapsed_time_list, name="epoch_150")
        #
        # # 保存为 CSV 文件


    # Save final model
    torch.save(model.state_dict(), save_ + 'final.pth')

    # Plot training and validation curves
    plt.figure(figsize=(12, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(plt_train_acc, label='Train Accuracy')
    plt.plot(plt_val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(plt_train_loss, label='Train Loss')
    plt.plot(plt_val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

    # 测试集评估
    model.eval()
    test_acc, test_loss = 0.0, 0.0
    y_true_test, y_pred_test = [], []
    test_loss_list, test_acc_list = [], []

    with torch.no_grad():

        test_start = time.time()
        for data1, target1 in test_loader1:
            input1 = data1.to(device)
            targets = target1.to(device)

            test_pred = model(input1)  # 前向传播
            batch_loss = loss_fn(test_pred, targets)

            # 收集测试集的真实标签和预测标签
            _, predicted = torch.max(test_pred.data, 1)
            y_true_test.extend(targets.cpu().numpy())
            y_pred_test.extend(predicted.cpu().numpy())

            # 计算准确率
            test_acc += (test_pred.argmax(dim=1) == targets).sum().item()
            test_loss += batch_loss.item()

        test_time = time.time() - test_start
        # 计算平均测试指标
        test_loss /= len(test_loader1)
        test_acc /= len(test_loader1.dataset)

        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        # 计算测试集的混淆矩阵及分类报告
        test_cm = confusion_matrix(y_true_test, y_pred_test)
        classification_report(y_true_test, y_pred_test, output_dict=True)

        # 打印测试集的分类报告
        print("Test Classification Report:")
        print(classification_report(y_true_test, y_pred_test))

        classes = ['Wind', 'Man-made', 'Excavation']
        # 可视化测试集的混淆矩阵
        plot_confusion_matrix(test_cm, "Test Confusion Matrix", classes)

        # 打印测试集的最终指标
        print(f'Test Acc: {test_acc:.6f}, Loss: {test_loss:.6f}, test_time:{test_time:.6f}')


# # Training loop with improvements
# def train_VAL(model, train_loader1, train_loader2, val_loader1, val_loader2, test_loader1, test_loader2,optimizer, loss_fn, batch_size, num_epoch, device,
#               save_):
#     # Data loaders for dual-channel input
#     # 但这里 train_set 和 val_set 已经在main程序中package到了dataloader中
#     ################################
#     # train_loader1 = DataLoader(train_set1, batch_size=batch_size, shuffle=True, num_workers=4)
#     # train_loader2 = DataLoader(train_set2, batch_size=batch_size, shuffle=True, num_workers=4)
#
#     # val_loader1 = DataLoader(val_set1, batch_size=batch_size, shuffle=False, num_workers=4)
#     # val_loader2 = DataLoader(val_set2, batch_size=batch_size, shuffle=False, num_workers=4)
#     ################################
#     # Metrics and tracking lists
#     plt_train_loss, plt_val_loss = [], []
#     plt_train_acc, plt_val_acc = [], []
#     max_acc = 0
#
#     best_val_loss = float('inf')
#     patience = 5  # number of epochs to wait for improvement
#     epochs_without_improvement = 0
#     # Learning Rate Scheduler (optional)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
#
#     for epoch in range(num_epoch):
#         model.train() # 启动训练模式，权重可调
#         epoch_start_time = time.time()
#
#         train_acc, train_loss = 0.0, 0.0
#         val_acc, val_loss = 0.0, 0.0
#
#         # 用于存储训练集和验证集的真实标签和预测标签
#         y_true_train, y_pred_train = [], []
#         y_true_val, y_pred_val = [], []
#
#         # Train loop
#         for (data1, target1), (data2, target2) in zip(train_loader1, train_loader2):
#             input1, input2 = data1.to(device), data2.to(device)  # input：tensor(batch_size, channels, 224, 224)
#             # fixme: 2025.4.2 大问题——input1和input2不是同一类，只好把shuffle给关了
#             targets = target1.to(device)  # input：tensor(channels,)
#
#             optimizer.zero_grad()
#
#             with autocast(device_type=device.type):  # Automatic Mixed Precision (AMP)
#                 train_pred = model(input1, input2)  # Forward pass train_pred: tensor:(channels, 3), tensor([[1.1553, 2.75, -0.6338],\n [0.9, 1.9, -0.3]])
#                 batch_loss = loss_fn(train_pred, targets)  # input
#
#                 # # Check for NaN or Inf in loss
#                 # if torch.isnan(batch_loss) or torch.isinf(batch_loss):
#                 #     print("NaN or Inf detected in loss")
#                 #     continue
#
#             scaler.scale(batch_loss).backward()  # Backpropagation
#             scaler.step(optimizer)  # Update model parameters
#             scaler.update()  # Update the scaler
#
#             #######################################
#             # 收集训练集的真实标签和预测标签
#             _, predicted = torch.max(train_pred, 1)  # torch.max(tensor, dim)；即提取tensor 第一个维度的max，返回 maximum和对应的index(类)
#             # _: 为train_predict的最大值——1.4385；predicted:就是maximum对应的类
#             y_true_train.extend(targets.cpu().numpy())
#             y_pred_train.extend(predicted.cpu().numpy())
#
#             # 计算准确率
#             train_loss += batch_loss.item()
#             train_acc += (predicted == targets).sum().item() / len(targets)
#
#             #######################################
#         ##############计算混淆矩阵和分类
#         # 计算混淆矩阵及分类报告
#         # Compute metrics and plot confusion matrix
#         train_cm = confusion_matrix(y_true_train, y_pred_train)
#         # train_metrics = compute_metrics(y_true_train, y_pred_train)
#
#
#         train_loss /= len(train_loader1)
#         train_acc /= len(train_loader1)
#
#         # Validation loop
#         model.eval()
#         with torch.no_grad():
#             for (data1, target1), (data2, target2) in zip(val_loader1, val_loader2):
#                 input1, input2 = data1.to(device), data2.to(device)
#                 targets = target1.to(device)
#
#                 val_pred = model(input1, input2)  # Forward pass
#                 batch_loss = loss_fn(val_pred, targets)
#
#                 val_loss += batch_loss.item()
#                 val_acc += (val_pred.argmax(dim=1) == targets).sum().item()
#
#             _, predicted = torch.max(val_pred, 1)  # torch.max(tensor, dim)；即提取tensor 第一个维度的max，返回 maximum和对应的index(类)
#             # _: 为val_pred的最大值——1.4385；predicted:就是maximum对应的类
#             y_true_val.extend(targets.cpu().numpy())
#             y_pred_val.extend(predicted.cpu().numpy())
#
#             # Calculate average validation metrics
#             val_loss /= len(val_loader1)
#             val_acc /= len(val_loader1)
#
#             ##############计算混淆矩阵和分类
#             # 计算混淆矩阵及分类报告
#             val_cm = confusion_matrix(y_true_val, y_pred_val)
#             val_metrics = compute_metrics(y_true_val, y_pred_val)
#
#             ##############计算混淆矩阵和分类
#
#         ################
#         classes = ['extra_acid', 'acid', 'fine']
#         plot_confusion_matrix(train_cm, "Training Confusion Matrix", classes)
#         plot_confusion_matrix(val_cm, "Validation Confusion Matrix", classes)
#
#         # Log metrics and print
#         plt_train_acc.append(train_acc)
#         plt_train_loss.append(train_loss)
#         plt_val_acc.append(val_acc)
#         plt_val_loss.append(val_loss)
#
#         # Save best model based on validation accuracy
#         if val_acc > max_acc:
#             torch.save(model.state_dict(), save_ + 'best.pth')
#             max_acc = val_acc
#
#         # Early stopping logic
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             epochs_without_improvement = 0
#             torch.save(model.state_dict(), "best_model.pth")  # Save the best model
#         # else:
#         #     epochs_without_improvement += 1
#         #     if epochs_without_improvement >= patience:
#         #         print("Early stopping...")
#         #         break  # Stop training early if no improvement in validation loss
#
#         elapsed_time = time.time() - epoch_start_time
#         print(f'[{epoch + 1:03d}/{num_epoch}] {elapsed_time:.2f}s '
#               f'Train Acc: {train_acc:.6f}, Loss: {train_loss:.6f} | '
#               f'Val Acc: {val_acc:.6f}, Loss: {val_loss:.6f}')
#
#     # Save final model
#     torch.save(model.state_dict(), save_ + 'final.pth')
#
#     # Plot training and validation curves
#     plt.figure(figsize=(12, 6))
#
#     # Accuracy plot
#     plt.subplot(1, 2, 1)
#     plt.plot(plt_train_acc, label='Train Accuracy')
#     plt.plot(plt_val_acc, label='Validation Accuracy')
#     plt.legend()
#     plt.title('Accuracy')
#
#     # Loss plot
#     plt.subplot(1, 2, 2)
#     plt.plot(plt_train_loss, label='Train Loss')
#     plt.plot(plt_val_loss, label='Validation Loss')
#     plt.legend()
#     plt.title('Loss')
#
#     plt.show()
#
#     # 测试集评估
#     model.eval()
#     test_acc, test_loss = 0.0, 0.0
#     y_true_test, y_pred_test = [], []
#
#     with torch.no_grad():
#         for (data1, target1), (data2, target2) in zip(test_loader1, test_loader2):
#             input1, input2 = data1.to(device), data2.to(device)
#             targets = target1.to(device)
#
#             test_pred = model(input1, input2)  # 前向传播
#             batch_loss = loss_fn(test_pred, targets)
#
#             # 收集测试集的真实标签和预测标签
#             _, predicted = torch.max(test_pred.data, 1)
#             y_true_test.extend(targets.cpu().numpy())
#             y_pred_test.extend(predicted.cpu().numpy())
#
#             # 计算准确率
#             test_acc += (test_pred.argmax(dim=1) == targets).sum().item()
#             test_loss += batch_loss.item()
#
#         # 计算平均测试指标
#         test_loss /= len(test_loader1)
#         test_acc /= len(test_loader1)
#
#         # 计算测试集的混淆矩阵及分类报告
#         test_cm = confusion_matrix(y_true_test, y_pred_test)
#         classification_report(y_true_test, y_pred_test, output_dict=True)
#
#         # 打印测试集的分类报告
#         print("Test Classification Report:")
#         print(classification_report(y_true_test, y_pred_test))
#
#         classes = ['extra_acid', 'acid', 'fine']
#         # 可视化测试集的混淆矩阵
#         plot_confusion_matrix(test_cm, "Test Confusion Matrix", classes)
#
#         # 打印测试集的最终指标
#         print(f'Test Acc: {test_acc:.6f}, Loss: {test_loss:.6f}')


