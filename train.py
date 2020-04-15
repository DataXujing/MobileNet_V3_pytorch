from __future__ import print_function, division
import os
import torch
from torch import nn,optim
import pandas as pd              
# from skimage import io, transform    
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch


from data_pro import *    # data
from mobilenetv3 import *  # model

from tqdm import tqdm
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 忽略警告
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#----------------model define-----------------
model = mobilenetv3(n_class=6, input_size=224, mode='large')
model._initialize_weights()    # 初始化权值


print("-"*100)
print(model)
print("-"*100)

# 损失函数优化器,fine tuning
lr = 0.001
T_max = 300 # T_max周期后重新设置学习率

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=4e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0.000001, last_epoch=-1)
    

#test acc
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n

# train
def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    # loss = torch.nn.CrossEntropyLoss()
    pbar = tqdm(range(num_epochs))
    test_acc = 0.0
    iter_ = 0
    for epoch in pbar:
        scheduler.step()  # 更新学习率
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = criterion(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

            pbar.set_description('Epoch %d, Loss %.4f, Train acc %.3f, Test acc %.3f, Time %.1f sec'
               % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        test_acc = evaluate_accuracy(test_iter, net)

        if epoch % 10 == 0:
            torch.save(net.state_dict(), './checkpoint/pnasnet_{}_{}.pth'.format(epoch,test_acc))
        # print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
        #       % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        



if __name__ == "__main__":

    batch_size = 256

    train_data=ChangDataset(data_dir="./data/train",transform=transform_train)
    test_data = ChangDataset(data_dir="./data/test",transform=transform_test)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=4)

    train(model, train_loader, test_loader, batch_size=batch_size, optimizer=optimizer, device=device, num_epochs=300)