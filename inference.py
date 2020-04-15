from __future__ import print_function, division
import os
import torch
from torch import nn,optim
import torch.nn.functional as F
import pandas as pd                 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


import torch
from mobilenetv3 import *  # model
from data_pro import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#----------------model define-----------------
model = mobilenetv3(n_class=6, input_size=224, mode='large')
state_dict = torch.load("./checkpoint/pnasnet_100_0.8644578313253012.pth")
model.load_state_dict(state_dict)
model.to_device(device)
model.eval()
print(model)



def get_label(file):
    if "白光" in img_path:
        label = 0
    elif "电子染色" in img_path:
        label = 1
    elif "冰醋酸" in img_path:
        label = 2
    elif "碘染色" in img_path:
        label = 3
    elif "靛胭脂" in img_path:
        label = 4
    elif "美兰" in img_path:
        label = 5

    return label

files = os.listdir("./data/test")

real_labels = []
pred_labels = []
pred_probs = []

for file in files:
    real_labels.append(get_label(file))

    path_img = "./data/test/"+file

    img_1 = Image.open(path_img)
    img_2 = transform_test(img_1)

    input_tensor = img_2.unsqueeze(0) # 3x224X224 -> 1x3x224X224
    input_tensor = input_tensor.to(device)

    output_logits = model(input_tensor) # 1x6

    output_prob = F.softmax(output_logits,dim =1).detach().cpu() #对每一行进行softmax
    output_prob =output_prob.numpy() #对每一行进行softmax

    pred_label = np.argmax(output_logits.detach().cpu()).item()
    pred_labels.append(pred_label)
    pred_probs.append(np.max(output_prob))
    print("{} | {} | {}".format(file,pred_label,np.max(output_prob)))

target_names = ["未染色","电子染色","冰醋酸染色","卢戈氏碘液染色","靛胭脂染色","美兰染色"]

print("-----classification_report-----")
print(classification_report(real_labels,pred_labels,target_names=target_names))

print("----confusion_matrix-----")
cm = confusion_matrix(y_true=real_labels,y_pred=pred_labels)
print(cm)

print("------acc---------")
totalPic = np.sum(cm)
for i in range(6):
    print("%10s = %.4f"%(target_names[i]+" acc",(totalPic-np.sum(cm[:,i])-np.sum(cm[i,:])+2*cm[i,i]) / totalPic))

print("----------Sensitivity / Specificity-----------")
print("%10s%15s%15s"%("","Sensitivity","Specificity"))
for i in range(6):
    rsum = np.sum(cm[i,:])
    print("%10s%12.2f\t%12.2f"%(target_names[i],cm[i,i]/rsum,1-(np.sum(cm[:,i])-cm[i,i])/(totalPic-rsum)))

print("------阴/阳性预测值----------")
print("%10s%10s%10s"%("","阳性预测值","阴性预测值"))
for i in range(len(target_names)):
    pN = totalPic - np.sum(cm[:,i])
    print("%10s%12.2f\t%12.2f"%(target_names[i],cm[i,i]/np.sum(cm[:,i]),(pN-np.sum(cm[i,:])+cm[i,i])/pN))

