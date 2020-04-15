from __future__ import print_function, division
import os
import torch
import pandas as pd             
# from skimage import io, transform    #用于图像的IO和变换
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

# 忽略警告
import warnings
warnings.filterwarnings("ignore")


class RanseDataset(Dataset):
    '''
    染色数据集
    ./data
        train
            1.jpg
            2.jpg
        test
            1.jpg
            2.jpg

    '''
    def __init__(self,data_dir,transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.imgs = [os.path.join(self.data_dir,img) for img in os.listdir(self.data_dir)]


    def __len__(self):
        return len(self.imgs)


    def __getitem__(self,index):
        img_path = self.imgs[index]
        data = Image.open(img_path)

        # 替换成染色的
        if self.transform:
            data = self.transform(data)
        # label
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

        return np.array(data), label


# transforms # 3X224x224
input_size = 224
ch_norm_mean = (0.485, 0.456, 0.406)
ch_norm_std = (0.229, 0.224, 0.225)

# 数据处理及增强
transform_train = transforms.Compose([
    transforms.Resize(int(input_size/0.90)),
    transforms.CenterCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),# 0-1
    transforms.Normalize(ch_norm_mean, ch_norm_std),
])

transform_test = transforms.Compose([
    transforms.Resize(int(input_size/0.90)),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(ch_norm_mean, ch_norm_std),
])



#show image
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    plt.show()

if __name__ == "__main__":
    train_data=ChangDataset(data_dir="./data/train",transform=transform_train)
    test_data = ChangDataset(data_dir="./data/test",transform=transform_test)


    train_loader = DataLoader(dataset=train_data, batch_size=6, shuffle=True, num_workers=1)
    test_loader = DataLoader(dataset=test_data, batch_size=6, shuffle=False, num_workers=1)


    for test_img, test_label in test_loader:
        print(test_label)
        show_images([test_img[i][0] for i in range(6)], 2,3, scale=0.8)
        break

