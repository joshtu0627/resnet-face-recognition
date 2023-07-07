import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models import resnet18, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
import torch.nn.functional as F

import random

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from scipy.optimize import differential_evolution

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from random import shuffle
import time
import copy
import cv2
import json
import os
import csv

from setting import SETTING, MODEL_CONFIG
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# transform function of data
data_transforms = {
    # 訓練資料集採用資料增強與標準化轉換
    'train': transforms.Compose([
        # transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [
                             0.229, 0.224, 0.225])  # 標準化
    ]),
    # 驗證資料集僅採用資料標準化轉換
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [
                             0.229, 0.224, 0.225])  # 標準化
    ]),
}


class FaceDataset(Dataset):
    '''
    x: Features.
    y: Targets.
    transform : Transform methods.
    transform_type : Which transform method to use.
    '''

    def __init__(self, x, y=None, transform=None, transform_type=None):
        # self.x = [torch.FloatTensor(i) for i in x]
        self.x = x
        self.y = y
        self.transform = transform
        self.transform_type = transform_type

    def __getitem__(self, idx):
        feature = self.x[idx]
        label = self.y[idx]

        if self.transform is not None:
            if self.transform_type is not None:
                feature = self.transform[self.transform_type](feature)

        return feature, label

    def __len__(self):
        return len(self.x)


def pad_img(img, width, height):
    # 調整圖片大小，保持長寬比不變
    aspect_ratio = img.shape[1] / img.shape[0]
    if aspect_ratio > width / height:
        new_height = int(width / aspect_ratio)
        img_resized = cv2.resize(img, (width, new_height))
    else:
        new_width = int(height * aspect_ratio)
        img_resized = cv2.resize(img, (new_width, height))

    # 獲取調整圖片的大小
    h, w, _ = img_resized.shape

    # 在圖片周圍添加黑色像素以達到目標大小
    top = (height - h) // 2
    bottom = (height - h) // 2
    left = (width - w) // 2
    right = (width - w) // 2
    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return img_padded

def perturb_image(xs, img):
    xs = xs.astype(int)

    pixels = np.split(xs, len(xs)/5)
    for pixel in pixels:
        x_pos, y_pos, r, g, b = pixel

        img[0, x_pos, y_pos] = r/255*2-1
        img[1, x_pos, y_pos] = g/255*2-1
        img[2, x_pos, y_pos] = b/255*2-1
    return img

count = 0
data = []
decrease = []
def predict_classes(xs, img, target_class, net, minimize=True):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    img_perturbed = perturb_image(xs, img.clone())
    img_perturbed = img_perturbed.to(device)
    
    res = F.softmax(net(img_perturbed.unsqueeze(0)))
    
    prediction = res[0, target_class].data.cpu().item()
    return prediction

def attack(img, label, net, target=None, pixels=3, maxiter=75, popsize=400, verbose=False):
	# img: 1*3*W*H tensor
	# label: a number
    original_img = img.clone()
    original_img = original_img.to(device)
    res = F.softmax(net(original_img.unsqueeze(0)))
    pre_prediction = res[0, target].data.cpu().item()
    print("start accuracy:",pre_prediction)

    bounds = [(0,109), (0,89), (0,255), (0,255), (0,255)] * pixels
    popmul = max(1, int(popsize/len(bounds)))
    predict_fn = lambda xs: predict_classes(
        xs, img, target, net, target is None)

    attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, polish=False)
    attacked_image = perturb_image(attack_result.x, img.clone())
    attacked_image = attacked_image.to(device)
    
    res = F.softmax(net(attacked_image.unsqueeze(0)))
    
    prediction = res[0, target].data.cpu().item()
    print("result accuracy: ", prediction)
    return 

def attack_all(net, loader, pixels=10, targeted=False, maxiter=75, popsize=400, verbose=False):
    for inputs, targets in loader:
        for img, target in zip(inputs, targets):
            attack(img, target, net, target, 
                    pixels=pixels, maxiter=maxiter, popsize=popsize, verbose=verbose)
    print(sum(decrease)/len(decrease))
    return 

adjust_time = 0
if SETTING['self_adjust'] == True:
    print('self adjusting is on ')
    print('will adjust', SETTING['param_name'], 'into', SETTING['param_values'])
    print()
    adjust_time = len(SETTING['param_values'])
else:
    adjust_time = 1

# load data file
print('start processing files...')
f = open(SETTING['meta_data_file'], 'r')
id2imgs = {}  # 存放的資料為 {人物id, 對應的照片名稱list}

lines = f.read().split()
for i in range(0, len(lines), 2):
    if int(lines[i+1]) in id2imgs:
        id2imgs[int(lines[i+1])].append(lines[i])
    else:
        id2imgs[int(lines[i+1])] = [lines[i]]

# choose people
people_num = SETTING['people_num']  # 指定多少人
attack_num = 5  # 指定攻擊照片數量

count = 0
choose = []
if SETTING['manual_choosing_peoples']:
    choose = SETTING['choosed_peoples']
else:
    if SETTING['random_choosing_people']:
        keys = list(id2imgs.keys())
        shuffle(keys)
        id2imgs = {key: id2imgs[key] for key in keys}
    for k, v in id2imgs.items():
        if len(v) > attack_num :  # 如果此人有的照片數量多於訓練數量+測試數量 則選擇此人
            choose.append(k)
            count += 1
            if count == people_num:
                break
print(choose)
print('finished processing files...\n')

# map id to 1~9 for training purposes
mapping = {}
for i in range(len(choose)):
    mapping[choose[i]] = i

# build dataset
print('start building dataset...')
frontface_detector = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')  # 偵測正臉位置的模型
sideface_detector = cv2.CascadeClassifier(
    'haarcascade_profileface.xml')  # 偵測側臉位置的模型

x_attacked = []
y_label = []
for p in choose:    # 每個人都跑一遍
    degree = [1]
    for d in degree:
        for i in range(25,26):   # 這個人的每張圖片
            img = cv2.imread(
                            f'{SETTING["original_data_folder"]}/{id2imgs[p][i]}')
            
            print(id2imgs[p][i])
            
            height, width = img.shape[:2]

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 色彩轉換成黑白
            face = frontface_detector.detectMultiScale(gray)  # 選出臉的框
            if len(face) == 0:  # 如果沒有正臉 取側臉
                face = sideface_detector.detectMultiScale(gray)
                for (x, y, w, h) in face:  # 因為這裡的圖片都是一張臉 所以直接取第一個
                    img = img[y:y+h, x:x+w]
                    break
            else:
                for (x, y, w, h) in face:
                    img = img[y:y+h, x:x+w]
                    break
            if SETTING['resize_padding'] == True:
                img = pad_img(img, int(width/2), int(height/2))
            else:
                img = cv2.resize(img, (int(width/2), int(height/2)),
                                    interpolation=cv2.INTER_LINEAR)  # 調整大小 若圖片大小不一 將無法餵給模型
            flipped = cv2.flip(img, 1)  # 翻轉圖片以得更多資料
            img1 = Image.fromarray(img)  # 把圖片轉為PIL格式 這樣才能進行data_transforms
            img2 = Image.fromarray(flipped)

                        # 建立訓練資料集
            x_attacked.append(img1)
            # x_attacked.append(img2)
            # 使用mapping到0~9的label, 像是id = 2289 -> id = 1
            y_label.append(mapping[p])
            # y_label.append(mapping[p])

# make dataset from FaceDataset
attack_dataset = FaceDataset(x_attacked, y_label, data_transforms, 'train')

# build dataloader from datasets
loader = DataLoader(
    attack_dataset, batch_size=MODEL_CONFIG['batch_size'], shuffle=False)
print('finished building dataset...\n')

# print dataset length
train_sizes = len(x_attacked)
print('dataset length:')
print(f"訓練集大小: {train_sizes}")
print()

net = torch.load('models/resnet18_pixel_model.pt')
attack_all(net, loader, pixels=5, targeted=False, maxiter=100, popsize=400, verbose=False)
