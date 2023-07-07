import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models import resnet18, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

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


def write_report(detailed_report, brief_report, configs, accuracy_score, report, clear_file):
    if clear_file:
        choose = input(
            'are you sure you want to clear the files? y for yes, n for no: ')
        while choose not in ['y', 'n']:
            choose = input(
                'are you sure you want to clear the files? y for yes, n for no: ')
        if choose == 'n':
            clear_file = False
    if clear_file:
        with open(detailed_report, 'w'):
            pass
        with open(brief_report, 'w'):
            pass

    id = 0
    file_exists = os.path.isfile(brief_report)
    if not file_exists or os.stat(brief_report).st_size == 0:
        id = 0
    else:
        with open(brief_report, 'r') as f:
            reader = csv.reader(f)
            id = len(list(reader))-1

    with open(detailed_report, 'a') as f:
        f.write(f'id : {id}\n')
        for config in configs:
            json.dump(config, f, indent=4)
            f.write('\n')

        f.write('\naccuracy_score: '+str(accuracy_score)+'\n')
        f.write(report)
        f.write('\n\n\n\n\n')

    with open(brief_report, 'a') as f:
        setting_selected_key = ['only_prediction','people_num', 'train_num', 'valid_num',
                                'test_num', 'train_rounds', 'resize_padding', 'shuffle',
                                'train_type', 'train_blur_degree', 'test_blur_degree',
                                'train_pixel_degree', 'test_pixel_degree','train_epsilon_degree',
                                'test_epsilon_degree','train_dp_blur_degree','train_dp_pixel_degree',
                                'test_dp_pixel_degree','save_model']
        model_config_selected_key = ['n_epochs', 'batch_size', 'learning_rate',
                                     'early_stop', 'freeze_weights']

        setting_dict = {k: SETTING[k] for k in setting_selected_key}
        model_config_dict = {k: MODEL_CONFIG[k]
                             for k in model_config_selected_key}

        merged_dict = {'id': id, 'accuracy': accuracy_score,
                       **setting_dict, **model_config_dict}
        with open(brief_report, 'a', newline='') as f:
            w = csv.DictWriter(f, merged_dict.keys())
            if not file_exists or os.stat(brief_report).st_size == 0:
                w.writeheader()
            w.writerow(merged_dict)

# training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()  # 記錄開始時間

    patience = 0  # 紀錄未優化的累積epoch數

    # 記錄最佳模型
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        model.train()

        # 訓練模型
        running_loss = 0.0  # 紀錄此epoch的loss平均
        running_corrects = 0  # 紀錄此epoch總共預測對幾個train data
        for inputs, labels in train_loader:
            # 將資料放置於 GPU 或 CPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 重設參數梯度
            optimizer.zero_grad()

            # 計算loss
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # 計算loss並反向傳播
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        scheduler.step()  # 更新 scheduler

        # 計算本輪訓練數據
        epoch_loss = running_loss / train_sizes
        epoch_acc = running_corrects.double() / train_sizes

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'train', epoch_loss, epoch_acc))

        # 驗證模型
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # 計算本輪驗證數據
        epoch_loss = running_loss / valid_sizes
        epoch_acc = running_corrects.double() / valid_sizes
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'valid', epoch_loss, epoch_acc))

        # 記錄最佳模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            patience += 1

        # 若抵達設定的early_stop, 停止訓練
        if patience > MODEL_CONFIG['early_stop']:
            print('early stopped')
            break

    # 計算耗費時間
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # 輸出最佳準確度
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


adjust_time = 0
if SETTING['self_adjust'] == True:
    print('self adjusting is on ')
    print('will adjust', SETTING['param_name'], 'into', SETTING['param_values'])
    print()
    adjust_time = len(SETTING['param_values'][0])
else:
    adjust_time = 1

for adjust_count in range(adjust_time):
    if SETTING['self_adjust'] == True:
        if SETTING['param_category'] == 'SETTING':
            for i, parem in enumerate(SETTING['param_name']):
                SETTING[SETTING['param_name'][i]] = SETTING['param_values'][i][adjust_count]
                print('changed',SETTING['param_name'], 'into', SETTING['param_values'][i][adjust_count])
        else:
            for i, parem in enumerate(SETTING['param_name']):
                MODEL_CONFIG[SETTING['param_name'][i]] = SETTING['param_values'][i][adjust_count]
                print('changed',SETTING['param_name'], 'into', SETTING['param_values'][i][adjust_count])
        

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
    train_num = SETTING['train_num']  # 指定訓練照片數量
    valid_num = SETTING['valid_num']  # 指定測試照片數量
    test_num = SETTING['test_num']  # 指定測試照片數量

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
            if len(v) > train_num + test_num + valid_num:  # 如果此人有的照片數量多於訓練數量+測試數量 則選擇此人
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

    x_train, x_test, x_valid = [], [], []
    y_train, y_test, y_valid = [], [], []
    train_degree = []
    test_degree = []
    if SETTING['train_type'] == 'original':
        train_degree = [0]
        test_degree = [0]
    elif SETTING['train_type'] == 'blur':
        train_degree = SETTING['train_blur_degree']
        test_degree = SETTING['test_blur_degree']
    elif SETTING['train_type'] == 'pixel':
        train_degree = SETTING['train_pixel_degree']
        test_degree = SETTING['test_pixel_degree']
    elif SETTING['train_type'] == 'dp_blur':
        for i in SETTING['train_epsilon_degree']:
            for j in SETTING['train_dp_blur_degree']:
                train_degree.append([i,j])
        for i in SETTING['test_epsilon_degree']:
            for j in SETTING['test_dp_blur_degree']:
                test_degree.append([i,j])
    elif SETTING['train_type'] == 'dp_pixel':
        for i in SETTING['train_epsilon_degree']:
            for j in SETTING['train_dp_pixel_degree']:
                train_degree.append([i,j])
        for i in SETTING['test_epsilon_degree']:
            for j in SETTING['test_dp_pixel_degree']:
                test_degree.append([i,j])
    else:
        train_degree = [0]
        test_degree = [0]

    for p in choose:    # 每個人都跑一遍
        print(p)
        if SETTING['shuffle'] == True:
            shuffle(id2imgs[p])     # 打亂每個人的照片順序

        for i in range(train_num+valid_num):   # 這個人的每張圖片
            for d in train_degree:
                img = None
                if SETTING['train_type'] == 'original':
                    img = cv2.imread(
                        f'{SETTING["original_data_folder"]}/{id2imgs[p][i]}')
                elif SETTING['train_type'] == 'blur':
                    img = cv2.imread(
                        f'{SETTING["blur_data_folder"]}/img_gaussian_blur_k{d}/{id2imgs[p][i]}')
                elif SETTING['train_type'] == 'pixel':
                    img = cv2.imread(
                        f'{SETTING["pixel_data_folder"]}/{d}x{d}/{id2imgs[p][i][:-4]}_{d}x{d}.jpg')
                elif SETTING['train_type'] == 'deblur':
                    img = cv2.imread(
                        f'{SETTING["deblur_data_folder"]}/{id2imgs[p][i]}')
                    img = cv2.resize(img, (int(178), int(218)),
                                     interpolation=cv2.INTER_LINEAR) 
                elif SETTING['train_type'] == 'dp_blur':
                    epsilon, deg = d
                    #0.1_k15
                    #000001_epsilon0.1.jpg
                    img = cv2.imread(
                        f'{SETTING["dp_blur_data_folder"]}/{epsilon}_k{deg}/{id2imgs[p][i][:-4]}_epsilon{epsilon}.jpg')
                elif SETTING['train_type'] == 'dp_pixel':
                    epsilon, deg = d
                    #2x2_epsilon0.1
                    #000001_epsilon0.1.jpg
                    img = cv2.imread(
                        f'{SETTING["dp_pixel_data_folder"]}/{deg}x{deg}_epsilon{epsilon}/{id2imgs[p][i][:-4]}_epsilon{epsilon}.jpg')
                else:
                    print('illegal type')
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

                if i < train_num:             # 建立訓練資料集
                    x_train.append(img1)
                    x_train.append(img2)
                    # 使用mapping到0~9的label, 像是id = 2289 -> id = 1
                    y_train.append(mapping[p])
                    y_train.append(mapping[p])
                elif i < train_num+valid_num:  # 建立驗證資料集
                    x_valid.append(img1)
                    x_valid.append(img2)
                    y_valid.append(mapping[p])
                    y_valid.append(mapping[p])

        for i in range(train_num+valid_num, train_num+valid_num+test_num):   # 這個人的每張圖片
            for d in test_degree:
                img = None
                if SETTING['train_type'] == 'original':
                    img = cv2.imread(
                        f'{SETTING["original_data_folder"]}/{id2imgs[p][i]}')
                elif SETTING['train_type'] == 'blur':
                    img = cv2.imread(
                        f'{SETTING["blur_data_folder"]}/img_gaussian_blur_k{d}/{id2imgs[p][i]}')
                elif SETTING['train_type'] == 'pixel':
                    img = cv2.imread(
                        f'{SETTING["pixel_data_folder"]}/{d}x{d}/{id2imgs[p][i][:-4]}_{d}x{d}.jpg')
                elif SETTING['train_type'] == 'sd':
                    img = cv2.imread(
                        f'stable_diffusion_outputs/k15/{id2imgs[p][i]}')
                    img = cv2.resize(img, (int(178), int(218)),
                                     interpolation=cv2.INTER_LINEAR) 
                elif SETTING['train_type'] == 'dp_blur':
                    epsilon, deg = d
                    #0.1_k15
                    #000001_epsilon0.1.jpg
                    img = cv2.imread(
                        f'{SETTING["dp_blur_data_folder"]}/{epsilon}_k{deg}/{id2imgs[p][i][:-4]}_epsilon{epsilon}.jpg')
                elif SETTING['train_type'] == 'dp_pixel':
                    epsilon, deg = d
                    #2x2_epsilon0.1
                    #000001_epsilon0.1.jpg
                    img = cv2.imread(
                        f'{SETTING["dp_pixel_data_folder"]}/{deg}x{deg}_epsilon{epsilon}/{id2imgs[p][i][:-4]}_epsilon{epsilon}.jpg')
                else:
                    print('illegal type')
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
                img = Image.fromarray(img)  # 把圖片轉為PIL格式 這樣才能進行data_transforms

                x_test.append(img)
                y_test.append(mapping[p])

    # make dataset from FaceDataset
    train_dataset, test_dataset, valid_dataset = FaceDataset(x_train, y_train, data_transforms, 'train'), \
        FaceDataset(x_test, y_test, data_transforms, 'val'), FaceDataset(
            x_valid, y_valid, data_transforms, 'val')

    # build dataloader from datasets
    train_loader = DataLoader(
        train_dataset, batch_size=MODEL_CONFIG['batch_size'], shuffle=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=MODEL_CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=MODEL_CONFIG['batch_size'], shuffle=False)
    print('finished building dataset...\n')

    # print dataset length
    train_sizes = len(x_train)
    valid_sizes = len(x_valid)
    test_sizes = len(x_test)
    print('dataset length:')
    print(f"訓練集大小: {train_sizes}")
    print(f"驗證集大小: {valid_sizes}")
    print(f"測試集大小: {test_sizes}")
    print()

    train_rounds = SETTING['train_rounds']
    print(f'training on {SETTING["train_type"]} dataset')
    accuracy2result = {}
    best_acc = 0
    best_model = None
    for i in range(train_rounds):
        model_ft = None
        if SETTING['only_prediction']:
        # 載入模型
            print(f'predicting {i+1}/{train_rounds}')
            model_ft = torch.load(SETTING['model_for_prediction'])
            print('model',SETTING['model_for_prediction'],'loaded')
        else:
            print(f'training {i+1}/{train_rounds}')
            print('using ',MODEL_CONFIG['model_type'],' to train')
            if MODEL_CONFIG['model_type'] == 'resnet-18':
                model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            elif MODEL_CONFIG['model_type'] == 'resnet-34':
                model_ft = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            elif MODEL_CONFIG['model_type'] == 'resnet-50':
                model_ft = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            elif MODEL_CONFIG['model_type'] == 'resnext-50':
                model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
            else:
                print('illegal model!')
                exit()
            # 鎖定 ResNet 預訓練模型參數
            if MODEL_CONFIG['freeze_weights'] == 'True':
                for param in model_ft.parameters():
                    param.requires_grad = False

            # 取得 ResNet18 最後一層的輸入特徵數量
            num_ftrs = model_ft.fc.in_features

            # 調整resnet18的輸入特徵數量
            model_ft.fc = nn.Linear(num_ftrs, len(choose))

            # 將模型放置於 GPU 或 CPU
            model_ft = model_ft.to(device)

            # 使用 cross entropy loss
            criterion = nn.CrossEntropyLoss()

            # 學習優化器
            optimizer_ft = None
            if MODEL_CONFIG['freeze_weights'] == 'True':
                optimizer_ft = optim.SGD(model_ft.fc.parameters(),
                                        lr=MODEL_CONFIG['learning_rate'], momentum=0.9)
            else:
                optimizer_ft = optim.SGD(model_ft.parameters(),
                                        lr=MODEL_CONFIG['learning_rate'], momentum=0.9)

            # 每 7 個 epochs 將 learning rate 降為原本的 0.1 倍
            exp_lr_scheduler = lr_scheduler.StepLR(
                optimizer_ft, step_size=7, gamma=0.1)

            # 訓練模型
            model_ft = train_model(model_ft, criterion, optimizer_ft,
                                exp_lr_scheduler, num_epochs=MODEL_CONFIG['n_epochs'])        

        # 測試模型準度
        # 用這種方法是因為窩不知道怎樣比較好寫
        model_ft.eval()
        now_preds = []
        now_labels = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            now_labels += [int(i) for i in labels]
            labels = labels.to(device)

            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            now_preds += [int(i) for i in preds]
        print(now_preds)
        print(now_labels)
        accuracy = accuracy_score(now_preds, now_labels)
        print(f'round {i+1} accuracy: {accuracy}\n')
        accuracy2result[accuracy] = [now_labels, now_preds]
        if SETTING['save_model'] == True and SETTING['only_prediction'] == False:
            if accuracy > best_acc:
                best_model = model_ft
                best_acc = accuracy
                print('updated best model\n')
        # 重新選人
        if SETTING['random_choosing_people']:
            count = 0
            choose = []
            keys = list(id2imgs.keys())
            shuffle(keys)
            id2imgs = {key: id2imgs[key] for key in keys}
            for k, v in id2imgs.items():
                if len(v) > train_num + test_num + valid_num:  # 如果此人有的照片數量多於訓練數量+測試數量 則選擇此人
                    choose.append(k)
                    count += 1
                    if count == people_num:
                        break
            # map id to 1~9 for training purposes
            mapping = {}
            for i in range(len(choose)):
                mapping[choose[i]] = i
            print(choose)

    sorted_result = sorted(accuracy2result.items(), key=lambda x: x[0])
    count = 1
    all_labels = []
    all_preds = []
    for r in sorted_result:
        if count >= 1 and count < SETTING['train_rounds']-1:
            all_labels += r[1][0]
            all_preds += r[1][1]
        count += 1
    accuracy = accuracy_score(all_labels,all_preds )
    report = classification_report( all_labels,all_preds,
                                   target_names=[str(mapping[s]) for s in choose])
    print("accuracy score:", accuracy)
    print("report:\n", report)

    if SETTING['only_prediction'] == False and SETTING['save_model'] == True:
        print('best model accuracy:', best_acc)
        torch.save(best_model, SETTING['model_name'])
        print('model save as:',SETTING['model_name'])

    if SETTING['write_report'] == True:
        write_report(SETTING['detailed_report_file'], SETTING['brief_report_file'], [
            SETTING, MODEL_CONFIG], accuracy, report, SETTING['clear_previous_result'])
