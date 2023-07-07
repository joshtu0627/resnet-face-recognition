# ResNet Face Recognizer

這是一個使用pytorch的resnet人臉辨識器，此辨識器能夠方便的對參數進行調整，有自動生成report以及自動運行所設定的訓練順序，能夠省下使用者不少時間

## 數據集準備
請依照下列位置放置資料集
```
├── src
│   ├── this file
│   ├── setting.py
│   ├── identity_CelebA.txt
│   ├── report.txt
│   ├── img_align_celeba
│   ├── haarcascade_frontalface_default.xml
│   │   *front face detector, you can download it from
│   │     https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
│   ├── haarcascade_profileface.xml
│   │   *side face detector, you can download it from
│   │     https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_profileface.xml
│   ├── data
│   │   ├── pixelate
|   │   │   ├── 2x2 example filename:000001_2x2.jpg
|   │   │   ├── 4x4
|   │   │   ├── 8x8
|   │   │   ├── 16x16
│   │   ├── blur
|   │   │   ├── img_gaussian_blur_k15 example filename
|   │   │   ├── img_gaussian_blur_k45
|   │   │   ├── img_gaussian_blur_k99
```

## 使用方式
在setting.py調整想要的參數，並運行resnet_face_recognition_v3.1.py，運行結果將會出現在report.csv(較精簡的結果)，以及report.txt(較詳細的結果)