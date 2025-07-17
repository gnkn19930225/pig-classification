# 豬隻分類系統 (Pig Classification System)

這是一個基於深度學習的豬隻分類系統，使用ResNet50模型來識別監控影像中的豬隻狀態。

## 專案概述

本系統能夠自動分類監控影像為以下三種類別：
- **camera_failure**: 攝影機故障
- **with_pigs**: 有豬隻在場
- **without_pigs**: 無豬隻在場

## 功能特色

- 🐷 自動識別豬隻存在與否
- 📹 檢測攝影機故障狀態
- 🎯 高準確率分類（驗證準確率可達97以上）
- 🔄 支援模型比較和評估
- 📊 詳細的訓練過程視覺化
- 🚀 支援ONNX格式轉換，便於部署

## 檔案結構

```
classification/
├── data/                          # 資料集目錄
│   ├── train/                     # 訓練資料
│   │   ├── camera_failure/        # 攝影機故障樣本
│   │   ├── with_pigs/            # 有豬隻樣本
│   │   └── without_pigs/         # 無豬隻樣本
│   ├── val/                      # 驗證資料
│   └── val_250717/               # 額外驗證資料
├── train_resnet.py               # 主要訓練腳本
├── compare_models.py             # 模型比較工具
├── convert_to_onnx.py           # ONNX格式轉換
├── get_data.py                  # 資料處理工具
├── best_resnet_model_*.h5       # 訓練好的模型權重
├── comparison_results.csv       # 模型比較結果
└── training_plots.png           # 訓練過程圖表
```

## 環境需求

### Python套件
```bash
pip install tensorflow==2.1
pip install numpy
pip install matplotlib
pip install Pillow
pip install onnx
pip install onnxruntime
```

### 系統需求
- Python 3.8.10
- 至少8GB RAM
- 建議使用GPU進行訓練（可選）

## 使用方法

###1. 準備資料
確保您的資料集按照以下結構組織：
```
data/
├── train/
│   ├── camera_failure/
│   ├── with_pigs/
│   └── without_pigs/
└── val/
    ├── camera_failure/
    ├── with_pigs/
    └── without_pigs/
```

### 2. 訓練模型
```bash
python train_resnet.py
```

訓練過程會：
- 自動保存最佳權重檔案
- 顯示訓練進度
- 生成訓練過程圖表
- 分析錯誤分類的樣本

### 3. 比較模型
```bash
python compare_models.py
```

### 4. 轉換為ONNX格式
```bash
python convert_to_onnx.py
```

## 模型架構

- **基礎模型**: ResNet50預訓練權重）
- **輸入尺寸**:360640x3
- **分類層**: 
  - GlobalAveragePooling2  - Dense(1024 ReLU)
  - Dropout(0.5)
  - Dense(3, Softmax)

## 訓練參數

- **批次大小**: 16(訓練), 32 (驗證)
- **學習率**: 000110個epoch衰減50*訓練週期**: 60epochs
- **數據增強**: 旋轉、平移、縮放、水平翻轉

## 效能表現

根據最新的訓練結果：
- **驗證準確率**: 880.27
- **模型檔案**: `best_resnet_model_360x640_202507170935_val_acc_00.88275

## 注意事項

1 **記憶體使用**: 訓練過程需要大量記憶體，建議使用GPU2. **資料品質**: 確保訓練資料品質良好，標籤正確
3 **模型保存**: 系統會自動保存最佳權重，並刪除舊檔案
4. **錯誤分析**: 訓練結束後會顯示所有分類錯誤的樣本

## 貢獻指南

歡迎提交Issue和Pull Request來改善這個專案！

## 授權

本專案採用MIT授權條款。

## 聯絡資訊

如有任何問題或建議，請透過GitHub Issues與我們聯絡。 