from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, Callback, EarlyStopping, ReduceLROnPlateau
from model_architecture import create_resnet_model, get_preprocess_function
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
import shutil
import random

# 自定義回調函數：保存最佳權重並自動刪除舊檔案
class BestModelCheckpoint(Callback):
    def __init__(self, filepath_template, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1):
        super(BestModelCheckpoint, self).__init__()
        self.filepath_template = filepath_template  # 檔案路徑範本
        self.monitor = monitor                      # 監控的指標
        self.mode = mode                           # 模式（max或min）
        self.save_best_only = save_best_only       # 只保存最佳模型
        self.verbose = verbose                     # 是否顯示訊息
        
        # 初始化最佳值和當前檔案路徑
        if mode == 'max':
            self.best = -float('inf')
        else:
            self.best = float('inf')
        self.current_filepath = None
    
    def on_epoch_end(self, epoch, logs=None):
        """每個 epoch 結束時的回調"""
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        # 檢查是否需要保存新的最佳模型
        if self.mode == 'max':
            improved = current > self.best
        else:
            improved = current < self.best
        
        if improved:
            # 更新最佳值
            self.best = current
            
            # 產生新的檔案路徑
            new_filepath = self.filepath_template.format(val_accuracy=current)
            
            # 刪除舊的權重檔案
            if self.current_filepath and os.path.exists(self.current_filepath):
                try:
                    os.remove(self.current_filepath)
                    if self.verbose:
                        print(f"\n刪除舊權重檔案: {self.current_filepath}")
                except Exception as e:
                    print(f"\n警告：無法刪除舊權重檔案 {self.current_filepath}: {e}")
            
            # 保存新的權重檔案
            self.model.save_weights(new_filepath)
            self.current_filepath = new_filepath
            
            if self.verbose:
                print(f"\n保存新的最佳權重: {new_filepath} (val_accuracy: {current:.4f})")

def prepare_data_split(data_dir, train_ratio=0.8, random_state=42):
    """
    準備訓練和驗證資料分割
    
    Args:
        data_dir: 包含所有類別資料夾的目錄路徑
        train_ratio: 訓練資料比例，預設為0.8（80%）
        random_state: 隨機種子，確保結果可重現
    
    Returns:
        train_generator: 訓練資料生成器
        validation_generator: 驗證資料生成器
    """
    # 建立臨時訓練和驗證目錄
    temp_train_dir = 'temp_train'
    temp_val_dir = 'temp_val'
    
    # 如果臨時目錄已存在，先刪除
    if os.path.exists(temp_train_dir):
        shutil.rmtree(temp_train_dir)
    if os.path.exists(temp_val_dir):
        shutil.rmtree(temp_val_dir)
    
    # 建立臨時目錄結構
    os.makedirs(temp_train_dir, exist_ok=True)
    os.makedirs(temp_val_dir, exist_ok=True)
    
    # 遍歷每個類別資料夾
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        
        # 跳過非目錄的檔案
        if not os.path.isdir(class_path):
            continue
            
        # 建立對應的訓練和驗證目錄
        train_class_dir = os.path.join(temp_train_dir, class_name)
        val_class_dir = os.path.join(temp_val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # 獲取該類別的所有圖片檔案
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        # 分割訓練和驗證資料
        train_files, val_files = train_test_split(
            image_files, 
            train_size=train_ratio, 
            random_state=random_state,
            shuffle=True
        )
        
        # 複製檔案到對應目錄
        for file_name in train_files:
            src_path = os.path.join(class_path, file_name)
            dst_path = os.path.join(train_class_dir, file_name)
            shutil.copy2(src_path, dst_path)
            
        for file_name in val_files:
            src_path = os.path.join(class_path, file_name)
            dst_path = os.path.join(val_class_dir, file_name)
            shutil.copy2(src_path, dst_path)
        
        print(f"類別 {class_name}: 訓練 {len(train_files)} 張，驗證 {len(val_files)} 張")
    return temp_train_dir, temp_val_dir

# 產生隨機種子並準備資料分割
random_seed = random.randint(1, 10000)
print(f"使用隨機種子: {random_seed}")
print("正在準備訓練和驗證資料分割...")
train_dir, val_dir = prepare_data_split('data', train_ratio=0.8, random_state=random_seed)

# 取得預處理函數
preprocess_input = get_preprocess_function()

# 訓練數據生成器（包含數據增強）
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)

# 驗證數據生成器（不進行數據增強）
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# 訓練數據生成器
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(360, 640),
    batch_size=32,
    class_mode='categorical'
)
print(train_generator.class_indices)

# 驗證數據生成器（使用獨立的驗證資料夾）
validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(360, 640),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # 關閉隨機打亂，確保檔案名稱與數據順序一致
)

print(validation_generator.class_indices)

# 使用共用模型架構建立模型
model = create_resnet_model(input_shape=(360, 640, 3), num_classes=3)

# 產生帶有日期和隨機種子的檔案名稱
current_date = datetime.now().strftime("%Y%m%d_%H%M")

# 使用自定義的最佳權重保存回調（會自動刪除舊檔案）
checkpoint = BestModelCheckpoint(
    f'best_resnet_model_360x640_{current_date}_rs{random_seed}_val_acc_{{val_accuracy:.4f}}.h5',
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

# 添加EarlyStopping和ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# 訓練模型並記錄歷史
history = model.fit(
    train_generator,
    epochs=1000,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping, lr_reducer],
    workers=1,                    # 設定為單進程，避免 Windows 多進程問題
    use_multiprocessing=False 
)

# 載入最佳權重進行最終評估
if checkpoint.current_filepath and os.path.exists(checkpoint.current_filepath):
    print(f"\n載入最佳權重進行最終評估: {checkpoint.current_filepath}")
    model.load_weights(checkpoint.current_filepath)
else:
    print("\n警告：找不到最佳權重檔案，使用最後 epoch 的權重")

# 繪製損失和準確率折線圖
plt.figure(figsize=(12, 4))

# 損失圖
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 準確率圖
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 保存圖表
plt.savefig('training_plots.png')
plt.show()

# 獲取驗證集數據和標籤（修正版本）
validation_generator.reset()  # 重置生成器到開始位置
val_images = []
val_labels = []
val_filenames = []  # 記錄對應的檔案名稱

# 逐批次獲取驗證數據，確保順序一致
batch_index = 0
for i in range(len(validation_generator)):
    # 使用索引獲取批次，而非 next() 方法
    batch_images, batch_labels = validation_generator[i]
    val_images.append(batch_images)
    val_labels.append(batch_labels)
    
    # 計算當前批次對應的檔案名稱索引範圍
    start_idx = batch_index * validation_generator.batch_size
    end_idx = min(start_idx + validation_generator.batch_size, len(validation_generator.filenames))
    
    # 將對應的檔案名稱加入列表
    batch_filenames = validation_generator.filenames[start_idx:end_idx]
    val_filenames.extend(batch_filenames)
    
    batch_index += 1

# 將所有批次合併成完整的驗證集
val_images = np.concatenate(val_images, axis=0)
val_labels = np.concatenate(val_labels, axis=0)

# 使用模型預測驗證集
predictions = model.predict(val_images)
predicted_labels = np.argmax(predictions, axis=1)  # 獲取預測類別
true_labels = np.argmax(val_labels, axis=1)        # 獲取真實類別

# 建立類別名稱對應字典
class_names = {v: k for k, v in train_generator.class_indices.items()}
wrong_predictions = []

# 找出所有分錯的圖片
for i in range(len(predicted_labels)):
    if predicted_labels[i] != true_labels[i]:
        wrong_predictions.append({
            'index': i,                                    # 在驗證集中的索引
            'filename': val_filenames[i],                  # 對應的檔案名稱
            'true_label': class_names[true_labels[i]],     # 真實標籤
            'predicted_label': class_names[predicted_labels[i]],  # 預測標籤
            'confidence': predictions[i][predicted_labels[i]]     # 預測信心度
        })

# 打印分錯圖片的詳細資訊
print(f"\n總共找到 {len(wrong_predictions)} 張分錯的圖片：")
print("=" * 80)

for i, wp in enumerate(wrong_predictions, 1):
    print(f"{i:3d}. {os.path.basename(wp['filename'])}")
    print(f"     True Label: {wp['true_label']}")
    print(f"     Predicted Label: {wp['predicted_label']}")
    print(f"     Confidence: {wp['confidence']:.4f}")
    print("-" * 60)

# 顯示分錯的圖片
for i, wp in enumerate(wrong_predictions):
    try:
        # 構建完整的圖片路徑
        img_path = os.path.join(val_dir, wp['filename'])
        
        # 檢查檔案是否存在
        if not os.path.exists(img_path):
            print(f"警告：檔案 {img_path} 不存在")
            continue
            
        # 載入並顯示圖片
        img = Image.open(img_path).resize((640, 360))
        plt.figure(figsize=(10, 5.6))
        plt.imshow(img)
        plt.title(f"Image {i+1}: {os.path.basename(wp['filename'])}\n"
                 f"True: {wp['true_label']} | Predicted: {wp['predicted_label']}\n"
                 f"Confidence: {wp['confidence']:.4f}", fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"載入圖片 {wp['filename']} 時發生錯誤: {str(e)}")

# 清理臨時目錄
print("清理臨時目錄...")
try:
    shutil.rmtree(train_dir)
    shutil.rmtree(val_dir)
    print("臨時目錄清理完成")
except Exception as e:
    print(f"清理臨時目錄時發生錯誤: {e}")