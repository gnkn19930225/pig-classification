from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, Callback, EarlyStopping, ReduceLROnPlateau
from model_architecture import create_resnet_model, get_preprocess_function
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
import shutil
import random
import json

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

def prepare_kfold_data(data_dir, k_folds=5, random_state=42):
    """
    準備K折交叉驗證的資料分割
    
    Args:
        data_dir: 包含所有類別資料夾的目錄路徑
        k_folds: K折交叉驗證的折數，預設為5
        random_state: 隨機種子，確保結果可重現
    
    Returns:
        data_splits: 包含每個fold的檔案分割資訊的字典
    """
    data_splits = {}
    
    # 遍歷每個類別資料夾並收集所有圖片檔案
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        
        # 跳過非目錄的檔案
        if not os.path.isdir(class_path):
            continue
            
        # 獲取該類別的所有圖片檔案
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        # 建立KFold分割器
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        
        # 為每個fold產生索引分割
        class_splits = []
        for train_idx, val_idx in kfold.split(image_files):
            train_files = [image_files[i] for i in train_idx]
            val_files = [image_files[i] for i in val_idx]
            class_splits.append({
                'train': train_files,
                'val': val_files
            })
        
        data_splits[class_name] = class_splits
        print(f"類別 {class_name}: 總共 {len(image_files)} 張圖片，每個fold約 {len(image_files)//k_folds} 張用於驗證")
    
    return data_splits

def create_fold_directories(data_dir, data_splits, fold_idx):
    """
    為指定的fold建立訓練和驗證目錄
    
    Args:
        data_dir: 原始資料目錄
        data_splits: K折分割的資料資訊
        fold_idx: 當前fold的索引
    
    Returns:
        temp_train_dir: 訓練資料臨時目錄
        temp_val_dir: 驗證資料臨時目錄
    """
    # 建立當前fold的訓練和驗證目錄
    temp_train_dir = f'temp_train_fold_{fold_idx}'
    temp_val_dir = f'temp_val_fold_{fold_idx}'
    
    # 如果臨時目錄已存在，先刪除
    if os.path.exists(temp_train_dir):
        shutil.rmtree(temp_train_dir)
    if os.path.exists(temp_val_dir):
        shutil.rmtree(temp_val_dir)
    
    # 建立臨時目錄結構
    os.makedirs(temp_train_dir, exist_ok=True)
    os.makedirs(temp_val_dir, exist_ok=True)
    
    # 遍歷每個類別並建立對應的fold目錄
    for class_name, class_splits in data_splits.items():
        class_path = os.path.join(data_dir, class_name)
        
        # 建立對應的訓練和驗證目錄
        train_class_dir = os.path.join(temp_train_dir, class_name)
        val_class_dir = os.path.join(temp_val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # 取得當前fold的檔案分割
        current_split = class_splits[fold_idx]
        train_files = current_split['train']
        val_files = current_split['val']
        
        # 複製訓練檔案
        for file_name in train_files:
            src_path = os.path.join(class_path, file_name)
            dst_path = os.path.join(train_class_dir, file_name)
            shutil.copy2(src_path, dst_path)
            
        # 複製驗證檔案
        for file_name in val_files:
            src_path = os.path.join(class_path, file_name)
            dst_path = os.path.join(val_class_dir, file_name)
            shutil.copy2(src_path, dst_path)
        
        print(f"Fold {fold_idx+1} - 類別 {class_name}: 訓練 {len(train_files)} 張，驗證 {len(val_files)} 張")
    
    return temp_train_dir, temp_val_dir

def create_full_data_directory(data_dir):
    """
    建立包含所有資料的訓練目錄（用於最終模型訓練）
    
    Args:
        data_dir: 原始資料目錄
    
    Returns:
        temp_full_dir: 包含所有資料的臨時目錄
    """
    temp_full_dir = 'temp_full_data'
    
    # 如果臨時目錄已存在，先刪除
    if os.path.exists(temp_full_dir):
        shutil.rmtree(temp_full_dir)
    
    # 建立臨時目錄結構
    os.makedirs(temp_full_dir, exist_ok=True)
    
    # 遍歷每個類別資料夾
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        
        # 跳過非目錄的檔案
        if not os.path.isdir(class_path):
            continue
            
        # 建立對應的類別目錄
        full_class_dir = os.path.join(temp_full_dir, class_name)
        os.makedirs(full_class_dir, exist_ok=True)
        
        # 獲取該類別的所有圖片檔案
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        # 複製所有檔案
        for file_name in image_files:
            src_path = os.path.join(class_path, file_name)
            dst_path = os.path.join(full_class_dir, file_name)
            shutil.copy2(src_path, dst_path)
        
        print(f"最終訓練 - 類別 {class_name}: 總共 {len(image_files)} 張圖片")
    
    return temp_full_dir

def cleanup_fold_directories(temp_train_dir, temp_val_dir):
    """
    清理fold的臨時目錄
    
    Args:
        temp_train_dir: 訓練資料臨時目錄
        temp_val_dir: 驗證資料臨時目錄
    """
    try:
        if os.path.exists(temp_train_dir):
            shutil.rmtree(temp_train_dir)
        if os.path.exists(temp_val_dir):
            shutil.rmtree(temp_val_dir)
        print(f"已清理臨時目錄: {temp_train_dir}, {temp_val_dir}")
    except Exception as e:
        print(f"清理臨時目錄時發生錯誤 cleanup_fold_directories: {e}")

def cleanup_directory(directory):
    """
    清理指定目錄
    
    Args:
        directory: 要清理的目錄路徑
    """
    try:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        print(f"已清理目錄: {directory}")
    except Exception as e:
        print(f"清理目錄時發生錯誤 cleanup_directory: {e}")

# K折交叉驗證主程式
def main():
    # 設定K折交叉驗證參數
    K_FOLDS = 5
    
    # 產生隨機種子並準備K折資料分割
    random_seed = random.randint(1, 10000)
    print(f"使用隨機種子: {random_seed}")
    print(f"開始進行 {K_FOLDS} 折交叉驗證...")
    
    # 準備K折資料分割
    data_splits = prepare_kfold_data('data', k_folds=K_FOLDS, random_state=random_seed)
    
    # 取得預處理函數
    preprocess_input = get_preprocess_function()
    
    # 記錄每個fold的結果
    fold_results = []
    all_histories = []
    best_models = []
    
    # 產生帶有日期和隨機種子的基礎檔案名稱
    current_date = datetime.now().strftime("%Y%m%d_%H%M")
    
    print(f"\n{'='*80}")  
    print(f"第一階段：{K_FOLDS}折交叉驗證 - 評估模型性能")
    print(f"{'='*80}")
    
    # 開始K折交叉驗證
    for fold in range(K_FOLDS):
        print(f"\n{'='*60}")  
        print(f"開始第 {fold+1}/{K_FOLDS} 折訓練")
        print(f"{'='*60}")
        
        try:
            # 為當前fold建立訓練和驗證目錄
            train_dir, val_dir = create_fold_directories('data', data_splits, fold)
            
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
            
            # 建立訓練資料生成器
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(360, 640),
                batch_size=32,
                class_mode='categorical',
                shuffle=True
            )
            
            # 建立驗證資料生成器
            validation_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=(360, 640),
                batch_size=32,
                class_mode='categorical',
                shuffle=False
            )
            
            print(f"Fold {fold+1} - 類別索引對應: {train_generator.class_indices}")
            
            # 使用共用模型架構建立模型（每個fold都建立新的模型）
            model = create_resnet_model(input_shape=(360, 640, 3), num_classes=3)
            
            # 設定當前fold的權重檔案路徑
            checkpoint = BestModelCheckpoint(
                f'best_resnet_model_360x640_{current_date}_rs{random_seed}_fold{fold+1}_val_acc_{{val_accuracy:.4f}}.h5',
                monitor='val_accuracy',
                mode='max',
                verbose=1
            )
            
            # 設定回調函數
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            
            # 訓練模型
            history = model.fit(
                train_generator,
                epochs=1000,
                validation_data=validation_generator,
                callbacks=[checkpoint, early_stopping, lr_reducer],
                workers=1,
                use_multiprocessing=False
            )
            
            # 載入最佳權重進行評估
            if checkpoint.current_filepath and os.path.exists(checkpoint.current_filepath):
                print(f"\nFold {fold+1} - 載入最佳權重: {checkpoint.current_filepath}")
                model.load_weights(checkpoint.current_filepath)
                best_models.append(checkpoint.current_filepath)
            else:
                print(f"\nFold {fold+1} - 警告：使用最後 epoch 的權重")
                best_models.append(None)
            
            # 評估當前fold的性能
            val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
            
            # 找到最佳驗證準確率出現的epoch（用於計算最終訓練的平均epochs）
            best_val_acc = max(history.history['val_accuracy'])
            best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1  # +1因為epoch從1開始計算
            
            # 記錄fold結果
            fold_result = {
                'fold': fold + 1,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'best_val_accuracy': best_val_acc,
                'best_epoch': best_epoch,  # 最佳結果出現的epoch
                'best_model_path': checkpoint.current_filepath,
                'epochs_trained': len(history.history['loss'])  # 總共訓練的epochs
            }
            fold_results.append(fold_result)
            all_histories.append(history.history)
            
            print(f"\nFold {fold+1} 結果:")
            print(f"  驗證損失: {val_loss:.4f}")
            print(f"  驗證準確率: {val_accuracy:.4f}")
            print(f"  最佳驗證準確率: {fold_result['best_val_accuracy']:.4f}")
            print(f"  最佳結果epoch: {fold_result['best_epoch']}")
            print(f"  總訓練epochs數: {fold_result['epochs_trained']}")
            
        except Exception as e:
            print(f"Fold {fold+1} 訓練過程發生錯誤 main: {e}")
            fold_results.append({
                'fold': fold + 1,
                'val_loss': float('inf'),
                'val_accuracy': 0.0,
                'best_val_accuracy': 0.0,
                'best_model_path': None,
                'epochs_trained': 0,
                'error': str(e)
            })
            
        finally:
            # 清理當前fold的臨時目錄
            cleanup_fold_directories(train_dir, val_dir)
    
    # 計算K折交叉驗證的平均結果
    print(f"\n{'='*80}")
    print(f"{K_FOLDS}折交叉驗證完成！")
    print(f"{'='*80}")
    
    # 計算統計資訊
    valid_results = [r for r in fold_results if 'error' not in r]
    if valid_results:
        avg_val_loss = np.mean([r['val_loss'] for r in valid_results])
        avg_val_accuracy = np.mean([r['val_accuracy'] for r in valid_results])
        std_val_accuracy = np.std([r['val_accuracy'] for r in valid_results])
        avg_best_val_accuracy = np.mean([r['best_val_accuracy'] for r in valid_results])
        
        print(f"\n統計結果:")
        print(f"  平均驗證損失: {avg_val_loss:.4f}")
        print(f"  平均驗證準確率: {avg_val_accuracy:.4f} ± {std_val_accuracy:.4f}")
        print(f"  平均最佳驗證準確率: {avg_best_val_accuracy:.4f}")
        
        # 尋找最佳fold
        best_fold_idx = np.argmax([r['val_accuracy'] for r in valid_results])
        best_fold = valid_results[best_fold_idx]
        print(f"\n最佳fold: Fold {best_fold['fold']}")
        print(f"  驗證準確率: {best_fold['val_accuracy']:.4f}")
        print(f"  最佳模型路徑: {best_fold['best_model_path']}")
    
    # 詳細結果列表
    print(f"\n各fold詳細結果:")
    print(f"{'Fold':<6} {'Val Loss':<10} {'Val Acc':<10} {'Best Val Acc':<12} {'Best Epoch':<11} {'Total Epochs':<12} {'Model Path'}")
    print("-" * 100)
    for result in fold_results:
        if 'error' not in result:
            model_name = os.path.basename(result['best_model_path']) if result['best_model_path'] else 'None'
            print(f"{result['fold']:<6} {result['val_loss']:<10.4f} {result['val_accuracy']:<10.4f} "
                  f"{result['best_val_accuracy']:<12.4f} {result.get('best_epoch', 'N/A'):<11} "
                  f"{result['epochs_trained']:<12} {model_name}")
        else:
            print(f"{result['fold']:<6} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12} {'ERROR':<11} {'ERROR':<12} {result['error']}")
    
    # 第二階段：使用所有資料訓練最終模型
    print(f"\n{'='*80}")  
    print(f"第二階段：使用所有資料訓練最終模型")
    print(f"{'='*80}")
    
    try:
        # 計算K折交叉驗證中最佳結果的平均epochs數
        valid_results_with_best_epoch = [r for r in valid_results if 'best_epoch' in r]
        if valid_results_with_best_epoch:
            final_epochs = int(np.mean([r['best_epoch'] for r in valid_results_with_best_epoch]))
            print(f"根據K折交叉驗證最佳結果的平均epochs數設定最終訓練輪數: {final_epochs}")
            print(f"各fold最佳epochs: {[r['best_epoch'] for r in valid_results_with_best_epoch]}")
        else:
            final_epochs = 50  # 如果沒有有效的歷史記錄，使用預設值  
            print(f"使用預設epochs數: {final_epochs}")
        
        # 建立包含所有資料的訓練目錄
        full_data_dir = create_full_data_directory('data')
        
        # 建立最終模型的資料生成器（包含數據增強，使用所有資料訓練）
        final_train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            zoom_range=0.2,
            shear_range=0.2,
            brightness_range=[0.8, 1.2],
            horizontal_flip=True
        )
        
        # 訓練資料生成器（使用所有資料）
        final_train_generator = final_train_datagen.flow_from_directory(
            full_data_dir,
            target_size=(360, 640),
            batch_size=32,
            class_mode='categorical',
            shuffle=True
        )
        
        print(f"最終訓練 - 類別索引對應: {final_train_generator.class_indices}")
        print(f"最終訓練 - 訓練資料: {final_train_generator.samples} 張（使用所有資料）")
        
        # 建立最終模型
        final_model = create_resnet_model(input_shape=(360, 640, 3), num_classes=3)
        
        # 最終模型的權重檔案路徑（固定檔名，因為不需要監控驗證準確率）
        final_model_path = f'best_resnet_model_360x640_{current_date}_rs{random_seed}_epochs{final_epochs}.h5'
        
        # 設定回調函數（只使用學習率調整，不使用EarlyStopping）
        final_lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, min_lr=1e-6)
        
        print(f"\n開始最終模型訓練（固定 {final_epochs} epochs，使用所有資料）...")
        
        # 訓練最終模型
        final_history = final_model.fit(
            final_train_generator,
            epochs=final_epochs,
            callbacks=[final_lr_reducer],
            workers=1,
            use_multiprocessing=False
        )
        
        # 保存最終模型權重
        final_model.save_weights(final_model_path)
        print(f"\n保存最終模型權重: {final_model_path}")
        
        # 顯示最終模型結果
        final_loss = final_history.history['loss'][-1]
        final_accuracy = final_history.history['accuracy'][-1]
        
        print(f"\n最終模型結果:")
        print(f"  最終訓練損失: {final_loss:.4f}")
        print(f"  最終訓練準確率: {final_accuracy:.4f}")
        print(f"  訓練epochs數: {len(final_history.history['loss'])}")
        print(f"  最終模型權重: {final_model_path}")
        
    except Exception as e:
        print(f"最終模型訓練過程發生錯誤 main: {e}")
        final_model_path = None
        final_accuracy = 0.0
        final_history = None
        
    finally:
        # 清理最終訓練的臨時目錄
        cleanup_directory(full_data_dir)
    
    # 保存完整結果到JSON檔案
    results_filename = f'kfold_results_{current_date}_rs{random_seed}.json'
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump({
            'k_folds': K_FOLDS,
            'random_seed': random_seed,
            'fold_results': fold_results,
            'statistics': {
                'avg_val_loss': avg_val_loss if valid_results else None,
                'avg_val_accuracy': avg_val_accuracy if valid_results else None,
                'std_val_accuracy': std_val_accuracy if valid_results else None,
                'avg_best_val_accuracy': avg_best_val_accuracy if valid_results else None
            } if valid_results else None,
            'final_model': {
                'model_path': final_model_path,
                'train_accuracy': final_accuracy if 'final_accuracy' in locals() else None,
                'epochs_trained': len(final_history.history['loss']) if final_history else None
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果已保存到: {results_filename}")
    
    # 繪製所有fold的訓練曲線比較圖
    if all_histories:
        fig_rows = 3 if final_history else 2
        plt.figure(figsize=(15, 5 * fig_rows))
        
        # K折驗證結果圖
        # 損失圖
        plt.subplot(fig_rows, 2, 1)
        for i, history in enumerate(all_histories):
            if 'loss' in history:
                plt.plot(history['loss'], label=f'Fold {i+1} Train', alpha=0.7)
        plt.title('K-Fold Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(fig_rows, 2, 2)
        for i, history in enumerate(all_histories):
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label=f'Fold {i+1} Val', alpha=0.7)
        plt.title('K-Fold Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 準確率圖
        plt.subplot(fig_rows, 2, 3)
        for i, history in enumerate(all_histories):
            if 'accuracy' in history:
                plt.plot(history['accuracy'], label=f'Fold {i+1} Train', alpha=0.7)
        plt.title('K-Fold Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(fig_rows, 2, 4)
        for i, history in enumerate(all_histories):
            if 'val_accuracy' in history:
                plt.plot(history['val_accuracy'], label=f'Fold {i+1} Val', alpha=0.7)
        plt.title('K-Fold Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 最終模型訓練曲線（只有訓練資料，無驗證資料）
        if final_history:
            plt.subplot(fig_rows, 2, 5)
            plt.plot(final_history.history['loss'], label='Final Train Loss', color='red', linewidth=2)
            plt.title('Final Model Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(fig_rows, 2, 6)
            plt.plot(final_history.history['accuracy'], label='Final Train Accuracy', color='blue', linewidth=2)
            plt.title('Final Model Training Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'kfold_training_plots_{current_date}_rs{random_seed}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 總結報告
    print(f"\n{'='*80}")
    print(f"K折交叉驗證 + 最終模型訓練完成！")
    print(f"{'='*80}")
    
    if valid_results:
        print(f"\nK折交叉驗證性能評估:")
        print(f"  平均驗證準確率: {avg_val_accuracy:.4f} ± {std_val_accuracy:.4f}")
        print(f"  最佳fold準確率: {max([r['val_accuracy'] for r in valid_results]):.4f}")
    
    if final_model_path:
        print(f"\n最終生產模型:")
        print(f"  模型權重檔案: {final_model_path}")
        if 'final_accuracy' in locals():
            print(f"  訓練準確率: {final_accuracy:.4f}")
        print(f"  建議使用此模型進行實際推論任務")
    
    print(f"\n完整結果已保存到: {results_filename}")

# 執行主程式
if __name__ == "__main__":
    main()