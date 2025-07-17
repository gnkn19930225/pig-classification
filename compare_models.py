from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.resnet50 import preprocess_input
import onnxruntime as ort
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
import io
import time
import os
from datetime import datetime, timezone, timedelta
import csv

# 牧場設定字典
# 格式：{牧場名稱: {主機位址, 輸出目錄}}
FARM_SETTINGS = {
    "115": {
        "host": "http://192.168.115.190:5000/",
        "output_dir": "路竹場"
    },
    "50": {
        "host": "http://192.168.50.190:5000/",
        "output_dir": "布袋場"
    },
    "109": {
        "host": "http://192.168.109.190:5000/",
        "output_dir": "六甲一場"
    },
    "116": {
        "host": "http://192.168.116.190:5000/",
        "output_dir": "鹽埔場"
    },
    "114": {
        "host": "http://192.168.114.190:5000/",
        "output_dir": "麟洛場"
    },
    "110": {
        "host": "http://192.168.110.190:5000/",
        "output_dir": "六甲二場"
    },
    "111": {
        "host": "http://192.168.111.190:5000/",
        "output_dir": "芳苑場"
    },
    "107": {
        "host": "http://192.168.107.100:5000/",
        "output_dir": "水林場"
    },
    "117": {
        "host": "http://192.168.117.190:5000/",
        "output_dir": "桐德場"
    },
    "105": {
        "host": "http://192.168.105.190:5000/",
        "output_dir": "里港場"
    },
    "106": {
        "host": "http://192.168.106.190:5000/",
        "output_dir": "福川場"
    },
    "108": {
        "host": "http://192.168.108.190:5000/",
        "output_dir": "關廟場"
    },
    "112": {
        "host": "http://192.168.112.190:5000/",
        "output_dir": "大仁場"
    }
}

def rebuild_h5_model():
    """重建 H5 模型架構"""
    print("正在重建 H5 模型架構...")
    # 建立 ResNet50 基礎模型
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(360, 640, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # 凍結基礎模型的權重
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("✓ H5 模型架構重建完成")
    return model

def get_farm_cameras(farm_host):
    """獲取牧場的攝影機列表"""
    try:
        # 發送 API 請求獲取攝影機配置
        response = requests.get(f"{farm_host}/api/config", timeout=10)
        response.raise_for_status()
        
        cameras_raw = response.json()["cameras"]
        cameras = []
        
        # 去除 _main 或 _sub 後綴，避免重複
        for camera in cameras_raw.keys():
            base_name = camera.replace("_main", "").replace("_sub", "")
            if base_name not in cameras:
                cameras.append(base_name)
        
        return cameras
    except Exception as e:
        print(f"❌ 獲取攝影機列表失敗: {e}")
        return []

def download_farm_image(farm_host, camera_name, target_size=(640, 360)):
    """從牧場下載指定攝影機的圖片"""
    try:
        # 建立攝影機圖片的 URL
        snapshot_url = f"{farm_host}/api/{camera_name}/latest.jpg"
        print(f"正在下載: {snapshot_url}")
        
        # 下載圖片
        response = requests.get(snapshot_url, timeout=10)
        response.raise_for_status()
        
        # 載入並轉換圖片
        image = Image.open(io.BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"✓ 圖片下載成功，原始尺寸: {image.size}")
        
        # 調整圖片尺寸
        resized_image = image.resize(target_size)
        print(f"✓ 圖片尺寸調整為: {target_size}")
        
        return resized_image
        
    except Exception as e:
        print(f"❌ 下載圖片失敗: {e}")
        return None

def preprocess_for_h5(image):
    """H5 模型圖片預處理"""
    # 轉換為 numpy 陣列並增加 batch 維度
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    # 使用 TensorFlow ResNet50 預處理
    img_array = preprocess_input(img_array)
    print(f"H5 預處理後形狀: {img_array.shape}")
    print(f"H5 預處理後數值範圍: [{img_array.min():.3f}, {img_array.max():.3f}]")
    return img_array

def preprocess_for_onnx(image):
    """ONNX 模型圖片預處理 - 與 H5 完全相同"""
    # 轉換為 numpy 陣列並增加 batch 維度
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 使用與 TensorFlow 相同的預處理方式
    # RGB 轉 BGR，然後減去 ImageNet 均值
    img_array = img_array[..., ::-1]  # RGB 轉 BGR
    mean = [103.939, 116.779, 123.68]  # BGR 通道的 ImageNet 均值
    img_array[..., 0] -= mean[0]  # B 通道
    img_array[..., 1] -= mean[1]  # G 通道  
    img_array[..., 2] -= mean[2]  # R 通道
    
    print(f"ONNX 預處理後形狀: {img_array.shape}")
    print(f"ONNX 預處理後數值範圍: [{img_array.min():.3f}, {img_array.max():.3f}]")
    return img_array

def predict_with_h5(model, img_array):
    """使用 H5 模型進行預測"""
    print("=== H5 模型推論 ===")
    # 執行預測
    predictions = model.predict(img_array, verbose=0)
    return predictions[0]

def predict_with_onnx(onnx_path, img_array):
    """使用 ONNX 模型進行預測"""
    print("=== ONNX 模型推論 ===")
    try:
        # 載入 ONNX 模型
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        print(f"ONNX 輸入名稱: {input_name}")
        print(f"ONNX 期望輸入形狀: {ort_session.get_inputs()[0].shape}")
        
        # 執行推論
        predictions = ort_session.run([output_name], {input_name: img_array})
        return predictions[0][0]
        
    except Exception as e:
        print(f"❌ ONNX 推論失敗: {e}")
        return None

def compare_predictions(h5_pred, onnx_pred):
    """對比兩個模型的預測結果"""
    class_names = ['camera_failure', 'with_pigs', 'without_pigs']
    
    print("\n" + "="*60)
    print("預測結果對比:")
    print("="*60)
    
    # 處理 H5 模型結果
    h5_class = int(np.argmax(h5_pred))
    h5_confidence = h5_pred[h5_class]
    h5_label = class_names[h5_class]
    
    # 處理 ONNX 模型結果
    if onnx_pred is not None:
        onnx_class = int(np.argmax(onnx_pred))
        onnx_confidence = onnx_pred[onnx_class]
        onnx_label = class_names[onnx_class]
        
        print(f"H5   模型預測: {h5_label} (信心度: {h5_confidence:.4f})")
        print(f"ONNX 模型預測: {onnx_label} (信心度: {onnx_confidence:.4f})")
        
        # 檢查預測結果是否一致
        if h5_class == onnx_class:
            print("✓ 兩個模型預測結果一致")
            is_consistent = True
        else:
            print("❌ 兩個模型預測結果不同！")
            is_consistent = False
            
        # 顯示詳細機率對比
        print("\n詳細機率對比:")
        print(f"{'類別':<15} {'H5 機率':<10} {'ONNX 機率':<10} {'差異':<10}")
        print("-" * 50)
        max_diff = 0
        for i, class_name in enumerate(class_names):
            diff = abs(h5_pred[i] - onnx_pred[i])
            max_diff = max(max_diff, diff)
            print(f"{class_name:<15} {h5_pred[i]:<10.4f} {onnx_pred[i]:<10.4f} {diff:<10.4f}")
            
    else:
        print(f"H5 模型預測: {h5_label} (信心度: {h5_confidence:.4f})")
        print("ONNX 模型預測失敗")
        is_consistent = False
        max_diff = float('inf')
    
    print("="*60)
    
    # 回傳比較結果
    return {
        'h5_prediction': h5_label,
        'h5_confidence': h5_confidence,
        'onnx_prediction': onnx_label if onnx_pred is not None else None,
        'onnx_confidence': onnx_confidence if onnx_pred is not None else None,
        'is_consistent': is_consistent,
        'max_difference': max_diff if onnx_pred is not None else float('inf')
    }

def plot_comparison(image, h5_pred, onnx_pred, farm_name, camera_name):
    """繪製對比圖表"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 顯示原始圖片
    axes[0].imshow(image)
    axes[0].set_title(f"Test Image\nFarm: {farm_name}\nCamera: {camera_name}")
    axes[0].axis('off')
    
    class_names = ['camera_failure', 'with_pigs', 'without_pigs']
    
    # H5 模型結果
    if h5_pred is not None:
        h5_class = int(np.argmax(h5_pred))
        h5_confidence = h5_pred[h5_class]
        h5_colors = ['red' if i == h5_class else 'lightblue' for i in range(3)]
        
        axes[1].bar(class_names, h5_pred, color=h5_colors)
        axes[1].set_title(f"H5 Model\n{class_names[h5_class]}\nConf: {h5_confidence:.4f}")
        axes[1].set_ylabel('Probability')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_ylim(0, 1)
    
    # ONNX 模型結果
    if onnx_pred is not None:
        onnx_class = int(np.argmax(onnx_pred))
        onnx_confidence = onnx_pred[onnx_class]
        onnx_colors = ['red' if i == onnx_class else 'lightgreen' for i in range(3)]
        
        axes[2].bar(class_names, onnx_pred, color=onnx_colors)
        axes[2].set_title(f"ONNX Model\n{class_names[onnx_class]}\nConf: {onnx_confidence:.4f}")
        axes[2].set_ylabel('Probability')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].set_ylim(0, 1)
    else:
        axes[2].text(0.5, 0.5, 'ONNX\nPrediction\nFailed', 
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title("ONNX Model - Error")
    
    plt.tight_layout()
    plt.show()

def save_comparison_results(results, output_file='comparison_results.csv'):
    """將比較結果保存到 CSV 檔案"""
    # 檢查檔案是否存在，如果不存在則建立標題列
    file_exists = os.path.exists(output_file)
    
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'farm_name', 'camera_name', 'h5_prediction', 'h5_confidence', 
                     'onnx_prediction', 'onnx_confidence', 'is_consistent', 'max_difference']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 如果檔案不存在，寫入標題列
        if not file_exists:
            writer.writeheader()
        
        # 寫入結果資料
        writer.writerow(results)
    
    print(f"✓ 結果已保存至 {output_file}")

def process_farm_batch(farm_name, farm_setting, h5_model, onnx_model_path):
    """批次處理單一牧場的所有攝影機"""
    print(f"\n{'='*80}")
    print(f"開始處理牧場: {farm_name}")
    print(f"{'='*80}")
    
    try:
        # 獲取牧場的攝影機列表
        cameras = get_farm_cameras(farm_setting['host'])
        if not cameras:
            print(f"❌ 無法獲取 {farm_name} 的攝影機列表")
            return
        
        print(f"✓ 找到 {len(cameras)} 台攝影機: {cameras}")
        
        # 處理每台攝影機
        for camera_name in cameras:
            print(f"\n--- 處理攝影機: {camera_name} ---")
            
            try:
                # 下載圖片
                image = download_farm_image(farm_setting['host'], camera_name)
                if image is None:
                    continue
                
                # 預處理圖片
                h5_input = preprocess_for_h5(image)
                onnx_input = preprocess_for_onnx(image)
                
                # 進行預測
                h5_pred = predict_with_h5(h5_model, h5_input)
                onnx_pred = predict_with_onnx(onnx_model_path, onnx_input)
                
                # 比較結果
                comparison = compare_predictions(h5_pred, onnx_pred)
                
                # 顯示比較圖表
                plot_comparison(image, h5_pred, onnx_pred, farm_name, camera_name)
                
                # 準備保存資料
                timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
                result_data = {
                    'timestamp': timestamp,
                    'farm_name': farm_name,
                    'camera_name': camera_name,
                    'h5_prediction': comparison['h5_prediction'],
                    'h5_confidence': comparison['h5_confidence'],
                    'onnx_prediction': comparison['onnx_prediction'],
                    'onnx_confidence': comparison['onnx_confidence'],
                    'is_consistent': comparison['is_consistent'],
                    'max_difference': comparison['max_difference']
                }
                
                # 保存結果到 CSV
                save_comparison_results(result_data)
                
                # 暫停一下讓用戶檢視圖表
                input("按 Enter 繼續下一張圖片...")
                plt.close('all')  # 關閉當前圖表
                
            except Exception as e:
                print(f"❌ 處理攝影機 {camera_name} 時發生錯誤: {e}")
                continue
                
    except Exception as e:
        print(f"❌ 處理牧場 {farm_name} 時發生錯誤: {e}")

def main():
    """主函數 - 批次處理所有牧場"""
    # 模型檔案路徑
    h5_model_path = "best_resnet_model_360x640_20250702_1609_val_acc_0.9787.h5"
    onnx_model_path = "best_resnet_model_360x640_20250702_1609_val_acc_0.9787.onnx"
    
    try:
        # 載入 H5 模型
        print("載入 H5 模型...")
        h5_model = rebuild_h5_model()
        h5_model.load_weights(h5_model_path)
        print("✓ H5 模型載入成功")
        
        # 建立結果輸出目錄
        os.makedirs("comparison_results", exist_ok=True)
        
        # 處理每個牧場
        total_farms = len(FARM_SETTINGS)
        for i, (farm_name, farm_setting) in enumerate(FARM_SETTINGS.items(), 1):
            print(f"\n進度: {i}/{total_farms} - 處理牧場: {farm_name}")
            
            # 批次處理牧場
            process_farm_batch(farm_name, farm_setting, h5_model, onnx_model_path)
            
            # 在處理完每個牧場後稍微暫停
            time.sleep(1)
        
        print(f"\n{'='*80}")
        print("所有牧場處理完成！")
        print("結果已保存至 comparison_results.csv")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"❌ 主程式執行失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 