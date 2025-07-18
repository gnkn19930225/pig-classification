import os
import h5py

def check_model_file(file_path):
    """
    檢查模型檔案的內容和格式
    """
    print(f"檢查檔案: {file_path}")
    print(f"檔案大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    
    try:
        # 檢查 HDF5 檔案結構
        with h5py.File(file_path, 'r') as f:
            print("\nHDF5 檔案結構:")
            def print_structure(name, obj):
                print(f"  {name}: {type(obj)}")
            f.visititems(print_structure)
            
            # 檢查是否包含模型配置
            if 'model_config' in f.attrs:
                print("✓ 找到模型配置")
                return "complete_model"
            elif 'layer_names' in f.attrs:
                print("⚠ 只找到權重資料，缺少模型配置")
                return "weights_only"
            else:
                print("❌ 無法確定檔案類型")
                return "unknown"
                
    except Exception as e:
        print(f"❌ 讀取檔案時發生錯誤: {e}")
        return "error"

def rebuild_and_convert(weights_path, onnx_output_path):
    """
    重建模型架構並載入權重，然後轉換為 ONNX
    """
    try:
        import tensorflow as tf
        from model_architecture import create_resnet_model
        
        print("正在重建模型架構...")
        
        # 使用共用模型架構建立模型
        model = create_resnet_model(input_shape=(360, 640, 3), num_classes=3)
        
        print("✓ 模型架構重建完成")
        print("\n模型摘要:")
        model.summary()
        
        # 載入權重
        print(f"\n正在載入權重: {weights_path}")
        model.load_weights(weights_path)
        print("✓ 權重載入成功")
        
        # 測試模型推論
        print("\n正在測試模型推論...")
        import numpy as np
        test_input = np.random.random((1, 360, 640, 3)).astype(np.float32)
        test_output = model.predict(test_input, verbose=0)
        print(f"✓ 推論測試成功，輸出形狀: {test_output.shape}")
        print(f"  輸出總和: {test_output.sum():.6f}")
        
        # 轉換為 ONNX
        print(f"\n開始轉換為 ONNX...")
        
        # 保存為完整的 Keras 模型（臨時）
        temp_model_path = "temp_complete_model.h5"
        model.save(temp_model_path)
        print(f"✓ 臨時完整模型已保存: {temp_model_path}")
        
        # 現在使用 tf2onnx 轉換
        import tf2onnx
        
        spec = (tf.TensorSpec((None, 360, 640, 3), tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=spec,
            opset=13,
            output_path=onnx_output_path
        )
        
        print(f"✓ ONNX 轉換完成: {onnx_output_path}")
        
        # 清理臨時檔案
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
            print("✓ 臨時檔案已清理")
            
        # 驗證 ONNX 模型
        print("\n正在驗證 ONNX 模型...")
        import onnx
        onnx_model = onnx.load(onnx_output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX 模型驗證通過")
        
        return True
        
    except Exception as e:
        print(f"❌ 轉換失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 尋找最新的權重檔案
    weight_files = [f for f in os.listdir('.') if f.startswith('best_resnet_model_360x640_') and f.endswith('.h5')]
    
    if not weight_files:
        print("❌ 找不到權重檔案")
        print("請確保權重檔案在當前目錄中，檔案名稱格式: best_resnet_model_360x640_*.h5")
        exit(1)
    
    # 使用最新的權重檔案
    latest_weights = max(weight_files, key=os.path.getctime)
    print(f"找到權重檔案: {latest_weights}")
    
    # 檢查檔案類型
    file_type = check_model_file(latest_weights)
    
    if file_type == "complete_model":
        print("\n這是完整的模型檔案，使用原始轉換腳本")
        os.system("python convert_to_onnx.py")
    elif file_type == "weights_only":
        print("\n這是權重檔案，開始重建模型並轉換...")
        # 根據權重檔案名稱生成對應的 ONNX 檔案名稱
        onnx_output = latest_weights.replace('.h5', '.onnx')
        
        if rebuild_and_convert(latest_weights, onnx_output):
            print(f"\n🎉 轉換成功！")
            print(f"ONNX 模型已保存為: {onnx_output}")
            print(f"現在可以在 .NET Core 中使用這個模型了")
        else:
            print("\n❌ 轉換失敗")
    else:
        print(f"\n❌ 無法處理此檔案類型: {file_type}") 