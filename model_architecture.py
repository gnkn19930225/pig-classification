from tensorflow.keras.applications import ResNet152
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications.resnet import preprocess_input

def create_resnet_model(input_shape=(360, 640, 3), num_classes=3):
    """
    建立 ResNet50 模型架構（Fine-tuning 版本）
    
    Args:
        input_shape: 輸入圖片尺寸，預設為 (360, 640, 3)
        num_classes: 分類類別數量，預設為 3
    
    Returns:
        model: 編譯好的 Keras 模型
    """
    # 建立 ResNet50 基礎模型，使用 ImageNet 預訓練權重
    base_model = ResNet152(weights="imagenet", include_top=False, input_shape=input_shape)
    
    # 改進模型結構（與訓練時完全一致）
    x = base_model.output                    # 獲取基礎模型的輸出
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # 凍結基礎模型的權重，只訓練新增的分類層
    for layer in base_model.layers:
        layer.trainable = False
    
    # 編譯模型（與訓練時完全一致）
    model.compile(
        optimizer='adam',               # 使用Adam優化器
        loss='categorical_crossentropy',     # 多分類交叉熵損失函數
        metrics=['accuracy']  # 監控準確率和top-k準確率指標
    )
    
    return model

def get_preprocess_function():
    """
    取得圖片預處理函數
    
    Returns:
        preprocess_input: ResNet50 的預處理函數
    """
    return preprocess_input

def get_class_names():
    """
    取得類別名稱列表
    
    Returns:
        list: 類別名稱列表
    """
    return ['camera_failure', 'with_pigs', 'without_pigs'] 