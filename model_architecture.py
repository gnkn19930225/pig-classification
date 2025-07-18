from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications.resnet50 import preprocess_input

def create_resnet_model(input_shape=(360, 640, 3), num_classes=3):
    """
    建立與訓練時完全一致的 ResNet50 模型架構
    
    Args:
        input_shape: 輸入圖片尺寸，預設為 (360, 640, 3)
        num_classes: 分類類別數量，預設為 3
    
    Returns:
        model: 編譯好的 Keras 模型
    """
    # 建立 ResNet50 基礎模型，使用 ImageNet 預訓練權重
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # 改進模型結構（與訓練時完全一致）
    x = base_model.output                    # 獲取基礎模型的輸出
    x = GlobalAveragePooling2D()(x)          # 全域平均池化層，將特徵圖轉為向量
    x = Dense(512, activation='relu')(x)     # 全連接層，512個神經元，ReLU激活函數
    x = BatchNormalization()(x)              # 批次正規化層，加速訓練並提高穩定性
    x = Dropout(0.3)(x)                      # Dropout層，30%的機率隨機關閉神經元，防止過擬合
    predictions = Dense(num_classes, activation='softmax')(x)  # 輸出層，使用softmax激活函數
    
    # 建立完整模型
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # 凍結基礎模型的權重，只訓練新增的分類層
    for layer in base_model.layers:
        layer.trainable = False
    
    # 編譯模型（與訓練時完全一致）
    model.compile(
        optimizer='adam',               # 使用Adam優化器
        loss='categorical_crossentropy',     # 多分類交叉熵損失函數
        metrics=['accuracy', 'top_k_categorical_accuracy']  # 監控準確率和top-k準確率指標
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