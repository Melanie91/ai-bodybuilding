import mysql.connector
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. 連接到 MySQL 資料庫並提取所有數據
conn = mysql.connector.connect(
    host="localhost",  # MySQL 伺服器地址
    user="root",  # MySQL 使用者名稱
    password="a85432433",  # MySQL 密碼
    database="pose_data_db"  # 資料庫名稱
)

cursor = conn.cursor()

# 使用具體的特徵列名稱來提取數據，這裡省略 'image_file' 並只提取關鍵點特徵
cursor.execute("""
    SELECT NOSE_1_X, NOSE_1_Y, LEFT_EYE_INNER_2_X, LEFT_EYE_INNER_2_Y, 
        LEFT_EYE_3_X, LEFT_EYE_3_Y, LEFT_EYE_OUTER_4_X, LEFT_EYE_OUTER_4_Y,
        RIGHT_EYE_INNER_5_X, RIGHT_EYE_INNER_5_Y, RIGHT_EYE_6_X, RIGHT_EYE_6_Y,
        RIGHT_EYE_OUTER_7_X, RIGHT_EYE_OUTER_7_Y, LEFT_EAR_8_X, LEFT_EAR_8_Y,
        RIGHT_EAR_9_X, RIGHT_EAR_9_Y, MOUTH_LEFT_10_X, MOUTH_LEFT_10_Y, 
        MOUTH_RIGHT_11_X, MOUTH_RIGHT_11_Y, LEFT_SHOULDER_12_X, LEFT_SHOULDER_12_Y, 
        RIGHT_SHOULDER_13_X, RIGHT_SHOULDER_13_Y, LEFT_ELBOW_14_X, LEFT_ELBOW_14_Y, 
        RIGHT_ELBOW_15_X, RIGHT_ELBOW_15_Y, LEFT_WRIST_16_X, LEFT_WRIST_16_Y, 
        RIGHT_WRIST_17_X, RIGHT_WRIST_17_Y, LEFT_PINKY_18_X, LEFT_PINKY_18_Y, 
        RIGHT_PINKY_19_X, RIGHT_PINKY_19_Y, LEFT_INDEX_20_X, LEFT_INDEX_20_Y, 
        RIGHT_INDEX_21_X, RIGHT_INDEX_21_Y, LEFT_THUMB_22_X, LEFT_THUMB_22_Y, 
        RIGHT_THUMB_23_X, RIGHT_THUMB_23_Y, LEFT_HIP_24_X, LEFT_HIP_24_Y, 
        RIGHT_HIP_25_X, RIGHT_HIP_25_Y, LEFT_KNEE_26_X, LEFT_KNEE_26_Y, 
        RIGHT_KNEE_27_X, RIGHT_KNEE_27_Y, LEFT_ANKLE_28_X, LEFT_ANKLE_28_Y, 
        RIGHT_ANKLE_29_X, RIGHT_ANKLE_29_Y, LEFT_HEEL_30_X, LEFT_HEEL_30_Y, 
        RIGHT_HEEL_31_X, RIGHT_HEEL_31_Y, LEFT_FOOT_INDEX_32_X, LEFT_FOOT_INDEX_32_Y, 
        RIGHT_FOOT_INDEX_33_X, RIGHT_FOOT_INDEX_33_Y, pose_label
    FROM pose_lms_front_latspread
""")
data = cursor.fetchall()

# 關閉資料庫連接
conn.close()

# 將數據轉換成 NumPy 陣列
data = np.array(data)

# 分離特徵和標籤
X = data[:, :-1]  # 提取前 66 個特徵
y = data[:, -1]   # 提取最後一個欄位作為標籤

# 將特徵轉換為 float32 以避免數據類型錯誤
X = X.astype(np.float32)
y = y.astype(np.float32)

# 將數據 reshape 成 CNN 輸入所需的形狀
try:
    X = X.reshape(-1, 33, 2, 1)
    print(f"reshape 後的 X 形狀：{X.shape}")
except ValueError as e:
    print(f"重塑數據時發生錯誤: {e}")
    exit()

# 標準化
X = X / np.max(X)

# 切分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 構建 CNN 分類模型
model = models.Sequential([
    layers.Input(shape=(33, 2, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # 二元分類，輸出 0 或 1
])

# 編譯模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # 二元分類的損失函數
              metrics=['accuracy'])

# 定義自定義回調來計算精確率、召回率和 F1 值
class MetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"Epoch {epoch + 1} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# 訓練模型並顯示精確率、召回率和 F1 值
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[MetricsCallback()])

# 保存模型到具體路徑
model.save("C:/pose/front_latspread_cnn.keras")
