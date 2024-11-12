import os
import cv2
import mediapipe as mp
import mysql.connector

# 初始化 Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# 設定圖片文件夾路徑
correct_folder = r'C:\pose\front_lat_spread_new'
incorrect_folder = r'C:\pose\wrong_pose_new'

# 建立 MySQL 連接
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="a85432433",
    database="pose_data_db"
)
cursor = conn.cursor()

# 查詢圖片是否已在資料庫中的 SQL 語句
sql_check = "SELECT COUNT(*) FROM pose_lms_front_latspread WHERE image_file = %s"

# 插入資料的 SQL 語句
sql_insert = """
INSERT INTO pose_lms_front_latspread (
    image_file, NOSE_1_X, NOSE_1_Y, LEFT_EYE_INNER_2_X, LEFT_EYE_INNER_2_Y, 
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
) 
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s)
"""

# 定義用於檢查和插入圖片的函數
def process_images_in_folder(folder, label):
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg'))]
    for image_file in image_files:
        image_path = os.path.join(folder, image_file)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"無法讀取圖片：{image_file}")
            continue

        # 檢查圖片是否已存在於資料庫中
        cursor.execute(sql_check, (image_file,))
        if cursor.fetchone()[0] > 0:
            print(f"圖片 {image_file} 已存在，跳過。")
            continue

        # 將圖片轉換為 RGB 並偵測姿勢
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
            flattened_landmarks = [float(coord) for lm in landmarks for coord in [lm[0], lm[1]]]

            # 補齊 landmarks 數量至 66（33 個點位，每個點位 x 和 y 坐標）
            while len(flattened_landmarks) < 66:
                flattened_landmarks.append(0.0)

            # 插入數據至資料庫
            if len(flattened_landmarks) == 66:
                val = (image_file, *flattened_landmarks, label)
                try:
                    cursor.execute(sql_insert, val)
                    conn.commit()
                    print(f"成功插入圖片 {image_file} 的數據（標籤 {label}）。")
                except mysql.connector.Error as err:
                    print(f"插入數據 {image_file} 時發生錯誤: {err}")
            else:
                print(f"錯誤: 圖片 {image_file} 的 landmarks 數量不正確，應為 66，實際為 {len(flattened_landmarks)}")
        else:
            print(f"圖片 {image_file} 未能檢測到姿勢標誌。")

# 處理正確和不正確的資料夾
process_images_in_folder(correct_folder, 1)  # 標籤 1 表示正確
process_images_in_folder(incorrect_folder, 0)  # 標籤 0 表示不正確

# 關閉資源
cursor.close()
conn.close()
pose.close()
