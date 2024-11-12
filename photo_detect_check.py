import os
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# 初始化 MediaPipe 的 Pose 模組
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 設定圖片資料夾路徑
image_folder_path = r'C:\pose\front_lat_spread'  # 替換成你的資料夾路徑

# 確保資料夾存在
if not os.path.exists(image_folder_path):
    print(f"Error: The folder path {image_folder_path} does not exist.")
    exit()

# 讀取圖片文件
image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# 檢查是否有圖片
if not image_files:
    print("No images found in the folder.")
    exit()

# 設置基於三個基準點的標準化方法
def get_three_point_reference_distance(landmarks):
    # 選擇三個基準點：左肩膀、右肩膀、髖部中心
    left_shoulder = np.array([landmarks[11].x, landmarks[11].y])
    right_shoulder = np.array([landmarks[12].x, landmarks[12].y])
    left_hip = np.array([landmarks[23].x, landmarks[23].y])
    right_hip = np.array([landmarks[24].x, landmarks[24].y])

    # 計算肩膀中心點
    shoulder_center = (left_shoulder + right_shoulder) / 2
    # 計算髖部中心點
    hip_center = (left_hip + right_hip) / 2

    # 計算肩膀和髖部中心點之間的距離作為參考
    torso_distance = np.linalg.norm(shoulder_center - hip_center)

    # 計算肩膀之間的距離
    shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)

    # 返回這兩個距離的平均值作為標準化參照
    return (torso_distance + shoulder_distance) / 2

# 設定新的資料夾路徑來保存校正後的圖片
output_folder_path = r'C:\pose\front_lat_spread_new'  # 替換成你要保存結果的資料夾
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 開始處理每張圖片
for image_file in image_files:
    image_path = os.path.join(image_folder_path, image_file)
    output_image_path = os.path.join(output_folder_path, image_file)

    # 如果校正後的圖片已經存在，跳過該圖片
    if os.path.exists(output_image_path):
        print(f"Image {image_file} already processed, skipping.")
        continue

    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image {image_file}. Skipping.")
        continue
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用 MediaPipe 進行姿勢偵測
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 計算基於三個基準點的距離作為參考
        reference_distance = get_three_point_reference_distance(landmarks)
        if reference_distance == 0:  # 避免距離為零
            print(f"Reference distance is zero in {image_file}, skipping this image.")
            continue

        # 標準化每個關鍵點
        normalized_landmarks = []
        for lm in landmarks:
            x_normalized = lm.x / reference_distance
            y_normalized = lm.y / reference_distance
            normalized_landmarks.append((x_normalized, y_normalized))

        # 視覺化標準化後的關鍵點
        image_height, image_width, _ = image.shape
        for idx, (x, y) in enumerate(normalized_landmarks):
            # 將標準化座標轉換回像素座標進行視覺化（為了檢查，可以暫時還原）
            x_pixel = int(x * reference_distance * image_width)
            y_pixel = int(y * reference_distance * image_height)
            cv2.circle(image, (x_pixel, y_pixel), 5, (0, 255, 0), -1)  # 綠色點標記關鍵點
            cv2.putText(image, str(idx), (x_pixel, y_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 保存校正後的圖片到新的資料夾
        cv2.imwrite(output_image_path, image)
        print(f"Processed and saved image: {image_file}")
    else:
        print(f"No landmarks detected in {image_file}.")
