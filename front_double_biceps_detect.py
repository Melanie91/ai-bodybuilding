import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os
import time

# Disable GPU acceleration (if no GPU support)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 初始化 MediaPipe Pose 模組
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# 嘗試加載訓練好的模型並打印模型摘要
try:
    model = load_model('C:/pose/1119front_db_cnn_new.keras')
    input_shape = model.input_shape
    print(f"模型預期的輸入形狀: {input_shape}")
except Exception as e:
    print(f"加載模型時出錯: {e}")
    exit()

# 打開攝像頭
cap = cv2.VideoCapture(0)  # 0 indicates the default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# 設置攝像頭參數
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 設置 OpenCV 視窗屬性
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', 1280, 720)  # 調整窗口大小為 1280x720
cv2.setWindowProperty('Camera Feed', cv2.WND_PROP_TOPMOST, 1)  # 設置窗口為最前面

# 顯示姿勢名稱並等待 10 秒
pose_name = "Front Double Biceps"
start_time = time.time()  # 記錄開始時間
show_in_center = True  # 控制顯示位置的變數
detect_started = False  # 用來判斷是否已經開始偵測

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # 判斷是否已經過了 10 秒鐘，改變顯示位置並開始偵測
    if time.time() - start_time >= 10 and not detect_started:
        show_in_center = False  # 10 秒後顯示在右上角
        detect_started = True  # 開始偵測

    # 在畫面上顯示姿勢名稱
    if show_in_center:
        cv2.putText(frame, pose_name, (frame.shape[1] // 2 - 100, frame.shape[0] // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3, cv2.LINE_AA)
    else:
        # 姿勢名稱顯示在右上角
        cv2.putText(frame, pose_name, (frame.shape[1] - 300, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    
    # 開始進行姿勢偵測
    if detect_started:
        # 將影像轉換成 RGB 格式
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 從影像中偵測姿勢
        results = pose.process(image_rgb)

        # 如果偵測到關鍵點，提取並進行預測
        if results.pose_landmarks:
            keypoints = []
            foot_detected = False
            # 提取所有 33 個點位
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints.extend([landmark.x, landmark.y])
                # 確認是否偵測到雙腳
                if idx in [mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]:
                    if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                        foot_detected = True

            # 如果提取的關鍵點數量不符合要求，跳過本次迭代
            if len(keypoints) != 66 or not foot_detected:
                cv2.putText(frame, 'Pose Not Detected Properly', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not foot_detected:
                    cv2.putText(frame, 'Feet Not Detected', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Press 'q' to exit", (frame.shape[1] - 300, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Ensure head and feet are visible in the frame", (30, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.imshow('Camera Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # 將數據轉換為 numpy 格式並調整為模型所需的輸入形狀
            keypoints_array = np.array(keypoints, dtype=np.float32).reshape(1, 33, 2, 1)

            # 使用模型進行預測
            try:
                prediction = model.predict(keypoints_array)
                score = 3
                total_score = int(prediction[0] * 33 * score)
                score_percentage = int((total_score / (33 * 3)) * 100)

                # 根據得分百分比判斷狀態
                if score_percentage < 60:
                    status = "bad"
                    color = (0, 0, 255)  # 紅色表示差
                elif 60 <= score_percentage < 90:
                    status = "good"
                    color = (0, 255, 255)  # 黃色表示普通
                else:
                    status = "great"
                    color = (0, 255, 0)  # 綠色表示優秀

                # 顯示狀態和分數
                cv2.putText(frame, f'Status: {status}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f'Score: {total_score}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # 顯示垂直條形圖 (進度條)
                bar_x = frame.shape[1] - 100  # 進度條的 X 軸位置，放在畫面右側
                bar_height = int(400 * score_percentage / 100)  # 根據得分比例調整條形圖的高度
                bar_y_start = frame.shape[0] - 50  # 條形圖從畫面底部開始向上顯示
                bar_y_end = bar_y_start - bar_height  # 根據得分計算條形圖的終點位置
                bar_color = (0, 255, 0) if score_percentage >= 90 else (0, 255, 255) if score_percentage >= 60 else (0, 0, 255)

                # 繪製垂直條形圖
                cv2.rectangle(frame, (bar_x, bar_y_start), (bar_x + 40, bar_y_end), bar_color, -1)
                cv2.putText(frame, f'{score_percentage}%', (bar_x - 30, bar_y_end - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, bar_color, 2)

            except Exception as e:
                print(f"預測時出錯: {e}")
                cv2.putText(frame, 'Prediction Error', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 在畫面上標示關鍵點（跳過臉部的點位）    
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if idx > 10:  # 跳過臉部點位
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # 在畫面右下角添加退出提示
    cv2.putText(frame, "Press 'q' to exit", (frame.shape[1] - 300, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 在畫面下方添加 "頭和腳需要在視窗中可見" 的提示
    cv2.putText(frame, "Ensure head and feet are visible in the frame", (30, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # 顯示影像
    cv2.imshow('Camera Feed', frame)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝像頭資源並關閉所有窗口
cap.release()
cv2.destroyAllWindows()
