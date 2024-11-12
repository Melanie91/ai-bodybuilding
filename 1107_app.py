from flask import Flask, jsonify
import subprocess  # 用於啟動外部腳本
import os

app = Flask(__name__)

# 禁用 GPU 加速
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 首頁路由，帶有三個按鈕
@app.route('/', methods=['GET'])
def index():
    return '''
        <h1>Flask server is running</h1>
        <button onclick="window.location.href='/run_pose_detection1'">Run Pose Detection 1</button>
        <button onclick="window.location.href='/run_pose_detection2'">Run Pose Detection 2</button>
        <button onclick="window.location.href='/run_pose_detection3'">Run Pose Detection 3</button>
    '''

# 主要的姿勢偵測 API - 按鈕 1
@app.route('/run_pose_detection1', methods=['GET'])
def run_pose_detection1():
    return run_pose_script('C:/pose/VSCode/OpenCV/Mediapipe/photo_detect/detect/frontpose_detect.py')

# 主要的姿勢偵測 API - 按鈕 2
@app.route('/run_pose_detection2', methods=['GET'])
def run_pose_detection2():
    return run_pose_script('C:/pose/VSCode/OpenCV/Mediapipe/photo_detect/detect/front_double_biceps_detect.py')

# 主要的姿勢偵測 API - 按鈕 3
@app.route('/run_pose_detection3', methods=['GET'])
def run_pose_detection3():
    return run_pose_script('C:/pose/VSCode/OpenCV/Mediapipe/photo_detect/detect/front_lat_spread_detect.py')

# 共用函數來啟動外部腳本
def run_pose_script(script_path):
    try:
        # 使用 subprocess 啟動外部腳本，並隱藏命令行窗口
        subprocess.Popen(
            ['python', script_path],
            creationflags=subprocess.CREATE_NO_WINDOW  # 隱藏命令行窗口
        )
        return jsonify({"message": f"Pose detection script {script_path} started successfully!"})
    except Exception as e:
        return jsonify({"message": f"Error occurred: {e}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
