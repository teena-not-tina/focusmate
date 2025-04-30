from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import threading
import time
import os
import signal
import sys
import numpy as np  # 최적화: NumPy 복사 사용
from datetime import datetime
from queue import Queue
from threading import Lock

# Flask 서버 초기화
app = Flask(__name__)
CORS(app)

# 카메라 설정
FACE_CAM_ID = 0
POSE_CAM_ID = 1

# 저장 폴더
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'saved_faces')
os.makedirs(SAVE_DIR, exist_ok=True)

# 전역 변수
face_cap = cv2.VideoCapture(FACE_CAM_ID, cv2.CAP_DSHOW)
pose_cap = cv2.VideoCapture(POSE_CAM_ID, cv2.CAP_DSHOW)
face_frame = None
pose_frame = None
saving_faces = False
running = True
last_seen_time = time.time()

save_queue = Queue()
frame_lock = Lock()

# Ctrl+C 종료 처리
def signal_handler(sig, frame):
    global running
    print("\n[🔚 종료 요청 감지됨] 카메라 해제 중...")
    face_cap.release()
    pose_cap.release()
    running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# 얼굴 카메라 캡처 쓰레드
def capture_face_frame():
    global face_frame
    while running:
        success, frame = face_cap.read()
        if success:
            with frame_lock:
                face_frame = np.copy(frame)  # 최적화된 복제
        time.sleep(0.03)  # 약 30 FPS

# 포즈 카메라 캡처 쓰레드
def capture_pose_frame():
    global pose_frame
    while running:
        success, frame = pose_cap.read()
        if success:
            pose_frame = np.copy(frame)
        time.sleep(0.03)

# 저장 요청을 큐에 추가
def enqueue_frame_for_saving():
    global saving_faces, face_frame
    print("[💾] 저장 대기 쓰레드 시작")
    timer = 0
    while running:
        now = time.time()
        if saving_faces:
            if now - last_seen_time > 15:
                saving_faces = False
                print("[⛔ 저장 중단됨: heartbeat 15초 동안 없음]")
                continue
            if timer >= 5:
                with frame_lock:
                    if face_frame is not None:
                        save_queue.put((np.copy(face_frame), datetime.now()))
                        print("[📥] 프레임 저장 요청됨")
                timer = 0
            else:
                timer += 0.5
        else:
            print("[💬] 현재 saving_faces = False (저장 중단 상태)")
        time.sleep(0.5)

# 저장 전용 쓰레드 (JPEG 품질 설정)
def save_worker():
    print("[📂] 저장 작업 쓰레드 시작")
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # 품질 90 설정
    while running:
        if not save_queue.empty():
            frame, ts = save_queue.get()
            filename = f"face_{ts.strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            success, encoded_image = cv2.imencode('.jpg', frame, encode_param)
            if success:
                with open(filepath, 'wb') as f:
                    f.write(encoded_image.tobytes())
                print(f"[✅ 저장 완료] {filename}")
            else:
                print(f"[❌ 저장 실패] {filename}")
        time.sleep(0.01)

# 얼굴 스트림 (face_frame만 사용)
@app.route('/face_stream')
def face_stream():
    def generate():
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # 스트리밍 품질 설정
        while running:
            with frame_lock:
                if face_frame is None:
                    continue
                frame_to_send = np.copy(face_frame)
            success, buffer = cv2.imencode('.jpg', frame_to_send, encode_param)
            if not success:
                continue
            frame = buffer.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )
            time.sleep(0.04)  # 25 FPS로 부드럽게

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 포즈 스트림 (pose_frame만 사용)
@app.route('/pose_stream')
def pose_stream():
    def generate():
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        while running:
            if pose_frame is None:
                continue
            frame_to_send = np.copy(pose_frame)
            success, buffer = cv2.imencode('.jpg', frame_to_send, encode_param)
            if not success:
                continue
            frame = buffer.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )
            time.sleep(0.04)  # 25 FPS

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# API: 저장 시작
@app.route('/start_saving', methods=['GET', 'POST'])
def start_saving():
    global saving_faces
    saving_faces = True
    print("[▶️] 얼굴 저장 시작됨")
    return jsonify({'status': 'started'})

# API: 저장 중단
@app.route('/stop_saving', methods=['GET', 'POST'])
def stop_saving():
    global saving_faces
    saving_faces = False
    print("[⏹️] 얼굴 저장 중단됨")
    return jsonify({'status': 'stopped'})

# API: heartbeat
@app.route('/heartbeat')
def heartbeat():
    global last_seen_time
    last_seen_time = time.time()
    return jsonify({'status': 'ok'})

# 서버 실행
if __name__ == '__main__':
    threading.Thread(target=capture_face_frame, daemon=True).start()
    threading.Thread(target=capture_pose_frame, daemon=True).start()
    threading.Thread(target=enqueue_frame_for_saving, daemon=True).start()
    threading.Thread(target=save_worker, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
