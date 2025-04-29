
import os
import sys
import time
import threading
import subprocess
import cv2

from config import *
from face_detector import FaceDetector
from hand_detector import HandDetector
from state_detector import StateDetector
from data_collector import DataCollector
from model_trainer import ModelTrainer
from gui import FacialStateGUI

class FacialStateDetectionApp:
    def __init__(self):
        self.gui = FacialStateGUI()
        self.register_event_handlers()
        
        self.face_detector = FaceDetector()
        self.hand_detector = HandDetector()
        self.state_detector = StateDetector()
        self.data_collector = DataCollector()
        self.model_trainer = ModelTrainer()
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            self.webcam_available = False
            self.gui.show_webcam_error()
        else:
            self.webcam_available = True
            print("웹캠 초기화 성공")
        
        self.is_running = False
        self.current_frame = None
        self.processed_frame = None
        
        self.video_playing = False
        self.video_process = None
        
        model_loaded = self.model_trainer.load_model()
        if model_loaded:
            self.state_detector.model = self.model_trainer.model
            self.state_detector.scaler = self.model_trainer.scaler
            self.gui.update_model_status(True)
        else:
            self.gui.update_model_status(False)

    def register_event_handlers(self):
        self.gui.on_start_stop = self.toggle_start_stop
        self.gui.on_collection_toggle = self.toggle_collection
        self.gui.on_analysis_toggle = self.toggle_analysis
        self.gui.on_train_model = self.start_training
        self.gui.on_capture_screenshot = self.capture_screenshot
        self.gui.on_state_selected = self.on_state_selected
        self.gui.on_max_samples_selected = self.on_max_samples_selected
        self.gui.on_open_data_folder = self.open_data_folder
        self.gui.on_closing = self.on_closing

    def toggle_start_stop(self):
        if self.is_running:
            self.is_running = False
            self.gui.set_start_stop_button_state(False)
            self.stop_video()
            self.gui.update_status("중지됨")
        else:
            if not self.webcam_available:
                self.gui.update_status("웹캠을 사용할 수 없습니다.", warning=True)
                return
            
            self.is_running = True
            self.gui.set_start_stop_button_state(True)
            self.play_video()
            self.gui.update_status("실행 중...")
            self.update_webcam()
            
            if not self.state_detector.model:
                self.gui.update_status("모델이 없습니다. 먼저 모델을 학습하거나 로드하세요.", warning=True)

    def play_video(self):
        try:
            gamma_path = os.path.join(os.getcwd(), "gamma.mp4")
            if os.path.exists(gamma_path):
                self.video_process = subprocess.Popen(
                    ['start', '', gamma_path], shell=True
                )
                self.video_playing = True
                print("🎬 gamma.mp4 재생 시작")
            else:
                print("gamma.mp4 파일을 찾을 수 없습니다.")
        except Exception as e:
            print(f"비디오 재생 오류: {e}")

    def stop_video(self):
        if self.video_process and self.video_playing:
            try:
                self.video_process.terminate()
                self.video_playing = False
                print("🎬 gamma.mp4 재생 중단")
            except Exception as e:
                print(f"비디오 종료 오류: {e}")

    def toggle_collection(self):
        if self.data_collector.is_collecting:
            self.data_collector.stop_collection()
            self.gui.set_collection_button_state(False)
            self.gui.update_status("데이터 수집 중지됨")
        else:
            state_code = int(self.gui.state_combo.get()[0])
            self.data_collector.start_collection(state_code)
            self.gui.set_collection_button_state(True)
            self.gui.update_sample_count(0)
            self.gui.update_status(f"{STATE_KOREAN[state_code]} 상태 데이터 수집 시작")

    def toggle_analysis(self):
        if self.state_detector.is_analyzing:
            self.state_detector.stop_analysis()
            self.gui.set_analysis_button_state(False)
            self.gui.update_status("상태 감지 중지됨")
        else:
            self.state_detector.start_analysis()
            self.gui.set_analysis_button_state(True)
            self.gui.update_status("상태 감지 시작됨")

    def start_training(self):
        if os.path.exists(CSV_FILE):
            self.gui.set_training_button_state(True)
            self.gui.update_status("모델 학습 시작...")
            threading.Thread(target=self.train_model_thread).start()
        else:
            self.gui.update_status("학습 데이터가 없습니다.", warning=True)

    def train_model_thread(self):
        try:
            success, accuracy = self.model_trainer.train_model()
            if success:
                self.state_detector.model = self.model_trainer.model
                self.state_detector.scaler = self.model_trainer.scaler
                self.gui.root.after(0, lambda: self.gui.update_status(f"모델 학습 완료! 정확도: {accuracy:.2f}"))
                self.gui.root.after(0, lambda: self.gui.update_model_status(True))
            else:
                self.gui.root.after(0, lambda: self.gui.update_status("모델 학습 실패!", warning=True))
        except Exception as e:
            print(f"모델 학습 중 오류: {e}")
        finally:
            self.gui.root.after(0, lambda: self.gui.set_training_button_state(False))

    def capture_screenshot(self):
        if self.processed_frame is not None:
            filename = self.data_collector.capture_screenshot(self.processed_frame)
            if filename:
                self.gui.update_status(f"스크린샷 저장됨: {filename}")
            else:
                self.gui.update_status("스크린샷 저장 실패", warning=True)
        else:
            self.gui.update_status("캡처할 화면이 없습니다.", warning=True)

    def on_state_selected(self, state_code):
        self.data_collector.current_state = state_code

    def on_max_samples_selected(self, max_samples):
        self.data_collector.set_max_samples(max_samples)

    def open_data_folder(self):
        try:
            abs_path = os.path.abspath(DATA_FOLDER)
            if os.path.exists(abs_path):
                if os.name == 'nt':
                    os.startfile(abs_path)
                else:
                    import subprocess
                    subprocess.run(['xdg-open', abs_path])
                self.gui.update_status(f"데이터 폴더 열기: {abs_path}")
            else:
                self.gui.update_status("데이터 폴더가 존재하지 않습니다.", warning=True)
        except Exception as e:
            self.gui.update_status(f"폴더 열기 오류: {e}", warning=True)

    def update_webcam(self):
        if not self.is_running or not self.webcam_available:
            return
        try:
            ret, self.current_frame = self.cap.read()
            if not ret:
                self.gui.update_status("웹캠에서 프레임을 읽을 수 없습니다.", warning=True)
                self.is_running = False
                self.gui.set_start_stop_button_state(False)
                return
            self.current_frame = cv2.flip(self.current_frame, 1)

            face_detected, frame_with_face, face_data = self.face_detector.detect_face(self.current_frame)
            self.gui.update_face_status(face_detected)

            hand_detected, frame_with_face_and_hand, hand_data = self.hand_detector.detect_hands(frame_with_face)
            self.gui.update_hand_status(hand_detected)

            self.processed_frame, detected_state, state_info = self.state_detector.analyze_state(
                frame_with_face_and_hand, face_data, hand_data)

            self.gui.update_detected_state(detected_state, duration=state_info.get('duration', 0.0))
            self.gui.update_confidence(state_info.get('confidence', 0))

            self.gui.update_webcam_display(self.processed_frame)

            if self.data_collector.is_collecting:
                landmarks_data = {}
                landmarks_data.update(face_data)
                landmarks_data.update(hand_data)
                landmarks_data["yawn_duration"] = state_info.get('duration', 0) if detected_state == 2 else 0
                landmarks_data["thinking_duration"] = state_info.get('duration', 0) if detected_state == 3 else 0

                self.data_collector.save_to_csv(landmarks_data)
                self.gui.update_sample_count(self.data_collector.collection_count)

            self.gui.root.after(10, self.update_webcam)
        except Exception as e:
            print(f"웹캠 업데이트 중 오류: {e}")
            self.gui.update_status(f"웹캠 업데이트 중 오류: {e}", warning=True)
            self.gui.root.after(100, self.update_webcam)

    def on_closing(self):
        self.is_running = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'face_detector'):
            self.face_detector.close()
        if hasattr(self, 'hand_detector'):
            self.hand_detector.close()
        print("프로그램 종료")

    def run(self):
        self.gui.run()

if __name__ == "__main__":
    app = FacialStateDetectionApp()
    app.run()
