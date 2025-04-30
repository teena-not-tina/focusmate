"""
State detection module.
Handles emotional state detection based on facial and hand features.
"""
import cv2
import numpy as np
import time
from collections import deque
import pickle
from config import *

class StateDetector:
    def __init__(self):
        """상태 감지 클래스 초기화"""
        # 상태 감지 변수
        self.initial_face_size = 0
        self.initial_eyebrow_distance = 0
        self.initial_mouth_height = 0
        self.yawn_start_time = None
        self.is_yawning = False
        self.thinking_start_time = None
        self.is_thinking = False
        
        # 상태 유지 변수
        self.state_history = deque(maxlen=10)  # 최근 10개 상태 기록
        self.current_detected_state = 0
        self.detected_state_code = 0
        
        # 안정성 관련 변수
        self.last_temp_state = 0
        self.stable_start_time = None
        self.stable_detected_state = 0
        self.stable_duration_threshold = 3  # 지속시간 (초)

        # 머신러닝 모델
        self.model = None
        self.scaler = None
        self.is_analyzing = False
    
    def start_analysis(self):
        """상태 분석 시작"""
        self.is_analyzing = True
        self.initial_face_size = 0
        self.initial_eyebrow_distance = 0
        self.initial_mouth_height = 0
        self.stable_start_time = None
        self.last_temp_state = 0
        self.stable_detected_state = 0
        return True
    
    def stop_analysis(self):
        """상태 분석 중지"""
        self.is_analyzing = False
        return True
    
    def load_model(self, model_file):
        """저장된 모델 로드"""
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            print(f"모델 로드 성공: {model_file}")
            return True
        except Exception as e:
            print(f"모델 로드 중 오류: {e}")
            return False

    def analyze_state(self, frame, face_data, hand_data):
        """현재 프레임과 특징 데이터로 상태 분석"""
        if not self.is_analyzing:
            return frame, 0, {}

        h, w, _ = frame.shape

        # 결과 저장할 변수들
        temp_detected_state = 0  # 기본은 "만족"
        state_info = {
            'confidence': 0,
            'duration': 0
        }

        if not face_data or not hand_data:
            # 얼굴/손이 없으면 상태 리셋
            self.last_temp_state = 0
            self.stable_start_time = None
            return frame, self.stable_detected_state, state_info

        ## === 기존 상태 분석 로직 === ##
        # 턱과 손 겹침
        hand_jaw_overlap = self.check_hand_jaw_overlap(hand_data, face_data)
        if hand_jaw_overlap:
            if not self.is_thinking:
                self.is_thinking = True
                self.thinking_start_time = time.time()
            else:
                thinking_duration = time.time() - self.thinking_start_time
                state_info['duration'] = thinking_duration
                if thinking_duration >= THINKING_DURATION:
                    temp_detected_state = 3  # 고민 중
        else:
            self.is_thinking = False
            self.thinking_start_time = None

        # 초기값 설정
        if self.initial_face_size == 0 and face_data["face_metrics"]["face_size"] > 0:
            self.initial_face_size = face_data["face_metrics"]["face_size"]
        if self.initial_eyebrow_distance == 0 and face_data["face_metrics"]["eyebrow_distance"] > 0:
            self.initial_eyebrow_distance = face_data["face_metrics"]["eyebrow_distance"]
        if self.initial_mouth_height == 0 and face_data["face_metrics"]["mouth_height_pos"] > 0:
            self.initial_mouth_height = face_data["face_metrics"]["mouth_height_pos"]

        # 지루함 (얼굴 작아짐)
        if face_data["face_metrics"]["face_size"] > 0 and self.initial_face_size > 0:
            size_ratio = face_data["face_metrics"]["face_size"] / self.initial_face_size
            if size_ratio < BORED_THRESHOLD:
                temp_detected_state = 1  # 지루함

        # 피곤함 (하품)
        if face_data["face_metrics"]["mouth_open_height"] > 0:
            mouth_open = face_data["face_metrics"]["mouth_open_height"]
            if mouth_open > YAWN_THRESHOLD:
                if not self.is_yawning:
                    self.is_yawning = True
                    self.yawn_start_time = time.time()
                else:
                    yawn_duration = time.time() - self.yawn_start_time
                    state_info['duration'] = yawn_duration
                    if yawn_duration >= YAWN_DURATION:
                        temp_detected_state = 2  # 피곤함
            else:
                self.is_yawning = False
                self.yawn_start_time = None

        # 불만족 (미간 거리, 입 높이)
        if face_data["face_metrics"]["eyebrow_distance"] > 0 and self.initial_eyebrow_distance > 0:
            eyebrow_ratio = face_data["face_metrics"]["eyebrow_distance"] / self.initial_eyebrow_distance
            if eyebrow_ratio < EYEBROW_THRESHOLD:
                temp_detected_state = 4  # 불만족
        if face_data["face_metrics"]["mouth_height_pos"] > 0 and self.initial_mouth_height > 0:
            mouth_shift = self.initial_mouth_height - face_data["face_metrics"]["mouth_height_pos"]
            if mouth_shift > MOUTH_UP_THRESHOLD:
                temp_detected_state = 4  # 불만족

        # 모델 예측
        if self.model and self.scaler:
            try:
                landmarks_data = {}
                landmarks_data.update(face_data["face_metrics"])
                landmarks_data.update(hand_data["hand_metrics"])
                landmarks_data["yawn_duration"] = time.time() - self.yawn_start_time if self.is_yawning else 0
                landmarks_data["thinking_duration"] = time.time() - self.thinking_start_time if self.is_thinking else 0
                model_prediction = self.predict_state(landmarks_data)
                if model_prediction is not None:
                    temp_detected_state = model_prediction
            except Exception as e:
                print(f"상태 예측 중 오류: {e}")

        ## === 상태 지속시간 적용 === ##
        current_time = time.time()
        if temp_detected_state != self.last_temp_state:
            self.stable_start_time = current_time
            self.last_temp_state = temp_detected_state
        else:
            if self.stable_start_time and (current_time - self.stable_start_time) >= self.stable_duration_threshold:
                self.stable_detected_state = temp_detected_state

        # 기록
        self.state_history.append(self.stable_detected_state)
        self.detected_state_code = self.stable_detected_state

        return frame, self.stable_detected_state, state_info

    def prepare_features(self, landmarks_data):
        """머신러닝 특징 준비"""
        features = []
        for key in ['face_size', 'face_aspect_ratio', 'eye_aspect_ratio_left', 
                    'eye_aspect_ratio_right', 'mouth_open_height', 'mouth_height_pos',
                    'eyebrow_distance', 'head_pose_pitch', 'head_pose_yaw', 'head_pose_roll']:
            features.append(landmarks_data.get(key, 0))
        for key in ['hand_face_distance', 'hand_jaw_overlap', 'hand_detected']:
            features.append(landmarks_data.get(key, 0))
        features.append(landmarks_data.get("yawn_duration", 0))
        features.append(landmarks_data.get("thinking_duration", 0))
        return features

    def predict_state(self, landmarks_data):
        """현재 상태 예측"""
        if not self.model or not self.scaler:
            return None
        try:
            features = self.prepare_features(landmarks_data)
            features = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            state = self.model.predict(features_scaled)[0]
            return state
        except Exception as e:
            print(f"상태 예측 중 오류: {e}")
            return None

    def check_hand_jaw_overlap(self, hand_data, face_data):
        """손과 턱의 겹침 확인"""
        if hand_data["hand_metrics"]["hand_detected"] > 0:
            return hand_data["hand_metrics"]["hand_jaw_overlap"] > 0
        return False
