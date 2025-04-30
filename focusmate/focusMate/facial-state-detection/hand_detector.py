"""
Hand detection and analysis module.
Handles hand detection, landmark extraction, and hand-face interaction metrics.
"""

import cv2
import numpy as np
import mediapipe as mp

class HandDetector:
    def __init__(self):
        """손 감지 클래스 초기화"""
        # MediaPipe 초기화
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
    
    def detect_hands(self, frame):
        """프레임에서 손 감지 및 분석"""
        h, w, _ = frame.shape
        
        # RGB로 변환 (MediaPipe 요구사항)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            hand_results = self.hands.process(rgb_frame)
            hand_data = self.extract_hand_features(frame, hand_results)
            
            # 손 랜드마크 그리기
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
                return True, frame, hand_data
            else:
                return False, frame, hand_data
        except Exception as e:
            print(f"MediaPipe 손 처리 중 오류: {e}")
            return False, frame, self.create_empty_hand_data()
    
    def extract_hand_features(self, frame, hand_results):
        """손 특징 추출"""
        h, w, _ = frame.shape
        hand1_landmarks = []
        hand2_landmarks = []
        hand_metrics = {
            'hand_face_distance': 0,
            'hand_jaw_overlap': 0,
            'hand_detected': 0
        }
        
        try:
            if hand_results.multi_hand_landmarks:
                hand_metrics['hand_detected'] = len(hand_results.multi_hand_landmarks)  # 감지된 손 개수
                
                # 첫 번째 손 랜드마크
                if len(hand_results.multi_hand_landmarks) > 0:
                    hand1 = hand_results.multi_hand_landmarks[0]
                    for i in range(21):  # MediaPipe 손은 21개 랜드마크
                        lm = hand1.landmark[i]
                        x, y = int(lm.x * w), int(lm.y * h)
                        hand1_landmarks.extend([x, y])
                    
                    # 손가락 끝 좌표 (검지)
                    finger_tip = hand1.landmark[8]
                    finger_tip_coords = (int(finger_tip.x * w), int(finger_tip.y * h))
                    
                    # 턱 위치 (얼굴이 있다고 가정)
                    jaw_pos = (w // 2, int(h * 0.8))  # 기본값 (얼굴 없을 경우)
                    
                    # 손과 턱 사이 거리
                    hand_jaw_dist = np.sqrt((finger_tip_coords[0] - jaw_pos[0])**2 + 
                                          (finger_tip_coords[1] - jaw_pos[1])**2)
                    hand_metrics['hand_face_distance'] = hand_jaw_dist / w  # 정규화된 거리
                    
                    # 손과 턱 겹침 여부 (거리 기반 추정)
                    hand_metrics['hand_jaw_overlap'] = 1 if hand_metrics['hand_face_distance'] < 0.1 else 0
                
                # 두 번째 손 랜드마크
                if len(hand_results.multi_hand_landmarks) > 1:
                    hand2 = hand_results.multi_hand_landmarks[1]
                    for i in range(21):
                        lm = hand2.landmark[i]
                        x, y = int(lm.x * w), int(lm.y * h)
                        hand2_landmarks.extend([x, y])
                else:
                    # 두 번째 손이 없으면 0으로 채움
                    hand2_landmarks = [0] * 42  # 21개 랜드마크의 x, y 좌표
            else:
                # 손이 없으면 0으로 채움
                hand1_landmarks = [0] * 42  # 21개 랜드마크의 x, y 좌표
                hand2_landmarks = [0] * 42
        
        except Exception as e:
            print(f"손 특징 추출 중 오류: {e}")
            hand1_landmarks = [0] * 42
            hand2_landmarks = [0] * 42
        
        return {
            "hand1_landmarks": hand1_landmarks,
            "hand2_landmarks": hand2_landmarks,
            "hand_metrics": hand_metrics
        }
    
    def create_empty_hand_data(self):
        """빈 손 데이터 생성 (손 감지 실패 시)"""
        hand1_landmarks = [0] * 42  # 21개 랜드마크의 x, y 좌표
        hand2_landmarks = [0] * 42  # 21개 랜드마크의 x, y 좌표
        hand_metrics = {
            'hand_face_distance': 0,
            'hand_jaw_overlap': 0,
            'hand_detected': 0
        }
        return {
            "hand1_landmarks": hand1_landmarks,
            "hand2_landmarks": hand2_landmarks,
            "hand_metrics": hand_metrics
        }
    
    def check_hand_jaw_overlap(self, hand_data, face_data=None):
        """손과 턱의 겹침 확인"""
        # 손이 감지된 경우에만 확인
        if hand_data["hand_metrics"]["hand_detected"] > 0:
            # 손과 턱 겹침 여부
            return hand_data["hand_metrics"]["hand_jaw_overlap"] > 0
        return False
    
    def close(self):
        """리소스 해제"""
        self.hands.close()