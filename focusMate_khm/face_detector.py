#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self):
        # 얼굴 감지를 위한 Haar Cascade 분류기 로드
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # InsightFace 초기화
        try:
            self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.app.prepare(ctx_id=0)  # GPU 사용 (ctx_id=0), CPU만 사용하려면 ctx_id=-1
            self.insightface_available = True
        except Exception as e:
            print(f"InsightFace를 초기화할 수 없습니다: {e}")
            self.insightface_available = False
    
    def detect_face(self, frame):
        """
        프레임에서 얼굴을 감지하고 관련 메트릭 반환
        
        Returns:
            face_detected (bool): 얼굴 감지 여부
            processed_frame (numpy.ndarray): 시각화를 위해 처리된 프레임
            face_data (dict): 얼굴 위치, 메트릭 등 정보
        """
        # 결과 초기화
        face_detected = False
        face_data = {
            'face_box': [0, 0, 0, 0],
            'face_metrics': {
                'mouth_open_height': 0,
                'head_pose_yaw': 0
            }
        }
        
        # 프레임 복사 (원본 유지)
        processed_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # InsightFace로 얼굴 탐지 및 랜드마크 추출
        if self.insightface_available:
            try:
                faces = self.app.get(rgb_frame)
                if len(faces) > 0:
                    # 가장 큰 얼굴 선택
                    largest_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
                    x1, y1, x2, y2 = map(int, largest_face.bbox)
                    face_data['face_box'] = [x1, y1, x2 - x1, y2 - y1]
                    
                    # 얼굴 영역 사각형 표시
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 랜드마크 추출 및 시각화
                    landmarks = largest_face.landmark_2d_106
                    for (x_lm, y_lm) in landmarks:
                        cv2.circle(processed_frame, (int(x_lm), int(y_lm)), 1, (0, 0, 255), -1)
                    
                    # 얼굴 메트릭 계산
                    face_metrics = self.calculate_face_metrics(landmarks)
                    face_data['face_metrics'] = face_metrics
                    
                    # 얼굴 감지 성공
                    face_detected = True
            except Exception as e:
                print(f"InsightFace 처리 중 오류: {e}")
        
        return face_detected, processed_frame, face_data
    
    def calculate_face_metrics(self, landmarks):
        """
        얼굴 랜드마크를 기반으로 다양한 얼굴 관련 메트릭을 계산
        
        Args:
            landmarks (numpy.ndarray): (x, y) 좌표 배열
            
        Returns:
            dict: 계산된 얼굴 메트릭
        """
        metrics = {
            'mouth_open_height': 0,
            'head_pose_yaw': 0
        }
        
        # 랜드마크가 106개 포인트 모델을 기반으로 할 때:
        if len(landmarks) == 106:
            try:
                # 입 벌어짐 (mouth open height) 계산
                upper_lip = landmarks[62]  # 상단 입술 중앙
                lower_lip = landmarks[66]  # 하단 입술 중앙
                
                # 입의 세로 길이 (상하 입술 거리)
                mouth_height = abs(lower_lip[1] - upper_lip[1])
                
                # 얼굴 세로 길이 (턱 끝 - 코 사이 거리)로 정규화
                nose_point = landmarks[33]  # 코 끝
                chin_point = landmarks[7]   # 턱 끝
                face_height = abs(chin_point[1] - nose_point[1])
                
                # 정규화된 입 벌어짐 (0.0 ~ 1.0)
                if face_height > 0:
                    normalized_mouth_height = mouth_height / face_height
                    metrics['mouth_open_height'] = normalized_mouth_height
                
                # 머리 회전 각도 (Yaw) 추정
                left_ear = landmarks[0]   # 왼쪽 얼굴 윤곽선
                right_ear = landmarks[32]  # 오른쪽 얼굴 윤곽선
                nose_bridge = landmarks[52]  # 코 브릿지 (코와 눈 사이)
                
                # 왼쪽 귀와 오른쪽 귀 사이의 너비
                face_width = abs(right_ear[0] - left_ear[0])
                
                # 코 브릿지와 얼굴 중심점 사이의 편차
                face_center_x = left_ear[0] + (face_width / 2)
                nose_offset = (nose_bridge[0] - face_center_x) / (face_width / 2)
                
                # yaw 각도 추정 (간단한 근사치)
                metrics['head_pose_yaw'] = abs(nose_offset)
                
            except Exception as e:
                print(f"얼굴 메트릭 계산 중 오류: {e}")
        
        return metrics
    
    def close(self):
        """감지기 리소스 해제"""
        # InsightFace 리소스 정리
        pass