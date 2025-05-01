#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        # 얼굴 감지를 위한 Haar Cascade 분류기 로드
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 랜드마크 감지를 위한 모델 경로들
        # 68개 랜드마크 감지 모델은 이런 경로에 있다고 가정하고 작성했습니다. 
        # 실제 파일 위치에 맞게 경로를 조정해야 합니다.
        self.face_landmark_path = 'models/shape_predictor_68_face_landmarks.dat'
        
        # DLib 얼굴 랜드마크 감지기 (직접 설치 필요)
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(self.face_landmark_path)
            self.dlib_available = True
        except (ImportError, Exception) as e:
            print(f"dlib 또는 랜드마크 모델을 로드할 수 없습니다: {e}")
            self.dlib_available = False
    
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Haar Cascade로 얼굴 탐지
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # 얼굴이 감지되면
        if len(faces) > 0:
            # 가장 큰 얼굴 선택 (화면에 여러 얼굴이 있을 경우)
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_data['face_box'] = [x, y, w, h]
            
            # 얼굴 영역 사각형 표시
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # DLib 사용 가능한 경우 랜드마크 기반 메트릭 계산
            if self.dlib_available:
                try:
                    # DLib 감지기로 얼굴 영역 추출
                    dlib_rects = self.detector(gray, 0)
                    
                    if len(dlib_rects) > 0:
                        # 가장 큰 얼굴 선택
                        rect = max(dlib_rects, key=lambda r: (r.right()-r.left())*(r.bottom()-r.top()))
                        
                        # 랜드마크 추출
                        shape = self.predictor(gray, rect)
                        landmarks = []
                        
                        # 랜드마크 좌표 추출
                        for i in range(68):
                            x_lm, y_lm = shape.part(i).x, shape.part(i).y
                            landmarks.append((x_lm, y_lm))
                            cv2.circle(processed_frame, (x_lm, y_lm), 1, (0, 0, 255), -1)
                            
                        # 얼굴 메트릭 계산
                        face_metrics = self.calculate_face_metrics(landmarks)
                        face_data['face_metrics'] = face_metrics
                except Exception as e:
                    print(f"랜드마크 처리 중 오류: {e}")
            
            # 얼굴 감지 성공
            face_detected = True
        
        return face_detected, processed_frame, face_data
    
    def calculate_face_metrics(self, landmarks):
        """
        얼굴 랜드마크를 기반으로 다양한 얼굴 관련 메트릭을 계산
        
        Args:
            landmarks (list): (x, y) 좌표 쌍의 리스트
            
        Returns:
            dict: 계산된 얼굴 메트릭
        """
        metrics = {
            'mouth_open_height': 0,
            'head_pose_yaw': 0
        }
        
        # 랜드마크가 68개 포인트 모델을 기반으로 할 때:
        if len(landmarks) == 68:
            try:
                # 입 벌어짐 (mouth open height) 계산
                # 상단 입술 중앙점 (51)와 하단 입술 중앙점 (57) 사이의 거리
                upper_lip = landmarks[51]  # 상단 입술 중앙
                lower_lip = landmarks[57]  # 하단 입술 중앙
                
                # 입의 세로 길이 (상하 입술 거리)
                mouth_height = abs(lower_lip[1] - upper_lip[1])
                
                # 얼굴 세로 길이 (턱 끝 - 코 사이 거리)로 정규화
                nose_point = landmarks[33]  # 코 끝
                chin_point = landmarks[8]   # 턱 끝
                face_height = abs(chin_point[1] - nose_point[1])
                
                # 정규화된 입 벌어짐 (0.0 ~ 1.0)
                if face_height > 0:
                    normalized_mouth_height = mouth_height / face_height
                    metrics['mouth_open_height'] = normalized_mouth_height
                
                # 머리 회전 각도 (Yaw) 추정
                # 양쪽 귀 끝의 좌우 비대칭으로 추정 (왼쪽 귀와 오른쪽 귀의 x 좌표 차이)
                left_ear = landmarks[0]   # 왼쪽 얼굴 윤곽선
                right_ear = landmarks[16]  # 오른쪽 얼굴 윤곽선
                nose_bridge = landmarks[27]  # 코 브릿지 (코와 눈 사이)
                
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
        # 필요한 경우 리소스 정리
        pass