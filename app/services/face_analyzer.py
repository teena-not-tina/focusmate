# Gemini API 관련 서비스

import cv2
import numpy as np
import logging
import insightface
from insightface.app import FaceAnalysis
from app.models.emotion import EMOTION_MAPPING

logger = logging.getLogger(__name__)

# 전역 변수 - 얼굴 분석기
_face_analyzer = None

def init_face_analyzer():
    """얼굴 분석기 초기화"""
    global _face_analyzer
    
    try:
        if _face_analyzer is None:
            logger.info("얼굴 분석기 초기화 중...")
            # CPU 모드로 분석기 초기화
            _face_analyzer = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
            _face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("얼굴 분석기 초기화 완료")
        return _face_analyzer
    except Exception as e:
        logger.error(f"얼굴 분석기 초기화 실패: {str(e)}")
        return None

def get_face_analyzer():
    """얼굴 분석기 반환"""
    global _face_analyzer
    if _face_analyzer is None:
        return init_face_analyzer()
    return _face_analyzer

def detect_emotion_from_image(image_data):
    """이미지에서 감정 상태 감지"""
    try:
        # 이미지 디코딩
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "이미지를 디코딩할 수 없습니다"}
            
        # 얼굴 분석기 가져오기
        face_analyzer = get_face_analyzer()
        if face_analyzer is None:
            return {"error": "얼굴 분석기가 초기화되지 않았습니다"}
            
        # 얼굴 감지
        faces = face_analyzer.get(img)
        
        if not faces:
            return {"error": "얼굴이 감지되지 않았습니다"}
            
        # 주요 얼굴 (가장 큰 얼굴)
        main_face = max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
        
        # 감정 상태 분석
        emotion_id = analyze_face_emotion(main_face)
        dominant_emotion = EMOTION_MAPPING.get(emotion_id, "neutral")
        
        logger.info(f"감정 상태 감지: {dominant_emotion}")
        
        return {
            "dominant_emotion": dominant_emotion,
            "confidence": 0.85  # 실제로는 더 정교한 계산 필요
        }
        
    except Exception as e:
        logger.error(f"감정 감지 오류: {str(e)}")
        return {"error": f"감정 감지 처리 중 오류 발생: {str(e)}"}

def analyze_face_emotion(face):
    """얼굴 특징점을 분석하여 감정 상태 추론"""
    try:
        # 이 부분은 실제 학습된 모델을 사용해야 하지만, 간소화된 버전 제공
        landmarks = face.landmark_2d_106
        if landmarks is None or len(landmarks) < 106:
            return 0  # 중립
            
        # 눈 개폐 상태 확인 (간소화된 버전)
        left_eye_height = np.linalg.norm(landmarks[33] - landmarks[40])
        left_eye_width = np.linalg.norm(landmarks[36] - landmarks[39])
        right_eye_height = np.linalg.norm(landmarks[87] - landmarks[94])
        right_eye_width = np.linalg.norm(landmarks[90] - landmarks[93])
        
        eye_aspect_ratio = ((left_eye_height / left_eye_width) + (right_eye_height / right_eye_width)) / 2
        
        # 입 상태 확인
        mouth_width = np.linalg.norm(landmarks[61] - landmarks[64])
        mouth_height = np.linalg.norm(landmarks[62] - landmarks[66])
        mouth_ratio = mouth_height / mouth_width
        
        # 눈썹 위치
        left_brow_height = landmarks[21][1]
        right_brow_height = landmarks[75][1]
        brow_height = (left_brow_height + right_brow_height) / 2
        
        # 감정 상태 결정 (간소화된 규칙 기반)
        if eye_aspect_ratio < 0.2:  # 눈을 거의 감은 경우
            return 4  # 피로
        elif mouth_ratio > 0.3:  # 입을 많이 벌린 경우
            return 2  # 흥미
        elif brow_height < 0.3:  # 눈썹이 내려간 경우
            return 3  # 불안
        elif eye_aspect_ratio > 0.28:  # 눈을 크게 뜬 경우
            return 1  # 집중
        else:
            return 0  # 중립
            
    except Exception as e:
        logger.error(f"얼굴 감정 분석 오류: {str(e)}")
        return 0  # 오류 발생 시 중립 반환