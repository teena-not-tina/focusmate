# RAG 시스템(검색 증강 생성 서비스)

import os
import google.generativeai as genai
import logging
from app.models.emotion import emotion_info
from app.services.rag_service import retrieve_relevant_context

logger = logging.getLogger(__name__)

# 전역 변수 - Gemini 모델
_gemini_model = None

def init_gemini_model():
    """Gemini 모델 초기화"""
    global _gemini_model
    
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다")
            return None
            
        # Google Gemini API 설정
        genai.configure(api_key=api_key)
        
        # 모델 초기화
        _gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        logger.info("Gemini 모델 초기화 완료")
        return _gemini_model
    except Exception as e:
        logger.error(f"Gemini 모델 초기화 실패: {str(e)}")
        return None

def get_gemini_model():
    """Gemini 모델 반환"""
    global _gemini_model
    if _gemini_model is None:
        return init_gemini_model()
    return _gemini_model

def generate_response(emotion_data, query=None, model=None):
    """감정 데이터와 쿼리에 기반한 응답 생성"""
    try:
        # 감정 정보 가져오기
        dominant_emotion = emotion_data.get("dominant_emotion", "neutral")
        emotion_context = emotion_info.get(dominant_emotion, emotion_info["neutral"])
        
        if model is None:
            model = get_gemini_model()
            if model is None:
                return {"error": "Gemini 모델이 초기화되지 않았습니다", "response": "서비스를 일시적으로 사용할 수 없습니다."}
        
        # 기본 시스템 메시지
        system_prompt = """당신은 학습자를 돕는 AI 학습 조교입니다. 
        감정 상태에 맞는 학습 조언을 제공하고, 학습자가 더 효율적으로 공부할 수 있도록 도와주세요.
        답변은 친절하고 명확하게, 그리고 가능한 간결하게 제공해주세요."""
        
        # 쿼리가 있는 경우
        if query and query.strip():
            # 관련 컨텍스트 검색
            context = retrieve_relevant_context(query)
            
            # 프롬프트 구성
            prompt = f"""{system_prompt}

학습자의 감정 상태: {dominant_emotion}
감정 상태 정보:
{emotion_context}

관련 추가 정보:
{context}

학습자 질문: {query}

위 정보를 바탕으로 학습자의 질문에 답변하고, 감정 상태에 맞는 학습 조언을 제공해주세요.
"""
        # 쿼리가 없는 경우 (기본 감정 응답)
        else:
            prompt = f"""{system_prompt}

학습자의 감정 상태: {dominant_emotion}
감정 상태 정보:
{emotion_context}

이 감정 상태를 고려하여, 학습자에게 도움이 될 수 있는 적절한 학습 조언과 학습 전략을 제안해주세요.
응답은 친절하고 간결하게 작성해주세요.
"""
        
        # 응답 생성
        response = model.generate_content(prompt)
        
        if not response or not hasattr(response, "text"):
            return {"error": "응답을 생성할 수 없습니다", "response": "죄송합니다, 현재 응답을 생성할 수 없습니다."}
            
        return {"response": response.text.strip()}
        
    except Exception as e:
        logger.error(f"응답 생성 오류: {str(e)}")
        return {"error": str(e), "response": "죄송합니다, 응답 생성 중 오류가 발생했습니다."}