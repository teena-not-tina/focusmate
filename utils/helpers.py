# 도우미 함수

import os
import time
import logging

logger = logging.getLogger(__name__)

def ensure_dir(dir_path):
    """디렉토리가 없으면 생성"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"디렉토리 생성: {dir_path}")
    return dir_path

def get_timestamp():
    """현재 타임스탬프를 문자열로 반환"""
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def format_duration(seconds):
    """초 단위 시간을 mm:ss 형식으로 포맷팅"""
    if seconds < 60:
        return f"{seconds}초"
    else:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes}분 {remaining_seconds}초"

def truncate_text(text, max_length=100):
    """텍스트를 일정 길이로 자르고 생략 부호 추가"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def sanitize_filename(filename):
    """파일명에 사용할 수 없는 문자 제거"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename