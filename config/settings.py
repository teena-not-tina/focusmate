import os
from dotenv import load_dotenv

def load_config(config_name=None):
    """환경 변수 및 설정 로드"""
    # .env 파일 로드
    load_dotenv()
    
    # 환경 변수 설정
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    
    # 설정 값 반환
    return {
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY"),
        "DEBUG": os.environ.get("DEBUG", "True").lower() == "true",
        "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
        "LOG_DIR": "logs"
    }