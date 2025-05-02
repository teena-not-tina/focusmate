import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(app=None):
    """로깅 설정 초기화"""
    log_level = app.config.get('LOG_LEVEL', 'INFO') if app else 'INFO'
    log_dir = app.config.get('LOG_DIR', 'logs') if app else 'logs'
    
    # 로그 디렉토리 생성
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 로깅 기본 설정
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(
                os.path.join(log_dir, "app.log"),
                maxBytes=10485760,  # 10MB
                backupCount=5
            ),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("로깅 시스템 초기화 완료")
    return logger