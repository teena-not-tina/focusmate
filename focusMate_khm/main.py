#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from facial_detector_app import FacialDetectorApp

# 필요한 패키지 확인 및 설치 안내
def check_dependencies():
    missing_packages = []
    
    try:
        import psutil
    except ImportError:
        missing_packages.append("psutil")
    
    try:
        from selenium import webdriver
    except ImportError:
        missing_packages.append("selenium")
        
    try:
        import cv2
    except ImportError:
        missing_packages.append("opencv-python")
    
    try:
        import dlib
    except ImportError:
        missing_packages.append("dlib")
        
    if missing_packages:
        print(f"다음 패키지가 설치되어 있지 않습니다: {', '.join(missing_packages)}")
        print("다음 명령어로 필요한 패키지를 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\n패키지 설치 후 프로그램을 다시 실행하세요.")
        sys.exit(1)

if __name__ == "__main__":
    # 의존성 확인
    check_dependencies()
    
    # 애플리케이션 실행
    app = FacialDetectorApp()
    app.run()