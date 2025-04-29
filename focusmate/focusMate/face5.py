import cv2
import numpy as np
import time
import threading
import os
import pandas as pd
from datetime import datetime
import mediapipe as mp
import csv
from collections import deque
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

# InsightFace 가져오기 (사용 가능한 경우)
try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    INSIGHTFACE_AVAILABLE = True
    print("InsightFace 가져오기 성공")
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("InsightFace를 사용할 수 없습니다. 'pip install insightface onnxruntime' 명령으로 설치하세요.")

# 상태 감지 조건 설정
BORED_THRESHOLD = 0.8        # 얼굴이 20% 이상 작아지면 지루함 (0.8배 이하)
YAWN_THRESHOLD = 0.05        # 입이 5cm 이상 벌어진 것으로 간주
YAWN_DURATION = 3.0          # 하품 감지 유지 시간 (초)
THINKING_DURATION = 5.0      # 고민 중 상태 유지 시간 (초)
EYEBROW_THRESHOLD = 0.9      # 미간 거리가 10% 이상 짧아지면 불만족 (0.9배 이하)
MOUTH_UP_THRESHOLD = 0.01    # 입 높이가 1cm 이상 올라가면 불만족

# 상태 코드 정의
STATE_LABELS = {
    0: 'satisfied',    # 만족 (기본 상태)
    1: 'bored',        # 지루함
    2: 'tired',        # 피곤함
    3: 'thinking',     # 고민 중
    4: 'dissatisfied'  # 불만족
}

# 한글 상태 이름
STATE_KOREAN = {
    0: '만족(기본)',
    1: '지루함',
    2: '피곤함',
    3: '고민 중',
    4: '불만족'
}

# 상태별 색상 (BGR 형식)
STATE_COLORS = {
    0: (46, 204, 113),   # 만족: 녹색
    1: (52, 152, 219),   # 지루함: 파란색
    2: (0, 0, 255),      # 피곤함: 빨간색
    3: (255, 165, 0),    # 고민 중: 주황색
    4: (142, 68, 173)    # 불만족: 보라색
}

# GUI 색상 (RGB 형식)
GUI_COLORS = {
    'bg_dark': '#2c3e50',      # 배경색 (어두운)
    'bg_mid': '#34495e',       # 배경색 (중간)
    'text': '#ecf0f1',         # 텍스트 기본색
    'highlight': '#3498db',    # 강조색
    'green': '#2ecc71',        # 녹색 (버튼 등)
    'red': '#e74c3c',          # 빨간색 (경고 등)
    'orange': '#e67e22',       # 주황색 (주의 등)
    'yellow': '#f1c40f',       # 노란색
    'purple': '#9b59b6'        # 보라색
}

# 데이터 수집 및 분석 클래스
class FacialDataCollector:
    def __init__(self, root=None):
        # GUI 루트 (외부에서 제공된 경우)
        self.external_root = root is not None
        self.root = root if root else tk.Tk()
        
        # 데이터 수집 관련 변수
        self.data_folder = "facial_data"
        self.model_folder = "models"
        self.csv_file = os.path.join(self.data_folder, "facial_landmarks.csv")
        self.model_file = os.path.join(self.model_folder, "state_detector_model.pkl")
        self.current_state = 0  # 0: 만족(기본), 1: 지루함, 2: 피곤함, 3: 고민 중, 4: 불만족
        self.is_collecting = False
        self.is_analyzing = False
        self.collection_count = 0
        self.max_samples = 100  # 각 상태별 최대 샘플 수
        
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
        self.current_detected_state = 'satisfied'
        self.detected_state_code = 0
        
        # 머신러닝 모델
        self.model = None
        self.scaler = None
        
        # 폴더 생성
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)
        
        # CSV 파일 초기화 (존재하지 않을 경우)
        self.initialize_csv()
        
        # 기존 모델 로드 (존재하는 경우)
        self.load_model()
        
        # MediaPipe 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
            
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # InsightFace 초기화
        self.insightface_available = False
        if INSIGHTFACE_AVAILABLE:
            try:
                self.face_app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106', 'recognition'])
                self.face_app.prepare(ctx_id=0, det_size=(640, 640))
                self.insightface_available = True
                print("InsightFace 초기화 성공")
            except Exception as e:
                print(f"InsightFace 초기화 중 오류: {e}")
                self.insightface_available = False
        
        # 웹캠 초기화
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("웹캠을 열 수 없습니다. 카메라가 연결되어 있는지 확인하세요.")
            self.webcam_available = False
        else:
            self.webcam_available = True
            print("웹캠 초기화 성공")
        
        # GUI 처리를 위한 변수
        self.is_running = False
        self.current_frame = None
        self.webcam_frame = None
        self.processed_frame = None
        self.tk_img = None
        
        # GUI 설정
        if not self.external_root:
            self.setup_gui()
    
    def setup_gui(self):
        """GUI 설정"""
        self.root.title("얼굴 및 손 랜드마크 수집 및 상태 추정")
        self.root.configure(bg=GUI_COLORS['bg_dark'])
        self.root.geometry("1200x700")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 메인 프레임
        main_frame = tk.Frame(self.root, bg=GUI_COLORS['bg_dark'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 왼쪽 패널 (웹캠 화면)
        webcam_panel = tk.Frame(main_frame, bg=GUI_COLORS['bg_mid'], 
                              highlightbackground=GUI_COLORS['highlight'], 
                              highlightthickness=2)
        webcam_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 웹캠 제목
        webcam_title = tk.Label(webcam_panel, text="실시간 웹캠 화면", 
                              font=("Helvetica", 14, "bold"), 
                              bg=GUI_COLORS['bg_mid'], fg=GUI_COLORS['text'])
        webcam_title.pack(pady=5)
        
        # 웹캠 표시 영역
        self.webcam_display = tk.Label(webcam_panel, bg="black")
        self.webcam_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 상태 표시줄
        status_frame = tk.Frame(webcam_panel, bg=GUI_COLORS['bg_mid'])
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = tk.Label(status_frame, text="준비됨", 
                                   font=("Helvetica", 10), 
                                   bg=GUI_COLORS['bg_mid'], fg=GUI_COLORS['text'])
        self.status_label.pack(side=tk.LEFT)
        
        # 오른쪽 패널 (컨트롤 패널)
        control_panel = tk.Frame(main_frame, bg=GUI_COLORS['bg_mid'], width=350)
        control_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        control_panel.pack_propagate(False)
        
        # 제목
        title_label = tk.Label(control_panel, text="얼굴 상태 모니터링 시스템", 
                             font=("Helvetica", 16, "bold"), 
                             bg=GUI_COLORS['bg_mid'], fg=GUI_COLORS['text'])
        title_label.pack(pady=10)
        
        # 상태 표시 섹션
        self.create_status_section(control_panel)
        
        # 구분선
        ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10, padx=20)
        
        # 데이터 수집 섹션
        self.create_data_collection_section(control_panel)
        
        # 구분선
        ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10, padx=20)
        
        # 상태 감지 섹션
        self.create_detection_section(control_panel)
        
        # 구분선
        ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10, padx=20)
        
        # 제어 버튼 섹션
        self.create_control_section(control_panel)
        
        # 키 바인딩
        self.root.bind('<Escape>', lambda e: self.on_closing())
        self.root.bind('s', lambda e: self.toggle_start_stop())
        self.root.bind('c', lambda e: self.toggle_collection())
        self.root.bind('a', lambda e: self.toggle_analysis())
        self.root.bind('t', lambda e: self.start_training())
        
        # InsightFace 사용 불가능하면 메시지 표시
        if not self.insightface_available:
            self.update_status("InsightFace를 사용할 수 없습니다. 'pip install insightface onnxruntime' 명령으로 설치하세요.")
        
        # 웹캠 사용 불가능하면 메시지 표시
        if not self.webcam_available:
            self.update_status("웹캠을 사용할 수 없습니다. 카메라 연결을 확인하세요.")
            error_msg = tk.Label(self.webcam_display, text="웹캠을 사용할 수 없습니다.\n카메라 연결을 확인하세요.",
                                fg="white", bg="black", font=("Helvetica", 16))
            error_msg.pack(expand=True)
    
    def create_status_section(self, parent):
        """상태 표시 섹션 생성"""
        status_frame = tk.LabelFrame(parent, text="현재 상태", font=("Helvetica", 12, "bold"),
                                   bg=GUI_COLORS['bg_mid'], fg=GUI_COLORS['text'])
        status_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # 상태 표시 그리드
        status_grid = tk.Frame(status_frame, bg=GUI_COLORS['bg_mid'])
        status_grid.pack(fill=tk.X, padx=10, pady=5)
        
        # 얼굴 감지 상태
        tk.Label(status_grid, text="얼굴 감지:", bg=GUI_COLORS['bg_mid'], fg=GUI_COLORS['text'],
               font=("Helvetica", 10)).grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.face_status_label = tk.Label(status_grid, text="감지되지 않음", bg=GUI_COLORS['bg_mid'],
                                       fg=GUI_COLORS['red'], font=("Helvetica", 10, "bold"))
        self.face_status_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=3)
        
        # 손 감지 상태
        tk.Label(status_grid, text="손 감지:", bg=GUI_COLORS['bg_mid'], fg=GUI_COLORS['text'],
               font=("Helvetica", 10)).grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.hand_status_label = tk.Label(status_grid, text="감지되지 않음", bg=GUI_COLORS['bg_mid'],
                                       fg=GUI_COLORS['red'], font=("Helvetica", 10, "bold"))
        self.hand_status_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=3)
        
        # 감지된 상태
        tk.Label(status_grid, text="감지된 상태:", bg=GUI_COLORS['bg_mid'], fg=GUI_COLORS['text'],
               font=("Helvetica", 10)).grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        self.detected_state_label = tk.Label(status_grid, text="만족(기본)", bg=GUI_COLORS['bg_mid'],
                                          fg=GUI_COLORS['green'], font=("Helvetica", 10, "bold"))
        self.detected_state_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=3)
        
        # 지속 시간
        tk.Label(status_grid, text="지속 시간:", bg=GUI_COLORS['bg_mid'], fg=GUI_COLORS['text'],
               font=("Helvetica", 10)).grid(row=3, column=0, sticky=tk.W, padx=5, pady=3)
        self.duration_label = tk.Label(status_grid, text="0.0초", bg=GUI_COLORS['bg_mid'],
                                     fg=GUI_COLORS['yellow'], font=("Helvetica", 10, "bold"))
        self.duration_label.grid(row=3, column=1, sticky=tk.W, padx=5, pady=3)
        
        # 미터 막대
        meter_frame = tk.Frame(status_frame, bg=GUI_COLORS['bg_mid'])
        meter_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 상태 신뢰도 표시
        tk.Label(meter_frame, text="상태 신뢰도:", bg=GUI_COLORS['bg_mid'], fg=GUI_COLORS['text'],
               font=("Helvetica", 10)).pack(side=tk.LEFT, padx=5)
        
        # 진행 막대 (실시간 상태 신뢰도)
        self.confidence_bar = ttk.Progressbar(meter_frame, orient="horizontal", length=200, mode="determinate")
        self.confidence_bar.pack(side=tk.LEFT, padx=5)
        self.confidence_bar["value"] = 0
        
        # 신뢰도 퍼센트
        self.confidence_label = tk.Label(meter_frame, text="0%", bg=GUI_COLORS['bg_mid'],
                                       fg=GUI_COLORS['text'], font=("Helvetica", 10))
        self.confidence_label.pack(side=tk.LEFT, padx=5)
    
    def create_data_collection_section(self, parent):
        """데이터 수집 섹션 생성"""
        collection_frame = tk.LabelFrame(parent, text="데이터 수집", font=("Helvetica", 12, "bold"),
                                       bg=GUI_COLORS['bg_mid'], fg=GUI_COLORS['text'])
        collection_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # 상태 선택 프레임
        state_frame = tk.Frame(collection_frame, bg=GUI_COLORS['bg_mid'])
        state_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(state_frame, text="수집할 상태:", bg=GUI_COLORS['bg_mid'], fg=GUI_COLORS['text'],
               font=("Helvetica", 10)).pack(side=tk.LEFT, padx=5)
        
        # 상태 선택 콤보박스
        self.state_combo = ttk.Combobox(state_frame, values=[
            "0: 만족(기본)", "1: 지루함", "2: 피곤함", "3: 고민 중", "4: 불만족"
        ], width=15)
        self.state_combo.current(0)
        self.state_combo.pack(side=tk.LEFT, padx=5)
        self.state_combo.bind("<<ComboboxSelected>>", self.on_state_selected)
        
        # 수집 시작/중지 버튼
        self.collection_btn = tk.Button(collection_frame, text="수집 시작", width=12,
                                      bg=GUI_COLORS['green'], fg="white", font=("Helvetica", 10, "bold"),
                                      command=self.toggle_collection)
        self.collection_btn.pack(pady=5)
        
        # 샘플 카운터 프레임
        sample_frame = tk.Frame(collection_frame, bg=GUI_COLORS['bg_mid'])
        sample_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(sample_frame, text="수집된 샘플:", bg=GUI_COLORS['bg_mid'], fg=GUI_COLORS['text'],
               font=("Helvetica", 10)).pack(side=tk.LEFT, padx=5)
        
        self.sample_count_label = tk.Label(sample_frame, text="0", bg=GUI_COLORS['bg_mid'],
                                         fg=GUI_COLORS['highlight'], font=("Helvetica", 10, "bold"))
        self.sample_count_label.pack(side=tk.LEFT, padx=5)
        
        # 최대 샘플 수 설정
        tk.Label(sample_frame, text="최대:", bg=GUI_COLORS['bg_mid'], fg=GUI_COLORS['text'],
               font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(20, 5))
        
        # 최대 샘플 수 콤보박스
        self.max_samples_combo = ttk.Combobox(sample_frame, values=[
            "50", "100", "200", "500", "1000"
        ], width=5)
        self.max_samples_combo.current(1)  # 기본값 100
        self.max_samples_combo.pack(side=tk.LEFT, padx=5)
        self.max_samples_combo.bind("<<ComboboxSelected>>", self.on_max_samples_selected)
    
    def create_detection_section(self, parent):
        """상태 감지 섹션 생성"""
        detection_frame = tk.LabelFrame(parent, text="상태 감지", font=("Helvetica", 12, "bold"),
                                      bg=GUI_COLORS['bg_mid'], fg=GUI_COLORS['text'])
        detection_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # 상태 감지 시작/중지 버튼
        self.analysis_btn = tk.Button(detection_frame, text="감지 시작", width=12,
                                   bg=GUI_COLORS['highlight'], fg="white", font=("Helvetica", 10, "bold"),
                                   command=self.toggle_analysis)
        self.analysis_btn.pack(pady=5)
        
        # 모델 학습 버튼
        model_frame = tk.Frame(detection_frame, bg=GUI_COLORS['bg_mid'])
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.training_btn = tk.Button(model_frame, text="모델 학습", width=12,
                                    bg=GUI_COLORS['orange'], fg="white", font=("Helvetica", 10, "bold"),
                                    command=self.start_training)
        self.training_btn.pack(side=tk.LEFT, padx=5)
        
        # 모델 상태
        self.model_status_label = tk.Label(model_frame, text="모델 없음", bg=GUI_COLORS['bg_mid'],
                                        fg=GUI_COLORS['red'], font=("Helvetica", 10))
        if self.model:
            self.model_status_label.config(text="모델 로드됨", fg=GUI_COLORS['green'])
        self.model_status_label.pack(side=tk.LEFT, padx=5)
        
        # 캡처 폴더 열기 버튼
        folder_btn = tk.Button(detection_frame, text="데이터 폴더 열기", width=15,
                             bg=GUI_COLORS['purple'], fg="white", font=("Helvetica", 10),
                             command=self.open_data_folder)
        folder_btn.pack(pady=5)
    
    def create_control_section(self, parent):
        """제어 버튼 섹션 생성"""
        control_frame = tk.Frame(parent, bg=GUI_COLORS['bg_mid'])
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # 시작/중지 버튼
        self.start_stop_btn = tk.Button(control_frame, text="시작", width=12, height=2,
                                     bg=GUI_COLORS['green'], fg="white", font=("Helvetica", 12, "bold"),
                                     command=self.toggle_start_stop)
        self.start_stop_btn.pack(side=tk.LEFT, padx=5)
        
        # 캡처 버튼
        self.capture_btn = tk.Button(control_frame, text="스크린샷", width=12, height=2,
                                  bg=GUI_COLORS['highlight'], fg="white", font=("Helvetica", 12, "bold"),
                                  command=self.capture_screenshot)
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        # 종료 버튼
        exit_btn = tk.Button(control_frame, text="종료", width=12, height=2,
                           bg=GUI_COLORS['red'], fg="white", font=("Helvetica", 12, "bold"),
                           command=self.on_closing)
        exit_btn.pack(side=tk.LEFT, padx=5)
        
        # 도움말 섹션
        help_frame = tk.Frame(parent, bg=GUI_COLORS['bg_mid'])
        help_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        # 도움말 텍스트
        help_text = tk.Text(help_frame, bg=GUI_COLORS['bg_dark'], fg=GUI_COLORS['text'],
                          font=("Helvetica", 9), height=7, wrap=tk.WORD)
        help_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 도움말 내용
        help_content = """단축키:
S: 시작/중지
C: 데이터 수집 시작/중지
A: 상태 감지 시작/중지
T: 모델 학습
ESC: 종료

상태 감지 조건:
• 지루함: 얼굴이 20% 이상 작아짐
• 피곤함: 입을 5cm 이상 벌려 3초 이상 유지
• 고민 중: 턱과 손이 겹쳐있고 5초 이상 유지
• 만족: 기본 상태
• 불만족: 미간 거리 10% 이상 짧아짐 또는 입이 1cm 이상 올라감"""
        
        help_text.insert(tk.END, help_content)
        help_text.config(state=tk.DISABLED)
    
    def on_state_selected(self, event):
        """상태 선택 콤보박스 이벤트 처리"""
        selected = self.state_combo.get()
        self.current_state = int(selected[0])  # 첫 글자(숫자)만 추출
        print(f"수집할 상태 선택: {selected}")
    
    def on_max_samples_selected(self, event):
        """최대 샘플 수 콤보박스 이벤트 처리"""
        try:
            self.max_samples = int(self.max_samples_combo.get())
            print(f"최대 샘플 수 설정: {self.max_samples}개")
        except ValueError:
            self.max_samples = 100
            print("잘못된 샘플 수 입력. 기본값 100으로 설정")
    
    def toggle_start_stop(self):
        """시작/중지 버튼 이벤트 처리"""
        if self.is_running:
            self.is_running = False
            self.start_stop_btn.config(text="시작", bg=GUI_COLORS['green'])
            self.update_status("중지됨")
        else:
            if not self.webcam_available:
                self.update_status("웹캠을 사용할 수 없습니다.")
                return
                
            self.is_running = True
            self.start_stop_btn.config(text="중지", bg=GUI_COLORS['red'])
            self.update_status("실행 중...")
            self.update_webcam()
            
            # 모델이 없는 경우 경고
            if not self.model:
                self.update_status("모델이 없습니다. 먼저 모델을 학습하거나 로드하세요.", warning=True)
    
    def start_training(self):
        """모델 학습 버튼 이벤트 처리"""
        if os.path.exists(self.csv_file):
            self.training_btn.config(text="학습 중...", state=tk.DISABLED)
            self.update_status("모델 학습 시작... 이 작업은 시간이 걸릴 수 있습니다.")
            threading.Thread(target=self.train_model_thread).start()
        else:
            self.update_status("학습 데이터가 없습니다. 먼저 데이터를 수집하세요.", warning=True)
    
    def train_model_thread(self):
        """별도 스레드에서 모델 학습"""
        try:
            success = self.train_model()
            if success:
                self.root.after(0, lambda: self.update_status("모델 학습 완료! 이제 상태 감지를 시작할 수 있습니다."))
                self.root.after(0, lambda: self.model_status_label.config(text="모델 로드됨", fg=GUI_COLORS['green']))
            else:
                self.root.after(0, lambda: self.update_status("모델 학습 실패!", warning=True))
                self.root.after(0, lambda: self.model_status_label.config(text="모델 실패", fg=GUI_COLORS['red']))
        except Exception as e:
            print(f"모델 학습 중 오류: {e}")
            self.root.after(0, lambda: self.update_status(f"모델 학습 중 오류: {e}", warning=True))
        finally:
            self.root.after(0, lambda: self.training_btn.config(text="모델 학습", state=tk.NORMAL))
    
    def capture_screenshot(self):
        """현재 화면 스크린샷 저장"""
        if self.processed_frame is not None:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_folder = "screenshots"
                os.makedirs(screenshot_folder, exist_ok=True)
                filename = f"{screenshot_folder}/screenshot_{timestamp}.jpg"
                
                cv2.imwrite(filename, self.processed_frame)
                self.update_status(f"스크린샷 저장됨: {filename}")
            except Exception as e:
                self.update_status(f"스크린샷 저장 중 오류: {e}", warning=True)
                print(f"스크린샷 저장 중 오류: {e}")
        else:
            self.update_status("캡처할 화면이 없습니다.", warning=True)
    
    def open_data_folder(self):
        """데이터 폴더 열기"""
        try:
            # 운영체제별 폴더 열기 명령
            abs_path = os.path.abspath(self.data_folder)
            if os.path.exists(abs_path):
                if os.name == 'nt':  # Windows
                    os.startfile(abs_path)
                else:  # macOS, Linux
                    import subprocess
                    if sys.platform == 'darwin':  # macOS
                        subprocess.run(['open', abs_path])
                    else:  # Linux
                        subprocess.run(['xdg-open', abs_path])
                self.update_status(f"데이터 폴더 열기: {abs_path}")
            else:
                self.update_status(f"데이터 폴더가 존재하지 않습니다: {abs_path}", warning=True)
        except Exception as e:
            self.update_status(f"폴더 열기 중 오류: {e}", warning=True)
            print(f"폴더 열기 중 오류: {e}")
    
    def update_status(self, message, warning=False):
        """상태 표시줄 업데이트"""
        self.status_label.config(text=message, fg=GUI_COLORS['red'] if warning else GUI_COLORS['text'])
        print(message)
    
    def update_confidence(self, confidence=0):
        """신뢰도 막대 업데이트"""
        self.confidence_bar["value"] = confidence
        self.confidence_label.config(text=f"{confidence}%")
    
    def update_webcam(self):
        """웹캠 화면 업데이트"""
        if not self.is_running or not self.webcam_available:
            return
        
        try:
            ret, self.current_frame = self.cap.read()
            if not ret:
                self.update_status("웹캠에서 프레임을 읽을 수 없습니다.", warning=True)
                self.is_running = False
                self.start_stop_btn.config(text="시작", bg=GUI_COLORS['green'])
                return
            
            # 프레임 처리
            self.processed_frame, landmarks_data = self.process_frame(self.current_frame)
            
            # GUI 업데이트
            self.update_gui_with_frame(self.processed_frame)
            
            # 다음 프레임 처리를 위한 재귀 호출
            self.root.after(10, self.update_webcam)
            
        except Exception as e:
            print(f"웹캠 업데이트 중 오류: {e}")
            self.update_status(f"웹캠 업데이트 중 오류: {e}", warning=True)
            # 오류 발생해도 계속 실행 시도
            self.root.after(100, self.update_webcam)
    
    def process_frame(self, frame):
        """프레임 처리 및 분석"""
        # 화면 뒤집기 (거울 효과)
        frame = cv2.flip(frame, 1)
        
        # 랜드마크 수집 및 분석
        landmarks_data = {}
        
        # RGB로 변환 (MediaPipe 요구사항)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 얼굴 감지 및 랜드마크 추출
        face_data = None
        if self.insightface_available:
            try:
                # InsightFace로 얼굴 감지
                faces = self.face_app.get(rgb_frame)
                
                if len(faces) > 0:
                    face = faces[0]  # 첫 번째 얼굴
                    face_data = self.extract_facial_features_insightface(frame, face)
                    
                    # 얼굴 감지됨
                    self.face_status_label.config(text="감지됨", fg=GUI_COLORS['green'])
                    
                    # 얼굴 상자 그리기
                    box = face.bbox.astype(np.int32)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    
                    # 랜드마크 그리기
                    if hasattr(face, 'landmark_2d_106'):
                        landmarks = face.landmark_2d_106.astype(np.int32)
                        for i in range(landmarks.shape[0]):
                            cv2.circle(frame, (landmarks[i][0], landmarks[i][1]), 1, (0, 0, 255), 2)
                else:
                    self.face_status_label.config(text="감지되지 않음", fg=GUI_COLORS['red'])
            except Exception as e:
                print(f"InsightFace 처리 중 오류: {e}")
        
        # InsightFace를 사용할 수 없거나 얼굴 감지 실패시 MediaPipe 사용
        if face_data is None:
            try:
                face_results = self.face_mesh.process(rgb_frame)
                if face_results.multi_face_landmarks:
                    face_data = self.extract_facial_features_mediapipe(frame, face_results)
                    
                    # 얼굴 감지됨
                    self.face_status_label.config(text="감지됨", fg=GUI_COLORS['green'])
                    
                    # 랜드마크 그리기
                    self.mp_drawing.draw_landmarks(
                        frame,
                        face_results.multi_face_landmarks[0],
                        self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None)
                else:
                    self.face_status_label.config(text="감지되지 않음", fg=GUI_COLORS['red'])
            except Exception as e:
                print(f"MediaPipe 얼굴 처리 중 오류: {e}")
        
        # 손 감지 및 랜드마크 추출
        hand_data = None
        try:
            hand_results = self.hands.process(rgb_frame)
            hand_data = self.extract_hand_features(frame, hand_results)
            
            # 손 랜드마크 그리기
            if hand_results.multi_hand_landmarks:
                # 손 감지됨
                self.hand_status_label.config(text="감지됨", fg=GUI_COLORS['green'])
                
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
            else:
                self.hand_status_label.config(text="감지되지 않음", fg=GUI_COLORS['red'])
        except Exception as e:
            print(f"MediaPipe 손 처리 중 오류: {e}")
            hand_data = {
                "hand1_landmarks": [0] * 42,
                "hand2_landmarks": [0] * 42,
                "hand_metrics": {
                    'hand_face_distance': 0,
                    'hand_jaw_overlap': 0,
                    'hand_detected': 0
                }
            }
        
        # 손과 턱의 겹침 확인
        if face_data and hand_data:
            overlap = self.check_hand_jaw_overlap(hand_data, face_data)
            
            # 고민 중 상태 확인 (턱 괴기)
            if overlap:
                if not self.is_thinking:
                    self.is_thinking = True
                    self.thinking_start_time = time.time()
                    cv2.putText(frame, "턱 괴기 감지", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    thinking_duration = time.time() - self.thinking_start_time
                    self.duration_label.config(text=f"{thinking_duration:.1f}초")
                    cv2.putText(frame, f"턱 괴기 시간: {thinking_duration:.1f}초", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if thinking_duration >= THINKING_DURATION:
                        cv2.putText(frame, "고민 중 상태 감지!", (10, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        if self.is_analyzing:
                            self.detected_states.append('thinking')
                            self.detected_state_code = 3  # 고민 중
                            self.detected_state_label.config(text="고민 중", fg=GUI_COLORS['orange'])
            else:
                self.is_thinking = False
                self.thinking_start_time = None
        
        # 데이터 병합 및 상태 지속 시간 추가
        if face_data and hand_data:
            # 얼굴 랜드마크
            landmarks_data["face_landmarks"] = face_data["face_landmarks"]
            
            # 얼굴 메트릭
            for key, value in face_data["face_metrics"].items():
                landmarks_data[key] = value
            
            # 손 랜드마크
            landmarks_data["hand1_landmarks"] = hand_data["hand1_landmarks"]
            landmarks_data["hand2_landmarks"] = hand_data["hand2_landmarks"]
            
            # 손 메트릭
            for key, value in hand_data["hand_metrics"].items():
                landmarks_data[key] = value
            
            # 상태 지속 시간
            landmarks_data["yawn_duration"] = time.time() - self.yawn_start_time if self.is_yawning else 0
            landmarks_data["thinking_duration"] = time.time() - self.thinking_start_time if self.is_thinking else 0
            
            # 상태 분석
            if self.is_analyzing:
                # 초기값 설정 (첫 감지시)
                if self.initial_face_size == 0 and face_data["face_metrics"]["face_size"] > 0:
                    self.initial_face_size = face_data["face_metrics"]["face_size"]
                    print(f"초기 얼굴 크기 설정: {self.initial_face_size:.4f}")
                
                if self.initial_eyebrow_distance == 0 and face_data["face_metrics"]["eyebrow_distance"] > 0:
                    self.initial_eyebrow_distance = face_data["face_metrics"]["eyebrow_distance"]
                    print(f"초기 미간 거리 설정: {self.initial_eyebrow_distance:.4f}")
                
                if self.initial_mouth_height == 0 and face_data["face_metrics"]["mouth_height_pos"] > 0:
                    self.initial_mouth_height = face_data["face_metrics"]["mouth_height_pos"]
                    print(f"초기 입 높이 설정: {self.initial_mouth_height:.4f}")
                
                # 지루함 감지 (얼굴 크기 감소)
                if face_data["face_metrics"]["face_size"] > 0 and self.initial_face_size > 0:
                    size_ratio = face_data["face_metrics"]["face_size"] / self.initial_face_size
                    size_text = f"얼굴 크기 비율: {size_ratio:.2f}"
                    if size_ratio < BORED_THRESHOLD:  # 0.8배 이하 = 20% 이상 작아짐
                        cv2.putText(frame, f"지루함 감지! {size_text}", (10, 120), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self.detected_states.append('bored')
                        self.detected_state_code = 1  # 지루함
                        self.detected_state_label.config(text="지루함", fg=GUI_COLORS['highlight'])
                    else:
                        cv2.putText(frame, size_text, (10, 120), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                # 하품(피곤함) 감지
                if face_data["face_metrics"]["mouth_open_height"] > 0:
                    mouth_open = face_data["face_metrics"]["mouth_open_height"]
                    mouth_text = f"입 벌어짐: {mouth_open:.3f}"
                    
                    if mouth_open > YAWN_THRESHOLD:  # 입이 충분히 벌어짐
                        if not self.is_yawning:
                            self.is_yawning = True
                            self.yawn_start_time = time.time()
                            cv2.putText(frame, f"하품 감지 시작! {mouth_text}", (10, 150), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        else:
                            yawn_duration = time.time() - self.yawn_start_time
                            self.duration_label.config(text=f"{yawn_duration:.1f}초")
                            cv2.putText(frame, f"하품 지속: {yawn_duration:.1f}초 {mouth_text}", (10, 150), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                            if yawn_duration >= YAWN_DURATION:  # 3초 이상 지속
                                cv2.putText(frame, "피곤함 상태 감지!", (10, 180), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                                self.detected_states.append('tired')
                                self.detected_state_code = 2  # 피곤함
                                self.detected_state_label.config(text="피곤함", fg=GUI_COLORS['red'])
                    else:
                        cv2.putText(frame, mouth_text, (10, 150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                        self.is_yawning = False
                        self.yawn_start_time = None
                
                # 불만족 감지 (미간 거리 변화)
                if face_data["face_metrics"]["eyebrow_distance"] > 0 and self.initial_eyebrow_distance > 0:
                    eyebrow_ratio = face_data["face_metrics"]["eyebrow_distance"] / self.initial_eyebrow_distance
                    eyebrow_text = f"미간 거리 비율: {eyebrow_ratio:.2f}"
                    
                    if eyebrow_ratio < EYEBROW_THRESHOLD:  # 0.9배 이하 = 10% 이상 짧아짐
                        cv2.putText(frame, f"불만족 감지(미간)! {eyebrow_text}", (10, 210), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                        self.detected_states.append('dissatisfied')
                        self.detected_state_code = 4  # 불만족
                        self.detected_state_label.config(text="불만족", fg=GUI_COLORS['purple'])
                    else:
                        cv2.putText(frame, eyebrow_text, (10, 210), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                # 불만족 감지 (입 높이 변화)
                if face_data["face_metrics"]["mouth_height_pos"] > 0 and self.initial_mouth_height > 0:
                    mouth_shift = self.initial_mouth_height - face_data["face_metrics"]["mouth_height_pos"]
                    mouth_pos_text = f"입 높이 변화: {mouth_shift:.3f}"
                    
                    if mouth_shift > MOUTH_UP_THRESHOLD:  # 1cm 이상 올라감
                        cv2.putText(frame, f"불만족 감지(입)! {mouth_pos_text}", (10, 240), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                        self.detected_states.append('dissatisfied')
                        self.detected_state_code = 4  # 불만족
                        self.detected_state_label.config(text="불만족", fg=GUI_COLORS['purple'])
                    else:
                        cv2.putText(frame, mouth_pos_text, (10, 240), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                # 상태 예측 (모델이 있는 경우)
                if self.model and self.scaler:
                    try:
                        predicted_state = self.predict_state(landmarks_data)
                        self.detected_state_code = predicted_state
                        self.detected_state_label.config(
                            text=STATE_KOREAN[predicted_state],
                            fg=self.get_state_color_tk(predicted_state)
                        )
                        confidence = 70 + (len(self.state_history) * 3)  # 70-100% 사이 신뢰도
                        self.update_confidence(confidence)
                        cv2.putText(frame, f"감지된 상태: {STATE_KOREAN[predicted_state]} ({confidence}%)", 
                                    (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                    self.get_state_color_cv2(predicted_state), 2)
                    except Exception as e:
                        print(f"상태 예측 중 오류: {e}")
            
            # 데이터 수집 중인 경우 CSV에 저장
            if self.is_collecting:
                if self.collection_count < self.max_samples:
                    self.save_to_csv(landmarks_data, self.current_state)
                    self.collection_count += 1
                    self.sample_count_label.config(text=str(self.collection_count))
                else:
                    self.is_collecting = False
                    self.collection_btn.config(text="수집 시작", bg=GUI_COLORS['green'])
                    self.update_status(f"최대 샘플 수({self.max_samples}개) 도달. 데이터 수집 자동 중지.")
        
        return frame, landmarks_data
    
    def update_gui_with_frame(self, frame):
        """처리된 프레임으로 GUI 업데이트"""
        try:
            # 이미지 크기 조정 및 변환
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_img = Image.fromarray(display_frame)
            
            # 웹캠 화면 크기 가져오기
            width = self.webcam_display.winfo_width()
            height = self.webcam_display.winfo_height()
            
            # 영상 크기 조정 (창 크기에 맞춤)
            if width > 1 and height > 1:  # 유효한 크기인 경우
                display_img = display_img.resize((width, height), Image.LANCZOS)
            
            # Tkinter 이미지 객체로 변환
            self.tk_img = ImageTk.PhotoImage(image=display_img)
            
            # 레이블에 이미지 설정
            self.webcam_display.configure(image=self.tk_img)
            self.webcam_display.image = self.tk_img  # 참조 유지
            
        except Exception as e:
            print(f"GUI 업데이트 중 오류: {e}")
    
    def get_state_color_tk(self, state_code):
        """상태 코드에 해당하는 Tkinter 색상 반환"""
        color_map = {
            0: GUI_COLORS['green'],     # 만족
            1: GUI_COLORS['highlight'], # 지루함
            2: GUI_COLORS['red'],       # 피곤함
            3: GUI_COLORS['orange'],    # 고민 중
            4: GUI_COLORS['purple']     # 불만족
        }
        return color_map.get(state_code, GUI_COLORS['text'])
    
    def get_state_color_cv2(self, state_code):
        """상태 코드에 해당하는 OpenCV BGR 색상 반환"""
        return STATE_COLORS.get(state_code, (255, 255, 255))
    
    def on_closing(self):
        """창 닫기 이벤트 처리"""
        self.is_running = False
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        
        if hasattr(self, 'hands'):
            self.hands.close()
        
        if not self.external_root:
            self.root.destroy()
            print("프로그램 종료")
    
    def run(self):
        """GUI 실행"""
        if not self.external_root:
            self.root.mainloop()
    
    # 여기에 CSV 저장, 모델 학습, 특징 추출 등의 함수 추가
    # (기존 FacialDataCollector 클래스의 나머지 메서드들 포함)

# 메인 실행
if __name__ == "__main__":
    app = FacialDataCollector()
    app.run()
_running = False
        self.start_stop_btn.config(text="시작", bg=GUI_COLORS['green'])
        self.update_status("중지됨")
        else:
            if not self.webcam_available:
                self.update_status("웹캠을 사용할 수 없습니다.")
                return
                
            self.is_running = True
            self.start_stop_btn.config(text="중지", bg=GUI_COLORS['red'])
            self.update_status("실행 중...")
            self.update_webcam()
    
    def toggle_collection(self):
        """데이터 수집 시작/중지 버튼 이벤트 처리"""
        if self.is_collecting:
            self.is_collecting = False
            self.collection_btn.config(text="수집 시작", bg=GUI_COLORS['green'])
            self.update_status("데이터 수집 중지됨")
        else:
            self.is_collecting = True
            self.collection_count = 0
            self.sample_count_label.config(text="0")
            self.collection_btn.config(text="수집 중지", bg=GUI_COLORS['red'])
            self.update_status(f"{STATE_KOREAN[self.current_state]} 상태 데이터 수집 시작")
    
        def toggle_analysis(self):
            """상태 감지 시작/중지 버튼 이벤트 처리"""
            if self.is_analyzing:
                self.is_analyzing = False
                self.analysis_btn.config(text="감지 시작", bg=GUI_COLORS['highlight'])
                self.update_status("상태 감지 중지됨")
            else:
                self.is_analyzing = True
                self.initial_face_size = 0
                self.initial_eyebrow_distance = 0
                self.initial_mouth_height = 0
                self.analysis_btn.config(text="감지 중지", bg=GUI_COLORS['red'])
                self.update_status("상태 감지 시작됨")
            
            # 모델이 없는 경우 경고
            if not self.model:
                self.update_status("모델이 없습니다. 먼저 모델을 학습하거나 로드하세요.", warning=True)
def initialize_csv(self):
        """CSV 파일 초기화 (필요시)"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # 헤더 작성
                header = ['timestamp', 'state']
                
                # 얼굴 랜드마크 컬럼 (x, y 좌표)
                for i in range(468):  # MediaPipe는 468개의 랜드마크
                    header.extend([f'face_x_{i}', f'face_y_{i}'])
                
                # 얼굴 메트릭 컬럼
                header.extend([
                    'face_size', 'face_aspect_ratio', 'eye_aspect_ratio_left', 
                    'eye_aspect_ratio_right', 'mouth_open_height', 'mouth_height_pos',
                    'eyebrow_distance', 'head_pose_pitch', 'head_pose_yaw', 'head_pose_roll'
                ])
                
                # 손 랜드마크 컬럼 (각 손 21개 랜드마크의 x, y 좌표)
                for i in range(21):
                    header.extend([f'hand1_x_{i}', f'hand1_y_{i}'])
                for i in range(21):
                    header.extend([f'hand2_x_{i}', f'hand2_y_{i}'])
                
                # 손 메트릭 컬럼
                header.extend(['hand_face_distance', 'hand_jaw_overlap', 'hand_detected'])
                
                # 상태 지속 시간
                header.extend(['yawn_duration', 'thinking_duration'])
                
                writer.writerow(header)
                print(f"CSV 파일 초기화: {self.csv_file}")
    
    def load_model(self):
        """저장된 모델 로드 (존재하는 경우)"""
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                print(f"모델 로드 성공: {self.model_file}")
                if hasattr(self, 'model_status_label'):
                    self.model_status_label.config(text="모델 로드됨", fg=GUI_COLORS['green'])
                return True
            except Exception as e:
                print(f"모델 로드 중 오류: {e}")
                return False
        else:
            print("모델 파일이 존재하지 않음")
            return False
    
    def extract_facial_features_mediapipe(self, frame, face_results):
        """MediaPipe로 얼굴 특징 추출"""
        h, w, _ = frame.shape
        face_landmarks = []
        
        try:
            # 첫 번째 얼굴의 랜드마크 좌표 추출
            landmarks = face_results.multi_face_landmarks[0]
            
            for i in range(468):  # MediaPipe는 468개의 랜드마크
                if i < len(landmarks.landmark):
                    lm = landmarks.landmark[i]
                    x, y = int(lm.x * w), int(lm.y * h)
                    face_landmarks.extend([x, y])
                else:
                    face_landmarks.extend([0, 0])  # 누락된 랜드마크를 0으로 채움
            
            # 주요 랜드마크 좌표
            # 눈 좌표 (MediaPipe 랜드마크 인덱스)
            left_eye = [33, 160, 158, 133, 153, 144]  # 왼쪽 눈
            right_eye = [362, 385, 387, 263, 373, 380]  # 오른쪽 눈
            
            # 입 좌표
            upper_lip = [13]  # 윗입술 중앙
            lower_lip = [14]  # 아랫입술 중앙
            
            # 미간 좌표
            left_eyebrow = [65]  # 왼쪽 눈썹 안쪽
            right_eyebrow = [295]  # 오른쪽 눈썹 안쪽
            
            # 얼굴 메트릭 계산
            # 얼굴 크기 (얼굴 너비 기준)
            left_cheek = landmarks.landmark[234]  # 왼쪽 볼
            right_cheek = landmarks.landmark[454]  # 오른쪽 볼
            face_width = abs(right_cheek.x - left_cheek.x) * w
            face_size = face_width / w  # 정규화된 크기
            
            # 눈 종횡비 (눈 높이/너비)
            def eye_aspect_ratio(eye_pts):
                eye_coords = []
                for idx in eye_pts:
                    lm = landmarks.landmark[idx]
                    eye_coords.append((lm.x * w, lm.y * h))
                
                # 수직 거리
                vertical_dist1 = np.linalg.norm(np.array(eye_coords[1]) - np.array(eye_coords[5]))
                vertical_dist2 = np.linalg.norm(np.array(eye_coords[2]) - np.array(eye_coords[4]))
                
                # 수평 거리
                horizontal_dist = np.linalg.norm(np.array(eye_coords[0]) - np.array(eye_coords[3]))
                
                # 눈 종횡비
                ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
                return ear
            
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            
            # 입 벌어짐 (입 높이)
            upper_lm = landmarks.landmark[upper_lip[0]]
            lower_lm = landmarks.landmark[lower_lip[0]]
            mouth_open_height = abs(lower_lm.y - upper_lm.y) * h / w  # 정규화된 높이
            
            # 입 높이 위치 (상대적 위치)
            mouth_height_pos = (upper_lm.y + lower_lm.y) / 2
            
            # 미간 거리
            left_eb = landmarks.landmark[left_eyebrow[0]]
            right_eb = landmarks.landmark[right_eyebrow[0]]
            eyebrow_distance = abs(right_eb.x - left_eb.x) * w / h  # 정규화된 거리
            
            # 얼굴 각도 추정 (간단한 근사값)
            nose_tip = landmarks.landmark[4]
            left_eye_center = landmarks.landmark[159]
            right_eye_center = landmarks.landmark[386]
            
            dx = right_eye_center.x - left_eye_center.x
            dy = right_eye_center.y - left_eye_center.y
            
            # 머리 회전 각도 (yaw)
            head_pose_yaw = abs(nose_tip.x - 0.5) * 2  # 중앙에서 벗어난 정도
            
            # 머리 상하 각도 (pitch)
            head_pose_pitch = abs(nose_tip.y - 0.5) * 2
            
            # 머리 기울기 (roll)
            head_pose_roll = np.arctan2(dy, dx) if dx != 0 else 0
            
            # 얼굴 종횡비
            chin = landmarks.landmark[152]  # 턱
            forehead = landmarks.landmark[10]  # 이마
            face_height = abs(chin.y - forehead.y) * h
            face_aspect_ratio = face_height / face_width if face_width > 0 else 0
            
            # 메트릭 데이터 구성
            metrics = {
                'face_size': face_size,
                'face_aspect_ratio': face_aspect_ratio,
                'eye_aspect_ratio_left': left_ear,
                'eye_aspect_ratio_right': right_ear,
                'mouth_open_height': mouth_open_height,
                'mouth_height_pos': mouth_height_pos,
                'eyebrow_distance': eyebrow_distance,
                'head_pose_pitch': head_pose_pitch,
                'head_pose_yaw': head_pose_yaw,
                'head_pose_roll': head_pose_roll
            }
            
            return {
                "face_landmarks": face_landmarks,
                "face_metrics": metrics
            }
            
        except Exception as e:
            print(f"MediaPipe 얼굴 특징 추출 중 오류: {e}")
            # 빈 데이터 반환
            face_landmarks = [0] * (468 * 2)  # 468개 랜드마크의 x, y 좌표
            metrics = {
                'face_size': 0,
                'face_aspect_ratio': 0,
                'eye_aspect_ratio_left': 0,
                'eye_aspect_ratio_right': 0,
                'mouth_open_height': 0,
                'mouth_height_pos': 0,
                'eyebrow_distance': 0,
                'head_pose_pitch': 0,
                'head_pose_yaw': 0,
                'head_pose_roll': 0
            }
            return {
                "face_landmarks": face_landmarks,
                "face_metrics": metrics
            }
    
    def extract_facial_features_insightface(self, frame, face):
        """InsightFace로 얼굴 특징 추출"""
        h, w, _ = frame.shape
        face_landmarks = []
        
        try:
            # InsightFace에서 2D 랜드마크 (106개) 추출
            if hasattr(face, 'landmark_2d_106'):
                landmarks = face.landmark_2d_106.astype(np.int32)
                
                # MediaPipe와 동일한 468개 형식으로 변환 (없는 부분은 0으로 채움)
                for i in range(468):
                    if i < 106:  # 실제 InsightFace 랜드마크
                        x, y = landmarks[i][0], landmarks[i][1]
                        face_landmarks.extend([x, y])
                    else:  # 나머지는 0으로 채움
                        face_landmarks.extend([0, 0])
                
                # 주요 랜드마크 인덱스 (InsightFace)
                # 눈 좌표
                left_eye = [66, 67, 68, 69, 70, 71]  # 왼쪽 눈 윤곽
                right_eye = [75, 76, 77, 78, 79, 80]  # 오른쪽 눈 윤곽
                
                # 입 좌표
                upper_lip = [89]  # 윗입술 중앙
                lower_lip = [95]  # 아랫입술 중앙
                
                # 미간 좌표
                left_eyebrow = [72]  # 왼쪽 눈썹 안쪽
                right_eyebrow = [81]  # 오른쪽 눈썹 안쪽
                
                # 얼굴 메트릭 계산
                # 얼굴 크기 (얼굴 너비 기준)
                left_cheek = landmarks[2]  # 왼쪽 볼 근처
                right_cheek = landmarks[14]  # 오른쪽 볼 근처
                face_width = abs(right_cheek[0] - left_cheek[0])
                face_size = face_width / w  # 정규화된 크기
                
                # 눈 종횡비 (눈 높이/너비)
                def eye_aspect_ratio(eye_pts):
                    eye_coords = [landmarks[i] for i in eye_pts]
                    
                    # 수직 거리
                    vertical_dist1 = np.linalg.norm(np.array(eye_coords[1]) - np.array(eye_coords[5]))
                    vertical_dist2 = np.linalg.norm(np.array(eye_coords[2]) - np.array(eye_coords[4]))
                    
                    # 수평 거리
                    horizontal_dist = np.linalg.norm(np.array(eye_coords[0]) - np.array(eye_coords[3]))
                    
                    # 눈 종횡비
                    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist) if horizontal_dist > 0 else 0
                    return ear
                
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                
                # 입 벌어짐 (입 높이)
                upper_lm = landmarks[upper_lip[0]]
                lower_lm = landmarks[lower_lip[0]]
                mouth_open_height = abs(lower_lm[1] - upper_lm[1]) / h  # 정규화된 높이
                
                # 입 높이 위치 (상대적 위치)
                mouth_height_pos = (upper_lm[1] + lower_lm[1]) / (2 * h)
                
                # 미간 거리
                left_eb = landmarks[left_eyebrow[0]]
                right_eb = landmarks[right_eyebrow[0]]
                eyebrow_distance = abs(right_eb[0] - left_eb[0]) / w  # 정규화된 거리
                
                # 얼굴 각도 추정 (간단한 근사값)
                nose_tip = landmarks[94]  # 코 끝
                left_eye_center = landmarks[66]  # 왼쪽 눈
                right_eye_center = landmarks[79]  # 오른쪽 눈
                
                dx = right_eye_center[0] - left_eye_center[0]
                dy = right_eye_center[1] - left_eye_center[1]
                
                # 머리 회전 각도 (yaw)
                head_pose_yaw = abs(nose_tip[0] / w - 0.5) * 2  # 중앙에서 벗어난 정도
                
                # 머리 상하 각도 (pitch)
                head_pose_pitch = abs(nose_tip[1] / h - 0.5) * 2
                
                # 머리 기울기 (roll)
                head_pose_roll = np.arctan2(dy, dx) if dx != 0 else 0
                
                # 얼굴 종횡비
                chin = landmarks[95]  # 턱
                forehead = landmarks[72]  # 이마
                face_height = abs(chin[1] - forehead[1])
                face_aspect_ratio = face_height / face_width if face_width > 0 else 0
                
                # 메트릭 데이터 구성
                metrics = {
                    'face_size': face_size,
                    'face_aspect_ratio': face_aspect_ratio,
                    'eye_aspect_ratio_left': left_ear,
                    'eye_aspect_ratio_right': right_ear,
                    'mouth_open_height': mouth_open_height,
                    'mouth_height_pos': mouth_height_pos,
                    'eyebrow_distance': eyebrow_distance,
                    'head_pose_pitch': head_pose_pitch,
                    'head_pose_yaw': head_pose_yaw,
                    'head_pose_roll': head_pose_roll
                }
                
                return {
                    "face_landmarks": face_landmarks,
                    "face_metrics": metrics
                }
            else:
                raise Exception("InsightFace 2D 랜드마크를 찾을 수 없습니다.")
                
        except Exception as e:
            print(f"InsightFace 얼굴 특징 추출 중 오류: {e}")
            # 빈 데이터 반환
            face_landmarks = [0] * (468 * 2)  # 468개 랜드마크의 x, y 좌표
            metrics = {
                'face_size': 0,
                'face_aspect_ratio': 0,
                'eye_aspect_ratio_left': 0,
                'eye_aspect_ratio_right': 0,
                'mouth_open_height': 0,
                'mouth_height_pos': 0,
                'eyebrow_distance': 0,
                'head_pose_pitch': 0,
                'head_pose_yaw': 0,
                'head_pose_roll': 0
            }
            return {
                "face_landmarks": face_landmarks,
                "face_metrics": metrics
            }
    
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
    
    def check_hand_jaw_overlap(self, hand_data, face_data):
        """손과 턱의 겹침 확인"""
        # 손이 감지된 경우에만 확인
        if hand_data["hand_metrics"]["hand_detected"] > 0:
            # 손과 턱 겹침 여부
            return hand_data["hand_metrics"]["hand_jaw_overlap"] > 0
        return False
    
    def save_to_csv(self, landmarks_data, state):
        """랜드마크 데이터를 CSV에 저장"""
        try:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                # 현재 시간
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                
                # 행 데이터 구성
                row = [timestamp, state]
                
                # 얼굴 랜드마크
                row.extend(landmarks_data["face_landmarks"])
                
                # 얼굴 메트릭
                for key in ['face_size', 'face_aspect_ratio', 'eye_aspect_ratio_left', 
                         'eye_aspect_ratio_right', 'mouth_open_height', 'mouth_height_pos',
                         'eyebrow_distance', 'head_pose_pitch', 'head_pose_yaw', 'head_pose_roll']:
                    row.append(landmarks_data.get(key, 0))
                
                # 손 랜드마크
                row.extend(landmarks_data["hand1_landmarks"])
                row.extend(landmarks_data["hand2_landmarks"])
                
                # 손 메트릭
                for key in ['hand_face_distance', 'hand_jaw_overlap', 'hand_detected']:
                    row.append(landmarks_data.get(key, 0))
                
                # 상태 지속 시간
                row.append(landmarks_data.get("yawn_duration", 0))
                row.append(landmarks_data.get("thinking_duration", 0))
                
                writer.writerow(row)
                
        except Exception as e:
            print(f"CSV 저장 중 오류: {e}")
    
    def prepare_features(self, landmarks_data):
        """머신러닝 특징 준비"""
        features = []
        
        # 얼굴 메트릭 추가
        for key in ['face_size', 'face_aspect_ratio', 'eye_aspect_ratio_left', 
                   'eye_aspect_ratio_right', 'mouth_open_height', 'mouth_height_pos',
                   'eyebrow_distance', 'head_pose_pitch', 'head_pose_yaw', 'head_pose_roll']:
            features.append(landmarks_data.get(key, 0))
        
        # 손 메트릭 추가
        for key in ['hand_face_distance', 'hand_jaw_overlap', 'hand_detected']:
            features.append(landmarks_data.get(key, 0))
        
        # 상태 지속 시간
        features.append(landmarks_data.get("yawn_duration", 0))
        features.append(landmarks_data.get("thinking_duration", 0))
        
        return features
    
    def train_model(self):
        """수집된 데이터로 상태 감지 모델 학습"""
        try:
            # CSV 파일 로드
            if not os.path.exists(self.csv_file):
                print("CSV 파일이 존재하지 않습니다.")
                return False
            
            df = pd.read_csv(self.csv_file)
            
            if df.empty:
                print("데이터가 비어 있습니다.")
                return False
            
            # 특징 선택 (얼굴 및 손 메트릭만)
            feature_columns = [
                'face_size', 'face_aspect_ratio', 'eye_aspect_ratio_left', 
                'eye_aspect_ratio_right', 'mouth_open_height', 'mouth_height_pos',
                'eyebrow_distance', 'head_pose_pitch', 'head_pose_yaw', 'head_pose_roll',
                'hand_face_distance', 'hand_jaw_overlap', 'hand_detected',
                'yawn_duration', 'thinking_duration'
            ]
            
            # 특징 데이터 준비
            X = df[feature_columns].values
            y = df['state'].values
            
            # 특징 스케일링
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # 데이터 분할 (훈련/테스트)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42)
            
            # 모델 학습 (랜덤 포레스트)
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42)
            self.model.fit(X_train, y_train)
            
            # 모델 평가
            accuracy = self.model.score(X_test, y_test)
            print(f"모델 정확도: {accuracy:.4f}")
            
            # 모델 저장
            model_data = {
                'model': self.model,
                'scaler': self.scaler
            }
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"모델 저장 완료: {self.model_file}")
            return True
            
        except Exception as e:
            print(f"모델 학습 중 오류: {e}")
            return False
    
    def predict_state(self, landmarks_data):
        """현재 상태 예측"""
        if not self.model or not self.scaler:
            return 0  # 모델이 없으면 기본 상태 (만족) 반환
        
        try:
            # 특징 추출
            features = self.prepare_features(landmarks_data)
            features = np.array(features).reshape(1, -1)
            
            # 특징 스케일링
            features_scaled = self.scaler.transform(features)
            
            # 예측
            state = self.model.predict(features_scaled)[0]
            
            # 예측 결과 기록
            self.state_history.append(state)
            
            # 가장 빈번한 상태 결정 (최근 기록 기반)
            if len(self.state_history) > 0:
                counts = {}
                for s in self.state_history:
                    counts[s] = counts.get(s, 0) + 1
                
                # 가장 빈번한 상태
                most_common_state = max(counts.items(), key=lambda x: x[1])[0]
                return most_common_state
            
            return state
            
        except Exception as e:
            print(f"상태 예측 중 오류: {e}")
            return 0  # 오류 발생시 기본 상태 반환

# 메인 실행
if __name__ == "__main__":
    app = FacialDataCollector()
    app.run()            