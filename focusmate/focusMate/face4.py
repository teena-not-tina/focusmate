import cv2
import numpy as np
import time
import threading
import os
import subprocess
import sys
from tkinter import *
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import webbrowser
import mediapipe as mp
from datetime import datetime

# InsightFace 가져오기
try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    INSIGHTFACE_AVAILABLE = True
    print("InsightFace 가져오기 성공")
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("InsightFace를 사용할 수 없습니다. 'pip install insightface' 명령으로 설치하세요.")

# 로컬 비디오 파일 경로 (상태별)
VIDEO_FILES = {
    'default': "gamma.mp4",       # 기본 비디오 파일
    'bored': "gamma.mp4",         # 베타파 비디오
    'tired': "gamma.mp4",         # 활기찬 비디오
    'thinking': "gamma.mp4",      # 감마파 비디오
    'satisfied': "gamma.mp4",     # 환희 가득한 비디오
    'dissatisfied': "gamma.mp4"   # 릴렉스 비디오
}

# 시간 설정 (초 단위)
STUDY_TIME = 25 * 60  # 25분
BREAK_TIME = 5 * 60   # 5분
ANALYSIS_INTERVAL = 5  # 매 5초마다 분석
CAPTURE_INTERVAL = 1   # 1초마다 캡처

# 상태 감지 조건 설정
BORED_THRESHOLD = 0.8  # 얼굴이 20% 이상 작아지면 지루함 (0.8배 이하)
YAWN_THRESHOLD = 0.05  # 입이 5cm 이상 벌어진 것으로 간주 (실제 픽셀값은 상대적)
YAWN_DURATION = 3.0    # 하품 감지 유지 시간 (초)
THINKING_DURATION = 5.0  # 고민 중 상태 유지 시간 (초)
EYEBROW_THRESHOLD = 0.9  # 미간 거리가 10% 이상 짧아지면 불만족 (0.9배 이하)
MOUTH_UP_THRESHOLD = 0.01  # 입 높이가 1cm 이상 올라가면 불만족 (픽셀 단위)

# 얼굴 및 손 인식 설정
class StudyMoodMonitor:
    def __init__(self):
        # 상태 감지 변수
        self.detected_states = []
        self.dominant_state = 'default'
        self.initial_face_size = 0
        self.initial_eyebrow_distance = 0
        self.initial_mouth_height = 0
        
        # 하품 감지 관련 변수
        self.yawn_start_time = None
        self.is_yawning = False
        
        # 고민 중 감지 관련 변수
        self.thinking_start_time = None
        self.is_thinking = False
        
        # 마지막 분석 시간
        self.last_analysis_time = 0
        
        # 손 제스처 관련 변수
        self.hand_on_face_detected = False
        
        # 실행 상태 변수
        self.is_running = False
        self.face_detected = False
        self.study_time_remaining = STUDY_TIME
        self.break_time_remaining = BREAK_TIME
        self.mode = 'waiting'  # waiting, studying, break
        
        # 타이머 스레드
        self.timer_thread = None
        
        # 비디오 재생 변수
        self.video_process = None
        self.is_video_playing = False
        
        # 캡처 관련 변수
        self.capture_enabled = False
        self.capture_folder = "captures"
        self.last_capture_time = 0
        
        # 캡처 폴더 생성
        if not os.path.exists(self.capture_folder):
            os.makedirs(self.capture_folder)
            print(f"캡처 폴더 생성: {self.capture_folder}")
        
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
        
        # MediaPipe 핸드 트래커 설정
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.mediapipe_available = True
            print("MediaPipe Hands 초기화 성공")
        except Exception as e:
            print(f"MediaPipe Hands 초기화 중 오류: {e}")
            self.mediapipe_available = False
            
        # 웹캠 설정
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("웹캠을 열 수 없습니다. 카메라가 연결되어 있는지 확인하세요.")
                self.webcam_available = False
            else:
                self.webcam_available = True
                print("웹캠 초기화 성공")
        except Exception as e:
            print(f"웹캠 초기화 중 오류: {e}")
            self.webcam_available = False
        
        # GUI 설정
        self.setup_gui()
        
    def setup_gui(self):
        # 메인 윈도우 설정
        self.root = Tk()
        self.root.title("공부 상태 모니터링 시스템")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        self.root.configure(background='#2c3e50')
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 메인 프레임
        main_frame = Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # 왼쪽 패널 (웹캠 화면)
        self.webcam_frame = Frame(main_frame, bg='black', width=640, height=480)
        self.webcam_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 5))
        self.webcam_frame.pack_propagate(False)
        
        # 웹캠 표시 레이블
        self.webcam_label = Label(self.webcam_frame, bg='black')
        self.webcam_label.pack(fill=BOTH, expand=True)
        
        # 오른쪽 패널 (상태 및 제어)
        control_frame = Frame(main_frame, bg='#34495e', width=300)
        control_frame.pack(side=RIGHT, fill=BOTH, padx=(5, 0))
        
        # 상태 정보 섹션
        info_frame = Frame(control_frame, bg='#34495e', padx=10, pady=10)
        info_frame.pack(fill=X, pady=(0, 5))
        
        # 타이틀
        title_label = Label(info_frame, text="공부 상태 모니터", 
                           font=("Helvetica", 16, "bold"), bg='#34495e', fg='white')
        title_label.pack(fill=X, pady=(0, 10))
        
        # 모드 프레임
        mode_frame = Frame(info_frame, bg='#34495e')
        mode_frame.pack(fill=X, pady=5)
        
        Label(mode_frame, text="현재 모드:", width=10, anchor=W,
              font=("Helvetica", 12), bg='#34495e', fg='white').pack(side=LEFT)
        
        self.mode_label = Label(mode_frame, text="대기 중", width=20, anchor=W,
                               font=("Helvetica", 12, "bold"), bg='#34495e', fg='#3498db')
        self.mode_label.pack(side=LEFT, fill=X, expand=True)
        
        # 타이머 프레임
        timer_frame = Frame(info_frame, bg='#34495e')
        timer_frame.pack(fill=X, pady=5)
        
        Label(timer_frame, text="남은 시간:", width=10, anchor=W,
              font=("Helvetica", 12), bg='#34495e', fg='white').pack(side=LEFT)
        
        self.timer_label = Label(timer_frame, text="25:00", width=20, anchor=W,
                                font=("Helvetica", 12, "bold"), bg='#34495e', fg='#e74c3c')
        self.timer_label.pack(side=LEFT, fill=X, expand=True)
        
        # 상태 프레임
        state_frame = Frame(info_frame, bg='#34495e')
        state_frame.pack(fill=X, pady=5)
        
        Label(state_frame, text="현재 상태:", width=10, anchor=W,
              font=("Helvetica", 12), bg='#34495e', fg='white').pack(side=LEFT)
        
        self.state_label = Label(state_frame, text="기본", width=20, anchor=W,
                                font=("Helvetica", 12, "bold"), bg='#34495e', fg='#2ecc71')
        self.state_label.pack(side=LEFT, fill=X, expand=True)
        
        # 얼굴 감지 상태
        face_frame = Frame(info_frame, bg='#34495e')
        face_frame.pack(fill=X, pady=5)
        
        Label(face_frame, text="얼굴 감지:", width=10, anchor=W,
              font=("Helvetica", 12), bg='#34495e', fg='white').pack(side=LEFT)
        
        self.face_status = Label(face_frame, text="감지되지 않음", width=20, anchor=W,
                               font=("Helvetica", 12), bg='#34495e', fg='#e74c3c')
        self.face_status.pack(side=LEFT, fill=X, expand=True)
        
        # 손 감지 상태
        hand_frame = Frame(info_frame, bg='#34495e')
        hand_frame.pack(fill=X, pady=5)
        
        Label(hand_frame, text="손 감지:", width=10, anchor=W,
              font=("Helvetica", 12), bg='#34495e', fg='white').pack(side=LEFT)
        
        self.hand_status = Label(hand_frame, text="감지되지 않음", width=20, anchor=W,
                               font=("Helvetica", 12), bg='#34495e', fg='#e74c3c')
        self.hand_status.pack(side=LEFT, fill=X, expand=True)
        
        # 캡처 상태
        capture_frame = Frame(info_frame, bg='#34495e')
        capture_frame.pack(fill=X, pady=5)
        
        Label(capture_frame, text="화면 캡처:", width=10, anchor=W,
              font=("Helvetica", 12), bg='#34495e', fg='white').pack(side=LEFT)
        
        self.capture_status = Label(capture_frame, text="비활성화", width=20, anchor=W,
                                  font=("Helvetica", 12), bg='#34495e', fg='#e74c3c')
        self.capture_status.pack(side=LEFT, fill=X, expand=True)
        
        # 비디오 상태
        video_frame = Frame(info_frame, bg='#34495e')
        video_frame.pack(fill=X, pady=5)
        
        Label(video_frame, text="비디오:", width=10, anchor=W,
              font=("Helvetica", 12), bg='#34495e', fg='white').pack(side=LEFT)
        
        self.video_status = Label(video_frame, text="재생 안 함", width=20, anchor=W,
                                font=("Helvetica", 12), bg='#34495e', fg='#e74c3c')
        self.video_status.pack(side=LEFT, fill=X, expand=True)
        
        # 상태 지속 시간
        duration_frame = Frame(info_frame, bg='#34495e')
        duration_frame.pack(fill=X, pady=5)
        
        Label(duration_frame, text="지속 시간:", width=10, anchor=W,
              font=("Helvetica", 12), bg='#34495e', fg='white').pack(side=LEFT)
        
        self.duration_label = Label(duration_frame, text="0초", width=20, anchor=W,
                                   font=("Helvetica", 12, "bold"), bg='#34495e', fg='#f39c12')
        self.duration_label.pack(side=LEFT, fill=X, expand=True)
        
        # 구분선
        ttk.Separator(control_frame, orient=HORIZONTAL).pack(fill=X, pady=10)
        
        # 제어 버튼 섹션
        control_buttons = Frame(control_frame, bg='#34495e', padx=10, pady=10)
        control_buttons.pack(fill=X)
        
        # 시작/중지 버튼
        self.start_stop_button = Button(control_buttons, text="시작", width=12, height=2,
                                      bg='#2ecc71', fg='white', font=("Helvetica", 12, "bold"),
                                      command=self.toggle_start_stop)
        self.start_stop_button.pack(side=LEFT, padx=5)
        
        # 초기화 버튼
        reset_button = Button(control_buttons, text="초기화", width=12, height=2,
                             bg='#e67e22', fg='white', font=("Helvetica", 12, "bold"),
                             command=self.reset_session)
        reset_button.pack(side=LEFT, padx=5)
        
        # 종료 버튼
        exit_button = Button(control_buttons, text="종료", width=12, height=2,
                           bg='#e74c3c', fg='white', font=("Helvetica", 12, "bold"),
                           command=self.on_closing)
        exit_button.pack(side=LEFT, padx=5)
        
        # 추가 제어 버튼 섹션
        extra_buttons = Frame(control_frame, bg='#34495e', padx=10, pady=10)
        extra_buttons.pack(fill=X)
        
        # 캡처 활성화/비활성화 버튼
        self.capture_button = Button(extra_buttons, text="캡처 시작", width=12, height=2,
                                   bg='#3498db', fg='white', font=("Helvetica", 12, "bold"),
                                   command=self.toggle_capture)
        self.capture_button.pack(side=LEFT, padx=5)
        
        # 캡처 폴더 열기 버튼
        open_folder_button = Button(extra_buttons, text="캡처 폴더", width=12, height=2,
                                   bg='#9b59b6', fg='white', font=("Helvetica", 12, "bold"),
                                   command=self.open_capture_folder)
        open_folder_button.pack(side=LEFT, padx=5)
        
        # 비디오 테스트 재생 버튼
        play_video_button = Button(extra_buttons, text="비디오 테스트", width=12, height=2,
                                 bg='#16a085', fg='white', font=("Helvetica", 12, "bold"),
                                 command=lambda: self.play_video('default'))
        play_video_button.pack(side=LEFT, padx=5)
        
        # 상태 설명
        description_frame = Frame(control_frame, bg='#34495e', padx=10, pady=10)
        description_frame.pack(fill=BOTH, expand=True)
        
        Label(description_frame, text="감지 가능한 상태", 
             font=("Helvetica", 12, "bold"), bg='#34495e', fg='white').pack(anchor=W)
        
        # 상태 설명 목록
        state_desc = Text(description_frame, wrap=WORD, height=10, bg='#2c3e50', fg='white',
                          font=("Helvetica", 10), padx=10, pady=10)
        state_desc.pack(fill=BOTH, expand=True, pady=5)
        state_desc.insert(END, "• 지루함: 얼굴이 20% 이상 작아짐\n")
        state_desc.insert(END, "• 피곤함: 하품 (입 5cm이상 벌려 3초 이상 유지)\n")
        state_desc.insert(END, "• 고민 중: 턱과 손이 겹쳐있고 5초 이상 유지\n")
        state_desc.insert(END, "• 만족: 기본\n")
        state_desc.insert(END, "• 불만족: 미간거리가 10%이상 짧아짐, 입 높이가 1cm이상 올라감\n")
        state_desc.config(state=DISABLED)
        
        # 상태 표시바
        status_bar = Frame(self.root, bg='#34495e', height=25)
        status_bar.pack(fill=X, side=BOTTOM)
        
        self.status_text = Label(status_bar, text="준비됨: ESC 키를 눌러 종료", 
                                bg='#34495e', fg='white', anchor=W, padx=10)
        self.status_text.pack(fill=X)
        
        # 키 바인딩
        self.root.bind('<Escape>', lambda e: self.on_closing())
        self.root.bind('s', lambda e: self.toggle_start_stop())
        self.root.bind('r', lambda e: self.reset_session())
        self.root.bind('c', lambda e: self.toggle_capture())
        self.root.bind('v', lambda e: self.play_video('default'))
        
        # InsightFace 사용 불가능하면 메시지 표시
        if not self.insightface_available:
            self.status_text.config(text="InsightFace를 사용할 수 없습니다. 'pip install insightface onnxruntime' 명령으로 설치하세요.")
        
        # 웹캠 사용 불가능하면 메시지 표시
        if not self.webcam_available:
            self.status_text.config(text="웹캠을 사용할 수 없습니다. 설치 후 다시 시도하세요.")
            error_msg = Label(self.webcam_frame, text="웹캠을 사용할 수 없습니다.\n카메라 연결을 확인하세요.",
                              fg="white", bg="black", font=("Helvetica", 16))
            error_msg.pack(expand=True)
    
    def toggle_start_stop(self):
        if self.is_running:
            self.stop()
            self.start_stop_button.config(text="시작", bg='#2ecc71')
            self.status_text.config(text="모니터링 중지됨")
        else:
            if not self.insightface_available:
                self.status_text.config(text="InsightFace를 설치해야 합니다. 'pip install insightface onnxruntime' 명령으로 설치하세요.")
                return
                
            self.start()
            self.start_stop_button.config(text="중지", bg='#e74c3c')
            self.status_text.config(text="모니터링 시작됨")
    
    def toggle_capture(self):
        """캡처 기능 활성화/비활성화"""
        self.capture_enabled = not self.capture_enabled
        
        if self.capture_enabled:
            self.capture_button.config(text="캡처 중지", bg='#e74c3c')
            self.capture_status.config(text="활성화", fg='#2ecc71')
            self.status_text.config(text=f"1초마다 화면 캡처 시작. 저장 경로: {self.capture_folder}")
            print(f"화면 캡처 활성화. 저장 경로: {self.capture_folder}")
        else:
            self.capture_button.config(text="캡처 시작", bg='#3498db')
            self.capture_status.config(text="비활성화", fg='#e74c3c')
            self.status_text.config(text="화면 캡처 중지됨")
            print("화면 캡처 비활성화")
    
    def open_capture_folder(self):
        """캡처 폴더 열기"""
        try:
            # 운영체제별 폴더 열기 명령
            abs_path = os.path.abspath(self.capture_folder)
            if os.path.exists(abs_path):
                if os.name == 'nt':  # Windows
                    os.startfile(abs_path)
                elif os.name == 'posix':  # macOS, Linux
                    if sys.platform == 'darwin':  # macOS
                        os.system(f'open "{abs_path}"')
                    else:  # Linux
                        os.system(f'xdg-open "{abs_path}"')
                print(f"캡처 폴더 열기: {abs_path}")
            else:
                print(f"캡처 폴더가 존재하지 않습니다: {abs_path}")
                self.status_text.config(text=f"캡처 폴더가 존재하지 않습니다: {self.capture_folder}")
        except Exception as e:
            print(f"캡처 폴더 열기 오류: {e}")
            self.status_text.config(text=f"캡처 폴더 열기 오류: {e}")
    
    def play_video(self, state):
        """로컬 비디오 파일 재생"""
        try:
            # 이미 재생 중인 비디오가 있으면 종료
            if self.is_video_playing and self.video_process:
                try:
                    if self.video_process.poll() is None:  # 프로세스가 아직 실행 중인지 확인
                        self.video_process.terminate()
                        print("이전 비디오 재생 종료")
                except Exception as e:
                    print(f"비디오 종료 중 오류: {e}")
            
            video_file = VIDEO_FILES.get(state, VIDEO_FILES['default'])
            # 파일이 존재하는지 확인
            if not os.path.exists(video_file):
                print(f"비디오 파일을 찾을 수 없습니다: {video_file}")
                self.status_text.config(text=f"비디오 파일을 찾을 수 없습니다: {video_file}")
                self.video_status.config(text="파일 없음", fg='#e74c3c')
                return
            
            # 운영체제에 따른 비디오 재생 명령
            if os.name == 'nt':  # Windows
                self.video_process = subprocess.Popen(['start', video_file], shell=True)
            elif os.name == 'posix':  # macOS, Linux
                if sys.platform == 'darwin':  # macOS
                    self.video_process = subprocess.Popen(['open', video_file])
                else:  # Linux
                    # 다양한 Linux 비디오 플레이어 시도
                    players = ['xdg-open', 'mpv', 'vlc', 'ffplay']
                    for player in players:
                        try:
                            self.video_process = subprocess.Popen([player, video_file])
                            break
                        except FileNotFoundError:
                            continue
            
            self.is_video_playing = True
            current_state = self.get_state_korean(state)
            self.video_status.config(text=f"재생 중: {current_state}", fg='#2ecc71')
            print(f"비디오 재생: {video_file}")
            self.status_text.config(text=f"비디오 재생 시작: {video_file}")
            
            # 비디오 재생 상태 업데이트 스레드
            self.root.after(500, self.check_video_status)
            
        except Exception as e:
            print(f"비디오 재생 중 오류: {e}")
            self.status_text.config(text=f"비디오 재생 중 오류: {e}")
            self.video_status.config(text="재생 오류", fg='#e74c3c')
            self.is_video_playing = False
    
    def check_video_status(self):
        """비디오 재생 상태 확인"""
        if self.is_video_playing and self.video_process:
            try:
                # 프로세스가 종료되었는지 확인
                if self.video_process.poll() is not None:
                    self.is_video_playing = False
                    self.video_status.config(text="재생 완료", fg='#3498db')
                    print("비디오 재생 완료")
                else:
                    # 아직 재생 중이면 계속 확인
                    self.root.after(1000, self.check_video_status)
            except Exception as e:
                print(f"비디오 상태 확인 오류: {e}")
                self.is_video_playing = False
                self.video_status.config(text="재생 오류", fg='#e74c3c')
    
    def capture_frame(self, frame):
        """현재 프레임을 이미지로 저장"""
        try:
            # 현재 시간 기준으로 파일명 생성
            current_time = time.time()
            # 마지막 캡처 후 CAPTURE_INTERVAL초 이상 지났는지 확인
            if current_time - self.last_capture_time >= CAPTURE_INTERVAL:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.capture_folder}/capture_{timestamp}.jpg"
                
                # 이미지 저장
                cv2.imwrite(filename, frame)
                print(f"화면 캡처 저장: {filename}")
                
                # 마지막 캡처 시간 업데이트
                self.last_capture_time = current_time
        except Exception as e:
            print(f"화면 캡처 중 오류: {e}")
    
    def reset_session(self):
        self.stop()
        self.reset_states()
        self.start_stop_button.config(text="시작", bg='#2ecc71')
        self.mode_label.config(text="대기 중")
        self.timer_label.config(text="25:00")
        self.state_label.config(text="기본")
        self.face_status.config(text="감지되지 않음", fg='#e74c3c')
        self.hand_status.config(text="감지되지 않음", fg='#e74c3c')
        self.video_status.config(text="재생 안 함", fg='#e74c3c')
        self.duration_label.config(text="0초")
        self.status_text.config(text="세션이 초기화되었습니다.")
    
    def start(self):
        """모니터링 시작"""
        if not self.webcam_available:
            self.status_text.config(text="웹캠을 사용할 수 없습니다. 설치 후 다시 시도하세요.")
            return
            
        self.is_running = True
        self.mode = 'waiting'
        self.reset_states()
        
        # 타이머 스레드 시작
        self.timer_thread = threading.Thread(target=self.timer_loop)
        self.timer_thread.daemon = True
        self.timer_thread.start()
        
        # 웹캠 처리 시작
        self.update_webcam()
    
    def stop(self):
        """모니터링 중지"""
        self.is_running = False
        if self.timer_thread and self.timer_thread.is_alive():
            self.timer_thread.join(timeout=1.0)
        
        # 비디오 재생 중지
        if self.is_video_playing and self.video_process:
            try:
                if self.video_process.poll() is None:
                    self.video_process.terminate()
                    print("비디오 재생 중지")
                self.is_video_playing = False
                self.video_status.config(text="재생 안 함", fg='#e74c3c')
            except Exception as e:
                print(f"비디오 중지 중 오류: {e}")
        
        self.reset_states()
        self.mode = 'waiting'
    
    def on_closing(self):
        """프로그램 종료"""
        self.is_running = False
        
        # 비디오 재생 중지
        if self.is_video_playing and self.video_process:
            try:
                if self.video_process.poll() is None:
                    self.video_process.terminate()
                    print("비디오 재생 종료")
            except Exception as e:
                print(f"비디오 종료 중 오류: {e}")
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'hands'):
            self.hands.close()
        self.root.destroy()
    
    def reset_states(self):
        """상태 감지 변수 초기화"""
        self.detected_states = []
        self.dominant_state = 'default'
        self.initial_face_size = 0
        self.initial_eyebrow_distance = 0
        self.initial_mouth_height = 0
        self.face_detected = False
        self.yawn_start_time = None
        self.is_yawning = False
        self.thinking_start_time = None
        self.is_thinking = False
        self.hand_on_face_detected = False
        self.study_time_remaining = STUDY_TIME
        self.break_time_remaining = BREAK_TIME
    
    def update_dominant_state(self):
        """가장 많이 감지된 상태 확인"""
        if not self.detected_states:
            self.dominant_state = 'default'
            return

        from collections import Counter
        counter = Counter(self.detected_states)
        self.dominant_state = counter.most_common(1)[0][0]
        self.state_label.config(text=self.get_state_korean(self.dominant_state))
    
    def get_state_korean(self, state):
        """상태 영문 코드를 한글로 변환"""
        state_dict = {
            'default': '기본',
            'bored': '지루함',
            'tired': '피곤함',
            'thinking': '고민 중',
            'satisfied': '만족',
            'dissatisfied': '불만족'
        }
        return state_dict.get(state, '알 수 없음')
    
    def timer_loop(self):
        """타이머 스레드 루프"""
        while self.is_running:
            time.sleep(1)
            
            if self.mode == 'studying':
                self.study_time_remaining -= 1
                minutes, seconds = divmod(self.study_time_remaining, 60)
                self.timer_label.config(text=f"{minutes:02d}:{seconds:02d}")
                
                # 5초마다 상태 분석 결과 업데이트
                if self.study_time_remaining % ANALYSIS_INTERVAL == 0:
                    self.update_dominant_state()
                
                # 공부 시간 종료
                if self.study_time_remaining <= 0:
                    self.mode = 'break'
                    self.mode_label.config(text="휴식 시간", fg='#3498db')
                    print(f"공부 세션 종료! 감지된 상태: {self.get_state_korean(self.dominant_state)}")
                    self.play_video(self.dominant_state)
            
            elif self.mode == 'break':
                self.break_time_remaining -= 1
                minutes, seconds = divmod(self.break_time_remaining, 60)
                self.timer_label.config(text=f"{minutes:02d}:{seconds:02d}")
                
                # 휴식 시간 종료
                if self.break_time_remaining <= 0:
                    self.reset_states()
                    self.mode = 'waiting'
                    self.mode_label.config(text="대기 중", fg='#e74c3c')
                    minutes, seconds = divmod(STUDY_TIME, 60)
                    self.timer_label.config(text=f"{minutes:02d}:{seconds:02d}")
                    print("휴식 시간 종료. 새 세션을 시작하려면 얼굴을 카메라에 보여주세요.")
                    self.status_text.config(text="휴식 시간 종료. 새 세션을 시작하려면 얼굴을 카메라에 보여주세요.")
    
    def draw_info(self, frame):
        """화면에 정보 표시"""
        # 현재 모드 표시
        mode_text = "대기 중"
        if self.mode == 'studying':
            mode_text = "공부 시간"
            color = (0, 255, 0)  # 녹색
        elif self.mode == 'break':
            mode_text = "휴식 시간"
            color = (255, 0, 0)  # 빨강
        else:
            color = (255, 255, 0)  # 노랑
            
        cv2.putText(frame, f"모드: {mode_text}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 남은 시간 표시
        if self.mode == 'studying':
            minutes, seconds = divmod(self.study_time_remaining, 60)
            timer_text = f"남은 시간: {minutes:02d}:{seconds:02d}"
        elif self.mode == 'break':
            minutes, seconds = divmod(self.break_time_remaining, 60)
            timer_text = f"휴식 시간: {minutes:02d}:{seconds:02d}"
        else:
            timer_text = "시간: --:--"
            
        cv2.putText(frame, timer_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 현재 상태 표시
        state_text = f"상태: {self.get_state_korean(self.dominant_state)}"
        cv2.putText(frame, state_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 캡처 상태 표시
        if self.capture_enabled:
            capture_text = "캡처: 활성화"
            capture_color = (0, 255, 0)  # 녹색
        else:
            capture_text = "캡처: 비활성화"
            capture_color = (128, 128, 128)  # 회색
            
        cv2.putText(frame, capture_text, (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, capture_color, 2)
        
        # 비디오 재생 상태 표시
        if self.is_video_playing:
            video_text = "비디오: 재생 중"
            video_color = (0, 255, 0)  # 녹색
        else:
            video_text = "비디오: 재생 안 함"
            video_color = (128, 128, 128)  # 회색
            
        cv2.putText(frame, video_text, (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, video_color, 2)
        
        # 상태 지속 시간 표시
        duration_text = ""
        duration_color = (255, 255, 255)  # 흰색
        
        if self.is_yawning and self.yawn_start_time:
            duration = time.time() - self.yawn_start_time
            duration_text = f"하품: {duration:.1f}초"
            if duration >= YAWN_DURATION:
                duration_color = (0, 255, 0)  # 녹색 (임계값 도달)
            else:
                duration_color = (255, 255, 0)  # 노랑 (진행 중)
        elif self.is_thinking and self.thinking_start_time:
            duration = time.time() - self.thinking_start_time
            duration_text = f"고민 중: {duration:.1f}초"
            if duration >= THINKING_DURATION:
                duration_color = (0, 255, 0)  # 녹색 (임계값 도달)
            else:
                duration_color = (255, 255, 0)  # 노랑 (진행 중)
        
        if duration_text:
            cv2.putText(frame, duration_text, (10, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, duration_color, 2)
            
            # GUI 지속 시간 레이블 업데이트
            self.duration_label.config(text=duration_text)
    
    def is_hand_overlapping_jaw(self, hand_landmarks, face_landmarks):
        """손과 턱이 겹쳐있는지 확인"""
        try:
            # 턱 부근 좌표
            jaw_landmarks = face_landmarks.landmark[93:103]  # 턱 라인 랜드마크
            jaw_y = np.mean([lm.y for lm in jaw_landmarks])
            jaw_x = np.mean([lm.x for lm in jaw_landmarks])
            
            # 손가락 끝 좌표
            hand_points = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]  # 손가락 끝점
            
            # 손이 턱 부근에 있는지 확인
            for point in hand_points:
                if abs(point.y - jaw_y) < 0.05 and abs(point.x - jaw_x) < 0.1:
                    return True
            
            return False
        except Exception as e:
            print(f"손-턱 겹침 확인 중 오류: {e}")
            return False
    
    def analyze_hand_gesture(self, hand_landmarks, face_landmarks=None):
        """MediaPipe 핸드 랜드마크를 분석하여 제스처 인식"""
        try:
            # 손바닥 중심 좌표
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            # 턱 괴기 감지 (고민 중)
            if face_landmarks and self.is_hand_overlapping_jaw(hand_landmarks, face_landmarks):
                if not self.is_thinking:
                    self.is_thinking = True
                    self.thinking_start_time = time.time()
                    print("턱 괴기 시작: 시간 측정 시작")
                elif time.time() - self.thinking_start_time >= THINKING_DURATION:
                    if 'thinking' not in self.detected_states[-3:]:  # 최근 3개 상태에 없는 경우에만 추가
                        self.detected_states.append('thinking')
                        print(f"턱 괴기 {THINKING_DURATION}초 이상 유지: 고민 중 상태")
            else:
                # 턱 괴기 중단
                self.is_thinking = False
                self.thinking_start_time = None
                
        except Exception as e:
            print(f"손 제스처 분석 중 오류: {e}")
    
    def detect_facial_states(self, face, frame_shape):
        """InsightFace로 얼굴 상태 감지"""
        try:
            ih, iw = frame_shape[:2]  # 프레임 높이, 너비
            
            # 얼굴 크기 계산
            box = face.bbox.astype(np.int32)
            face_width = box[2] - box[0]
            face_height = box[3] - box[1]
            face_size = face_width * face_height
            
            # 초기 얼굴 크기 설정
            if self.initial_face_size == 0:
                self.initial_face_size = face_size
            
            # 얼굴 크기 변화 감지 (지루함)
            size_ratio = face_size / self.initial_face_size if self.initial_face_size > 0 else 1.0
            if size_ratio < BORED_THRESHOLD:  # 0.8배 이하 = 20% 이상 작아짐
                self.detected_states.append('bored')
                print(f"얼굴 크기 감소 감지: 지루함 상태 (현재 크기 비율: {size_ratio:.2f})")
            
            # 랜드마크 분석 (106개 포인트)
            if hasattr(face, 'landmark_2d_106'):
                landmarks = face.landmark_2d_106
                
                # 미간 거리 계산
                left_eyebrow = landmarks[39]  # 왼쪽 눈썹 안쪽
                right_eyebrow = landmarks[42]  # 오른쪽 눈썹 안쪽
                eyebrow_distance = np.linalg.norm(right_eyebrow - left_eyebrow)
                
                # 초기 미간 거리 설정
                if self.initial_eyebrow_distance == 0:
                    self.initial_eyebrow_distance = eyebrow_distance
                
                # 미간 거리 변화 감지 (불만족)
                if self.initial_eyebrow_distance > 0:
                    eyebrow_ratio = eyebrow_distance / self.initial_eyebrow_distance
                    if eyebrow_ratio < EYEBROW_THRESHOLD:  # 0.9배 이하 = 10% 이상 짧아짐
                        self.detected_states.append('dissatisfied')
                        print(f"미간 거리 감소 감지: 불만족 상태 (현재 비율: {eyebrow_ratio:.2f})")
                
                # 입 위치 계산
                mouth_top = landmarks[52]  # 윗입술 중앙
                mouth_bottom = landmarks[58]  # 아랫입술 중앙
                mouth_height = np.mean([mouth_top[1], mouth_bottom[1]])  # y좌표 평균
                
                # 초기 입 높이 설정
                if self.initial_mouth_height == 0:
                    self.initial_mouth_height = mouth_height
                
                # 입 높이 변화 감지 (불만족)
                if self.initial_mouth_height > 0:
                    mouth_shift = self.initial_mouth_height - mouth_height  # 양수면 위로 올라감
                    if mouth_shift > MOUTH_UP_THRESHOLD:  # 1cm 이상 올라감
                        self.detected_states.append('dissatisfied')
                        print(f"입 높이 상승 감지: 불만족 상태 (현재 변화: {mouth_shift:.2f})")
                
                # 하품 감지 (입 크게 벌림)
                mouth_open_height = np.linalg.norm(mouth_bottom - mouth_top)
                mouth_ratio = mouth_open_height / face_height
                
                if mouth_ratio > YAWN_THRESHOLD:  # 입이 크게 벌어짐
                    if not self.is_yawning:
                        self.is_yawning = True
                        self.yawn_start_time = time.time()
                        print("하품 시작: 시간 측정 시작")
                    elif time.time() - self.yawn_start_time >= YAWN_DURATION:
                        if 'tired' not in self.detected_states[-3:]:  # 최근 3개 상태에 없는 경우에만 추가
                            self.detected_states.append('tired')
                            print(f"하품 {YAWN_DURATION}초 이상 유지: 피곤함 상태")
                else:
                    # 하품 중단
                    self.is_yawning = False
                    self.yawn_start_time = None
                
        except Exception as e:
            print(f"얼굴 상태 감지 중 오류: {e}")
    
    def update_webcam(self):
        """웹캠 프레임 업데이트 및 처리"""
        if not self.is_running or not self.webcam_available:
            return
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("웹캠에서 프레임을 읽을 수 없습니다.")
                self.status_text.config(text="오류: 웹캠에서 프레임을 읽을 수 없습니다.")
                # 다음 프레임 처리 계속 시도
                if self.is_running:
                    self.webcam_label.after(10, self.update_webcam)
                return
            
            # 화면 뒤집기 (거울 효과)
            frame = cv2.flip(frame, 1)
            
            # 캡처 기능이 활성화되어 있으면 프레임 저장
            if self.capture_enabled:
                self.capture_frame(frame)
            
            # InsightFace 얼굴 감지
            face_detected = False
            current_face = None
            
            if self.insightface_available:
                try:
                    # InsightFace로 얼굴 감지
                    faces = self.face_app.get(frame)
                    
                    if len(faces) > 0:
                        face_detected = True
                        current_face = faces[0]  # 첫 번째 얼굴
                        
                        # 얼굴 영역 표시
                        box = current_face.bbox.astype(np.int32)
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        
                        # 랜드마크 표시
                        if hasattr(current_face, 'landmark_2d_106'):
                            landmark = current_face.landmark_2d_106.astype(np.int32)
                            for i in range(landmark.shape[0]):
                                cv2.circle(frame, (landmark[i][0], landmark[i][1]), 1, (0, 0, 255), 2)
                        
                        # 얼굴 상태 분석
                        self.detect_facial_states(current_face, frame.shape)
                        
                except Exception as e:
                    print(f"InsightFace 분석 중 오류: {e}")
            
            # MediaPipe 손 감지
            hands_detected = False
            if self.mediapipe_available:
                try:
                    # RGB로 변환 (MediaPipe는 RGB 형식 요구)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(rgb_frame)
                    
                    if results.multi_hand_landmarks:
                        hands_detected = True
                        self.hand_status.config(text="감지됨", fg='#2ecc71')
                        
                        for hand_landmarks in results.multi_hand_landmarks:
                            # 손 랜드마크 그리기
                            self.mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style())
                            
                            # 손 제스처 분석 (얼굴과 손의 상호작용)
                            if face_detected and current_face and hasattr(current_face, 'landmark_2d_106'):
                                self.analyze_hand_gesture(hand_landmarks, current_face)
                            else:
                                self.analyze_hand_gesture(hand_landmarks)
                    else:
                        self.hand_status.config(text="감지되지 않음", fg='#e74c3c')
                        # 손이 감지되지 않으면 턱 괴기 상태 해제
                        self.is_thinking = False
                        self.thinking_start_time = None
                        
                except Exception as e:
                    print(f"MediaPipe Hands 분석 중 오류: {e}")
            
            # 얼굴 감지 상태 업데이트
            self.face_detected = face_detected
            
            try:
                if face_detected:
                    self.face_status.config(text="감지됨", fg='#2ecc71')
                    
                    if self.mode == 'waiting':
                        self.mode = 'studying'
                        self.mode_label.config(text="공부 시간", fg='#2ecc71')
                        print("얼굴 감지됨! 공부 시간을 시작합니다.")
                        self.status_text.config(text="얼굴 감지됨! 공부 시간을 시작합니다.")
                        # 얼굴이 감지되면 기본 비디오 재생 (별도 스레드에서)
                        self.root.after(100, lambda: self.play_video('default'))
                else:
                    self.face_status.config(text="감지되지 않음", fg='#e74c3c')
                    print("얼굴이 감지되지 않습니다.")
                    self.status_text.config(text="얼굴이 감지되지 않습니다. 카메라를 확인하세요.")
            except Exception as e:
                print(f"얼굴 감지 상태 업데이트 오류: {e}")
            
            # 상태 정보 표시
            self.draw_info(frame)
            
            # 이미지를 Tkinter에 표시하기 위해 변환
            try:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                
                # 이미지 크기 조정
                webcam_width = self.webcam_frame.winfo_width()
                webcam_height = self.webcam_frame.winfo_height()
                
                if webcam_width > 1 and webcam_height > 1:  # 유효한 크기인 경우에만
                    img = img.resize((webcam_width, webcam_height), Image.LANCZOS)
                
                imgtk = ImageTk.PhotoImage(image=img)
                self.webcam_label.imgtk = imgtk
                self.webcam_label.configure(image=imgtk)
            except Exception as e:
                print(f"GUI 업데이트 오류: {e}")
            
            # 다음 프레임 처리를 위한 재귀 호출
            if self.is_running:
                self.webcam_label.after(10, self.update_webcam)
        except Exception as e:
            print(f"웹캠 업데이트 전역 오류: {e}")
            # 오류 발생해도 계속 실행 시도
            if self.is_running:
                self.webcam_label.after(100, self.update_webcam)  # 오류 시 약간 더 긴 간격으로 재시도

# 메인 애플리케이션 실행
if __name__ == "__main__":
    monitor = StudyMoodMonitor()
    monitor.root.mainloop()