import cv2
import numpy as np
import time
import threading
import os
from tkinter import *
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import webbrowser
import mediapipe as mp
from datetime import datetime

# YouTube 링크 (상태별 음악)
MUSIC_LINKS = {
    'default': "https://www.youtube.com/watch?v=APOQv1oV0EY",  # 백색소음
    'bored': "https://www.youtube.com/watch?v=QZbuj3RJcjI",    # 베타파 소리
    'tired': "https://www.youtube.com/watch?v=SXRK5hRHYWY",    # 활기찬 음악
    'thinking': "https://www.youtube.com/watch?v=3RxKYHLNWMc", # 감마파 소리
    'satisfied': "https://www.youtube.com/watch?v=ZbZSe6N_BXs", # 환희 가득한 음악
    'dissatisfied': "https://www.youtube.com/watch?v=lFcSrYw-ARY" # 릴렉스 음악
}

# 시간 설정 (초 단위)
STUDY_TIME = 25 * 60  # 25분
BREAK_TIME = 5 * 60   # 5분
ANALYSIS_INTERVAL = 5  # 매 5초마다 분석
CAPTURE_INTERVAL = 1   # 1초마다 캡처

# 얼굴 및 손 인식 설정
class StudyMoodMonitor:
    def __init__(self):
        # 상태 감지 변수
        self.detected_states = []
        self.dominant_state = 'default'
        self.initial_face_size = 0
        self.yawn_start_time = None
        self.is_yawning = False
        self.last_analysis_time = 0
        
        # 손 제스처 관련 변수
        self.thumb_up_detected = False
        self.clapping_detected = False
        self.hand_on_face_detected = False
        
        # 실행 상태 변수
        self.is_running = False
        self.face_detected = False
        self.study_time_remaining = STUDY_TIME
        self.break_time_remaining = BREAK_TIME
        self.mode = 'waiting'  # waiting, studying, break
        
        # 타이머 스레드
        self.timer_thread = None
        
        # 음악 재생 제어 변수
        self.music_playing = False
        self.open_browser_thread = None
        
        # 캡처 관련 변수
        self.capture_enabled = False
        self.capture_folder = "captures"
        self.last_capture_time = 0
        
        # 캡처 폴더 생성
        if not os.path.exists(self.capture_folder):
            os.makedirs(self.capture_folder)
            print(f"캡처 폴더 생성: {self.capture_folder}")
        
        # insightface 및 mediapipe tasks API 사용 여부 설정
        self.insightface_available = False
        self.mediapipe_tasks_available = False
            
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
        
        # MediaPipe Face Detection 설정 추가
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5)
            self.mp_face_mesh = mp.solutions.face_mesh  
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
            print("MediaPipe Face Detection 초기화 성공")
        except Exception as e:
            print(f"MediaPipe Face Detection 초기화 중 오류: {e}")
            
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
        
        # 상태 설명
        description_frame = Frame(control_frame, bg='#34495e', padx=10, pady=10)
        description_frame.pack(fill=BOTH, expand=True)
        
        Label(description_frame, text="감지 가능한 상태", 
             font=("Helvetica", 12, "bold"), bg='#34495e', fg='white').pack(anchor=W)
        
        # 상태 설명 목록
        state_desc = Text(description_frame, wrap=WORD, height=10, bg='#2c3e50', fg='white',
                          font=("Helvetica", 10), padx=10, pady=10)
        state_desc.pack(fill=BOTH, expand=True, pady=5)
        state_desc.insert(END, "• 지루함: 얼굴이 30% 이상 작아짐\n")
        state_desc.insert(END, "• 피곤함: 하품 (입 크게 벌림)\n")
        state_desc.insert(END, "• 고민 중: 턱 괴기, 얼굴 감싸기\n")
        state_desc.insert(END, "• 만족: 손뼉, 엄지 올리기\n")
        state_desc.insert(END, "• 불만족: 미간 주름, 입 비뚤어짐\n")
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
                    if os.uname().sysname == 'Darwin':  # macOS
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
        
        self.reset_states()
        self.mode = 'waiting'
    
    def on_closing(self):
        """프로그램 종료"""
        self.is_running = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'hands'):
            self.hands.close()
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        self.root.destroy()
    
    def reset_states(self):
        """상태 감지 변수 초기화"""
        self.detected_states = []
        self.dominant_state = 'default'
        self.initial_face_size = 0
        self.face_detected = False
        self.yawn_start_time = None
        self.is_yawning = False
        self.thumb_up_detected = False
        self.clapping_detected = False
        self.hand_on_face_detected = False
        self.study_time_remaining = STUDY_TIME
        self.break_time_remaining = BREAK_TIME
        self.music_playing = False
    
    def open_browser_in_thread(self, url):
        """별도 스레드에서 브라우저 열기"""
        try:
            webbrowser.open(url)
            print(f"음악 재생 URL 열림: {url}")
        except Exception as e:
            print(f"음악 재생 중 오류: {e}")
        finally:
            self.music_playing = False
    
    def play_music(self, state):
        """YouTube에서 상태에 해당하는 음악 재생"""
        try:
            # 이미 음악이 재생 중이면 중복 실행 방지
            if self.music_playing:
                return

            self.music_playing = True
            url = MUSIC_LINKS.get(state, MUSIC_LINKS['default'])
            
            # 새 스레드에서 브라우저 열기
            self.open_browser_thread = threading.Thread(target=self.open_browser_in_thread, args=(url,))
            self.open_browser_thread.daemon = True
            self.open_browser_thread.start()
            
            print(f"음악 재생 요청: {state} 상태에 맞는 음악")
        except Exception as e:
            print(f"음악 재생 요청 중 오류: {e}")
            self.music_playing = False
    
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
                    self.play_music(self.dominant_state)
            
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
    
    def analyze_hand_gesture(self, hand_landmarks):
        """MediaPipe 핸드 랜드마크를 분석하여 제스처 인식"""
        try:
            # 손가락 끝점 좌표
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
            
            # 손바닥 중심 좌표
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            # 손가락 밑 부분 좌표
            thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
            index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            ring_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
            pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
            
            # 엄지 올리기 감지 (만족)
            if (thumb_tip.y < thumb_mcp.y) and (index_tip.y > index_mcp.y) and (middle_tip.y > middle_mcp.y):
                self.thumb_up_detected = True
                self.detected_states.append('satisfied')
                print("엄지 올리기 감지: 만족 상태")
            
            # 손뼉 감지 (만족) - 양손이 필요하므로 추가 로직 필요
            # 단순화를 위해 손이 빠르게 움직이는 것으로 추정
            
            # 턱 괴기, 얼굴 감싸기 감지 (고민 중) - 손과 얼굴 위치 비교 필요
            # 단순화를 위해 손이 상단에 있으면 고민 중으로 추정
            if (wrist.y > 0.6) and (thumb_tip.y < 0.4):
                self.hand_on_face_detected = True
                self.detected_states.append('thinking')
                print("턱 괴기 감지: 고민 중 상태")
                
        except Exception as e:
            print(f"손 제스처 분석 중 오류: {e}")
    
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
            
            # RGB로 변환 (MediaPipe는 RGB 형식 요구)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 이미지 크기 가져오기
            ih, iw, _ = frame.shape  # 높이, 너비, 채널

            # 얼굴 감지 - 초기값 설정
            face_detected = False
            
            # 얼굴 감지
            face_results = self.face_detection.process(rgb_frame)
            
            if face_results.detections:
                face_detected = True
                
                for detection in face_results.detections:
                    # 얼굴 영역 그리기
                    self.mp_drawing.draw_detection(frame, detection)
                    
                    # 얼굴 크기 계산
                    bboxC = detection.location_data.relative_bounding_box
                    # 위치 변수가 유효한지 확인
                    x, y, w, h = 0, 0, 0, 0
                    
                    try:
                        x = int(bboxC.xmin * iw)
                        y = int(bboxC.ymin * ih)
                        w = int(bboxC.width * iw)
                        h = int(bboxC.height * ih)
                        
                        # 얼굴 크기 변화 감지 (지루함: 작아짐)
                        face_size = w * h
                        if self.initial_face_size == 0:
                            self.initial_face_size = face_size
                        
                        size_ratio = face_size / self.initial_face_size if self.initial_face_size > 0 else 1.0
                        if size_ratio < 0.7:  # 30% 이상 작아지면
                            self.detected_states.append('bored')
                            print("얼굴 크기 감소 감지: 지루함 상태")
                    except Exception as e:
                        print(f"얼굴 좌표 계산 오류: {e}")
            
            # 얼굴 메시 감지 (표정 분석)
            try:
                mesh_results = self.face_mesh.process(rgb_frame)
                
                if mesh_results and mesh_results.multi_face_landmarks:
                    for face_landmarks in mesh_results.multi_face_landmarks:
                        # 얼굴 메시 그리기
                        self.mp_drawing.draw_landmarks(
                            frame,
                            face_landmarks,
                            self.mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None)
                        
                        try:
                            # 입 랜드마크 추출 (하품 감지)
                            lips_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]  # 입 주변 랜드마크
                            lips_coordinates = [(int(face_landmarks.landmark[i].x * iw), 
                                                int(face_landmarks.landmark[i].y * ih)) 
                                                for i in lips_indices]
                            
                            # 입 세로 길이 계산 (하품 감지)
                            upper_lip = min([coord[1] for coord in lips_coordinates])
                            lower_lip = max([coord[1] for coord in lips_coordinates])
                            mouth_height = lower_lip - upper_lip
                            face_height = h if h > 0 else 1  # 얼굴 감지된 높이, 0이면 1로 대체
                            
                            mouth_ratio = mouth_height / face_height
                            if mouth_ratio > 0.25:  # 입이 크게 벌어짐 (하품)
                                self.detected_states.append('tired')
                                print("하품 감지: 피곤함 상태")
                            
                            # 눈썹 랜드마크 추출 (불만족 감지)
                            eyebrow_indices = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
                            eyebrow_coordinates = [(int(face_landmarks.landmark[i].x * iw), 
                                                   int(face_landmarks.landmark[i].y * ih)) 
                                                   for i in eyebrow_indices]
                            
                            # 눈썹 거리 계산 (불만족 감지)
                            left_eyebrow_center = (sum([coord[0] for coord in eyebrow_coordinates[:5]]) / 5,
                                                  sum([coord[1] for coord in eyebrow_coordinates[:5]]) / 5)
                            right_eyebrow_center = (sum([coord[0] for coord in eyebrow_coordinates[5:]]) / 5,
                                                   sum([coord[1] for coord in eyebrow_coordinates[5:]]) / 5)
                            
                            eyebrow_distance = ((left_eyebrow_center[0] - right_eyebrow_center[0])**2 + 
                                               (left_eyebrow_center[1] - right_eyebrow_center[1])**2)**0.5
                            
                            if eyebrow_distance < 0.4 * w and w > 0:  # 눈썹이 가까워짐 (불만족)
                                self.detected_states.append('dissatisfied')
                                print("눈썹 가까워짐 감지: 불만족 상태")
                        except Exception as e:
                            print(f"얼굴 특징 분석 오류: {e}")
            except Exception as e:
                print(f"Face Mesh 처리 오류: {e}")
            
            # 손 감지 및 분석 (MediaPipe 사용)
            hands_detected = False
            if self.mediapipe_available:
                try:
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
                            
                            # 손 제스처 분석
                            self.analyze_hand_gesture(hand_landmarks)
                    else:
                        self.hand_status.config(text="감지되지 않음", fg='#e74c3c')
                        
                except Exception as e:
                    print(f"MediaPipe Hands 분석 중 오류: {e}")
            
            # 얼굴 감지 상태 업데이트
            try:
                if face_detected:
                    if not self.face_detected:
                        self.face_detected = True
                        self.face_status.config(text="감지됨", fg='#2ecc71')
                        
                        if self.mode == 'waiting':
                            self.mode = 'studying'
                            self.mode_label.config(text="공부 시간", fg='#2ecc71')
                            print("얼굴 감지됨! 공부 시간을 시작합니다.")
                            self.status_text.config(text="얼굴 감지됨! 공부 시간을 시작합니다.")
                            # 얼굴이 감지되면 기본 백색소음 음악 재생 (별도 스레드에서)
                            self.root.after(100, lambda: self.play_music('default'))
                else:
                    if self.face_detected:
                        self.face_detected = False
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