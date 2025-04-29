import cv2
import numpy as np
import time
import threading
import os
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import webbrowser
import mediapipe as mp
# insightface 및 필요한 모듈 가져오기
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import mediapipe as mp
print(mp.__version__)

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
        
        # InsightFace 설정
        try:
            # InsightFace 분석기 초기화
            self.face_app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
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
            
        # MediaPipe Tasks API 설정 추가
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            # 모델 파일 경로 설정
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'hand_landmarker.task')
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
            self.mediapipe_tasks_available = True
            print("MediaPipe Tasks API 초기화 성공")
        except Exception as e:
            print(f"MediaPipe Tasks API 초기화 중 오류: {e}")
            self.mediapipe_tasks_available = False
           
        
        # 웹캠 설정
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("웹캠을 열 수 없습니다. 카메라가 연결되어 있는지 확인하세요.")
                self.webcam_available = False
            else:
                self.webcam_available = True
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
    
    def update_webcam(self):
        """웹캠 프레임 업데이트 및 처리"""
        if not self.is_running or not self.webcam_available:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            print("웹캠에서 프레임을 읽을 수 없습니다.")
            self.status_text.config(text="오류: 웹캠에서 프레임을 읽을 수 없습니다.")
            return
        
        # 화면 뒤집기 (거울 효과)
        frame = cv2.flip(frame, 1)
        
        # 얼굴 감지 및 랜드마크 분석 (InsightFace 사용)
        face_detected = False
        if self.insightface_available:
            try:
                # InsightFace로 얼굴 감지
                faces = self.face_app.get(frame)
                
                if len(faces) > 0:
                    face_detected = True
                    face = faces[0]  # 첫 번째 얼굴
                    
                    # 얼굴 영역 표시
                    box = face.bbox.astype(np.int32)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    
                    # 랜드마크 표시
                    if hasattr(face, 'landmark_2d_106'):
                        landmark = face.landmark_2d_106.astype(np.int32)
                        for i in range(landmark.shape[0]):
                            cv2.circle(frame, (landmark[i][0], landmark[i][1]), 1, (0, 0, 255), 2)
                    
                    # 얼굴 특징 분석
                    self.analyze_face_state(face)
                    
            except Exception as e:
                print(f"InsightFace 분석 중 오류: {e}")
        
        # 손 감지 및 분석 (MediaPipe 사용)
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
                        
                        # 손 제스처 분석
                        self.analyze_hand_gesture(hand_landmarks)
                else:
                    self.hand_status.config(text="감지되지 않음", fg='#e74c3c')
                    
            except Exception as e:
                print(f"MediaPipe Hands 분석 중 오류: {e}")
                
        # 손 감지 및 분석 (MediaPipe Tasks API 사용)
        if hasattr(self, 'mediapipe_tasks_available') and self.mediapipe_tasks_available:
            try:
                # RGB로 변환 (MediaPipe는 RGB 형식 요구)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.hand_landmarker.detect(mp_image)
                
                if detection_result.hand_landmarks:
                    hands_detected = True
                    self.hand_status.config(text="감지됨", fg='#2ecc71')
                    
                    for hand_landmarks in detection_result.hand_landmarks:
                        # 손 랜드마크 그리기
                        self.draw_landmarks_on_image(frame, hand_landmarks)
                        
                        # 손 제스처 분석
                        self.analyze_hand_gesture_tasks(hand_landmarks)
                else:
                    self.hand_status.config(text="감지되지 않음", fg='#e74c3c')
                    
            except Exception as e:
                print(f"MediaPipe Tasks API 분석 중 오류: {e}")
        
        # 얼굴 감지 상태 업데이트
        if face_detected:
            if not self.face_detected:
                self.face_detected = True
                self.face_status.config(text="감지됨", fg='#2ecc71')
                
                if self.mode == 'waiting':
                    self.mode = 'studying'
                    self.mode_label.config(text="공부 시간", fg='#2ecc71')
                    print("얼굴 감지됨! 공부 시간을 시작합니다.")
                    self.status_text.config(text="얼굴 감지됨! 공부 시간을 시작합니다.")
                    # 얼굴이 감지되면 기본 백색소음 음악 재생
                    self.play_music('default')
        else:
            if self.face_detected:
                self.face_detected = False
                self.face_status.config(text="감지되지 않음", fg='#e74c3c')
                print("얼굴이 감지되지 않습니다.")
                self.status_text.config(text="얼굴이 감지되지 않습니다. 카메라를 확인하세요.")
        
        # 상태 정보 표시
        self.draw_info(frame)
        
        # 이미지를 Tkinter에 표시하기 위해 변환
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
        
        # 다음 프레임 처리를 위한 재귀 호출
        if self.is_running:
            self.webcam_label.after(10, self.update_webcam)
    
    def analyze_face_state(self, face):
        """얼굴 특징을 분석하여 상태 예측"""
        try:
            # 얼굴 크기 측정 (처음 감지된 얼굴 크기와 비교)
            box = face.bbox.astype(np.int32)
            face_width = box[2] - box[0]
            face_height = box[3] - box[1]
            face_size = face_width * face_height

            # 초기 얼굴 크기 설정
            if self.initial_face_size == 0:
                self.initial_face_size = face_size

            # 얼굴 크기 변화 감지 (지루함: 작아짐)
            size_ratio = face_size / self.initial_face_size
            if size_ratio < 0.7:  # 30% 이상 작아지면
                self.detected_states.append('bored')

            # 랜드마크를 이용한 표정 분석
            if hasattr(face, 'landmark_2d_106'):
                landmarks = face.landmark_2d_106

                # 입 크게 벌림 감지 (하품 - 피곤함)
                mouth_top = landmarks[52]  # 윗입술 중앙
                mouth_bottom = landmarks[58]  # 아랫입술 중앙
                nose_bottom = landmarks[87]  # 코 아래

                mouth_height = np.linalg.norm(mouth_bottom - mouth_top)
                face_height = np.linalg.norm(landmarks[8] - landmarks[27])  # 턱에서 미간까지

                mouth_ratio = mouth_height / face_height
                if mouth_ratio > 0.2:  # 입이 크게 벌어짐
                    self.detected_states.append('tired')

                # 눈썹 위치로 미간 주름 감지 (불만족)
                left_eyebrow = landmarks[21]  # 왼쪽 눈썹
                right_eyebrow = landmarks[22]  # 오른쪽 눈썹
                nose_bridge = landmarks[27]  # 코 다리

                eyebrow_distance = np.linalg.norm(left_eyebrow - right_eyebrow)
                nose_to_eyebrow_distance = np.linalg.norm(nose_bridge - (left_eyebrow + right_eyebrow) / 2)

                if eyebrow_distance < 0.1 * face_width and nose_to_eyebrow_distance < 0.2 * face_height:
                    self.detected_states.append('dissatisfied')

        except Exception as e:
            print(f"얼굴 상태 분석 중 오류: {e}")