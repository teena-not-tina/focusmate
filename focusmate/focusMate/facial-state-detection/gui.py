
"""
GUI module for facial state detection system.
Handles the graphical user interface and visual components.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
from config import *

class FacialStateGUI:
    def __init__(self, root=None):
        """GUI 클래스 초기화"""
        # GUI 루트 (외부에서 제공된 경우)
        self.external_root = root is not None
        self.root = root if root else tk.Tk()
        
        # GUI 설정
        self.setup_gui()
        
        # 웹캠 프레임 변수
        self.current_frame = None
        self.webcam_frame = None
        self.processed_frame = None
        self.tk_img = None
        
        # 이벤트 콜백 함수
        self.on_start_stop = None
        self.on_collection_toggle = None
        self.on_analysis_toggle = None
        self.on_train_model = None
        self.on_capture_screenshot = None
        self.on_state_selected = None
        self.on_max_samples_selected = None
        self.on_open_data_folder = None
        self.on_closing = None
    
    def setup_gui(self):
        """GUI 설정"""
        self.root.title("얼굴 및 손 랜드마크 수집 및 상태 추정")
        self.root.configure(bg=GUI_COLORS['bg_dark'])
        self.root.geometry("1200x700")
        # 창 최소 크기 설정
        self.root.minsize(1000, 600)
        # 창 크기 조절 가능, 하지만 최소 크기는 유지됨
        self.root.resizable(True, True)
        
        # 메인 프레임
        main_frame = tk.Frame(self.root, bg=GUI_COLORS['bg_dark'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 왼쪽 패널 (웹캠 화면) - 비율 설정
        webcam_panel = tk.Frame(main_frame, bg=GUI_COLORS['bg_mid'], 
                              highlightbackground=GUI_COLORS['highlight'], 
                              highlightthickness=2)
        webcam_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 오른쪽 패널 (컨트롤 패널) - 고정 너비
        control_panel = tk.Frame(main_frame, bg=GUI_COLORS['bg_mid'], width=350)
        control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 0))
        control_panel.pack_propagate(False)  # 크기 고정
        
        # 웹캠 제목
        webcam_title = tk.Label(webcam_panel, text="실시간 웹캠 화면", 
                              font=("Helvetica", 14, "bold"), 
                              bg=GUI_COLORS['bg_mid'], fg=GUI_COLORS['text'])
        webcam_title.pack(pady=5)
        
        # 웹캠 디스플레이를 포함할 프레임 (크기 제한)
        webcam_container = tk.Frame(webcam_panel, bg="black")
        webcam_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 실제 비율 유지하기 위한 내부 프레임
        aspect_frame = tk.Frame(webcam_container, bg="black")
        aspect_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # 웹캠 표시 영역
        self.webcam_display = tk.Label(aspect_frame, bg="black")
        self.webcam_display.pack()
        
        # 상태 표시줄
        status_frame = tk.Frame(webcam_panel, bg=GUI_COLORS['bg_mid'])
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = tk.Label(status_frame, text="준비됨", 
                                   font=("Helvetica", 10), 
                                   bg=GUI_COLORS['bg_mid'], fg=GUI_COLORS['text'])
        self.status_label.pack(side=tk.LEFT)
        
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
        self.root.bind('<Escape>', lambda e: self.handle_closing())
        self.root.bind('s', lambda e: self.handle_start_stop())
        self.root.bind('c', lambda e: self.handle_collection_toggle())
        self.root.bind('a', lambda e: self.handle_analysis_toggle())
        self.root.bind('t', lambda e: self.handle_train_model())
    
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
        self.state_combo.bind("<<ComboboxSelected>>", lambda e: self.handle_state_selected())
        
        # 수집 시작/중지 버튼
        self.collection_btn = tk.Button(collection_frame, text="수집 시작", width=12,
                                      bg=GUI_COLORS['green'], fg="white", font=("Helvetica", 10, "bold"),
                                      command=self.handle_collection_toggle)
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
        self.max_samples_combo.bind("<<ComboboxSelected>>", lambda e: self.handle_max_samples_selected())
    
    def create_detection_section(self, parent):
        """상태 감지 섹션 생성"""
        detection_frame = tk.LabelFrame(parent, text="상태 감지", font=("Helvetica", 12, "bold"),
                                      bg=GUI_COLORS['bg_mid'], fg=GUI_COLORS['text'])
        detection_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # 상태 감지 시작/중지 버튼
        self.analysis_btn = tk.Button(detection_frame, text="감지 시작", width=12,
                                   bg=GUI_COLORS['highlight'], fg="white", font=("Helvetica", 10, "bold"),
                                   command=self.handle_analysis_toggle)
        self.analysis_btn.pack(pady=5)
        
        # 모델 학습 버튼
        model_frame = tk.Frame(detection_frame, bg=GUI_COLORS['bg_mid'])
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.training_btn = tk.Button(model_frame, text="모델 학습", width=12,
                                    bg=GUI_COLORS['orange'], fg="white", font=("Helvetica", 10, "bold"),
                                    command=self.handle_train_model)
        self.training_btn.pack(side=tk.LEFT, padx=5)
        
        # 모델 상태
        self.model_status_label = tk.Label(model_frame, text="모델 없음", bg=GUI_COLORS['bg_mid'],
                                        fg=GUI_COLORS['red'], font=("Helvetica", 10))
        self.model_status_label.pack(side=tk.LEFT, padx=5)
        
        # 캡처 폴더 열기 버튼
        folder_btn = tk.Button(detection_frame, text="데이터 폴더 열기", width=15,
                             bg=GUI_COLORS['purple'], fg="white", font=("Helvetica", 10),
                             command=self.handle_open_data_folder)
        folder_btn.pack(pady=5)
    
    def create_control_section(self, parent):
        """제어 버튼 섹션 생성"""
        control_frame = tk.Frame(parent, bg=GUI_COLORS['bg_mid'])
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # 시작/중지 버튼
        self.start_stop_btn = tk.Button(control_frame, text="시작", width=12, height=2,
                                     bg=GUI_COLORS['green'], fg="white", font=("Helvetica", 12, "bold"),
                                     command=self.handle_start_stop)
        self.start_stop_btn.pack(side=tk.LEFT, padx=5)
        
        # 캡처 버튼
        self.capture_btn = tk.Button(control_frame, text="스크린샷", width=12, height=2,
                                  bg=GUI_COLORS['highlight'], fg="white", font=("Helvetica", 12, "bold"),
                                  command=self.handle_capture_screenshot)
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        # 종료 버튼
        exit_btn = tk.Button(control_frame, text="종료", width=12, height=2,
                           bg=GUI_COLORS['red'], fg="white", font=("Helvetica", 12, "bold"),
                           command=self.handle_closing)
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
    
    def update_webcam_display(self, frame):
        """웹캠 화면 업데이트"""
        try:
            # 원본 프레임 크기 가져오기
            h, w = frame.shape[:2]
            
            # 이미지 변환
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_img = Image.fromarray(display_frame)
            
            # 웹캠 컨테이너 크기 가져오기
            container_width = self.webcam_display.master.master.winfo_width()
            container_height = self.webcam_display.master.master.winfo_height()
            
            # 컨테이너 크기가 유효한지 확인
            if container_width < 10 or container_height < 10:
                container_width = 640  # 기본 너비
                container_height = 480  # 기본 높이
            
            # 종횡비 유지하면서 크기 조절
            img_ratio = w / h
            container_ratio = container_width / container_height
            
            if img_ratio > container_ratio:
                # 너비에 맞춤
                new_width = min(container_width, 800)  # 최대 너비 제한
                new_height = int(new_width / img_ratio)
            else:
                # 높이에 맞춤
                new_height = min(container_height, 600)  # 최대 높이 제한
                new_width = int(new_height * img_ratio)
            
            # 이미지 크기 조정
            display_img = display_img.resize((new_width, new_height), Image.LANCZOS)
            
            # 내부 프레임 크기 조정
            self.webcam_display.master.config(width=new_width, height=new_height)
            
            # Tkinter 이미지 객체로 변환
            self.tk_img = ImageTk.PhotoImage(image=display_img)
            
            # 레이블에 이미지 설정
            self.webcam_display.config(image=self.tk_img, width=new_width, height=new_height)
            self.webcam_display.image = self.tk_img  # 참조 유지
            
        except Exception as e:
            print(f"GUI 업데이트 중 오류: {e}")
    
    def update_status(self, message, warning=False):
        """상태 표시줄 업데이트"""
        self.status_label.config(text=message, fg=GUI_COLORS['red'] if warning else GUI_COLORS['text'])
        print(message)
    
    def update_confidence(self, confidence=0):
        """신뢰도 막대 업데이트"""
        self.confidence_bar["value"] = confidence
        self.confidence_label.config(text=f"{confidence}%")
    
    def update_face_status(self, detected):
        """얼굴 감지 상태 업데이트"""
        status = "감지됨" if detected else "감지되지 않음"
        color = GUI_COLORS['green'] if detected else GUI_COLORS['red']
        self.face_status_label.config(text=status, fg=color)
    
    def update_hand_status(self, detected):
        """손 감지 상태 업데이트"""
        status = "감지됨" if detected else "감지되지 않음"
        color = GUI_COLORS['green'] if detected else GUI_COLORS['red']
        self.hand_status_label.config(text=status, fg=color)
    
    def update_detected_state(self, state_code, duration=0.0):
        """감지된 상태 업데이트"""
        state_text = STATE_KOREAN.get(state_code, "알 수 없음")
        color = self.get_state_color_tk(state_code)
        self.detected_state_label.config(text=state_text, fg=color)
        self.duration_label.config(text=f"{duration:.1f}초")
    
    def update_model_status(self, model_loaded):
        """모델 상태 업데이트"""
        status = "모델 로드됨" if model_loaded else "모델 없음"
        color = GUI_COLORS['green'] if model_loaded else GUI_COLORS['red']
        self.model_status_label.config(text=status, fg=color)
    
    def update_sample_count(self, count):
        """샘플 수 업데이트"""
        self.sample_count_label.config(text=str(count))
    
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
    
    def set_collection_button_state(self, is_collecting):
        """수집 버튼 상태 업데이트"""
        if is_collecting:
            self.collection_btn.config(text="수집 중지", bg=GUI_COLORS['red'])
        else:
            self.collection_btn.config(text="수집 시작", bg=GUI_COLORS['green'])
    
    def set_analysis_button_state(self, is_analyzing):
        """분석 버튼 상태 업데이트"""
        if is_analyzing:
            self.analysis_btn.config(text="감지 중지", bg=GUI_COLORS['red'])
        else:
            self.analysis_btn.config(text="감지 시작", bg=GUI_COLORS['highlight'])
    
    def set_start_stop_button_state(self, is_running):
        """시작/중지 버튼 상태 업데이트"""
        if is_running:
            self.start_stop_btn.config(text="중지", bg=GUI_COLORS['red'])
        else:
            self.start_stop_btn.config(text="시작", bg=GUI_COLORS['green'])
    
    def set_training_button_state(self, is_training):
        """학습 버튼 상태 업데이트"""
        if is_training:
            self.training_btn.config(text="학습 중...", state=tk.DISABLED)
        else:
            self.training_btn.config(text="모델 학습", state=tk.NORMAL)
    
    def handle_start_stop(self):
        """시작/중지 버튼 이벤트 처리"""
        if self.on_start_stop:
            self.on_start_stop()
    
    def handle_collection_toggle(self):
        """데이터 수집 시작/중지 버튼 이벤트 처리"""
        if self.on_collection_toggle:
            self.on_collection_toggle()
    
    def handle_analysis_toggle(self):
        """상태 감지 시작/중지 버튼 이벤트 처리"""
        if self.on_analysis_toggle:
            self.on_analysis_toggle()
    
    def handle_train_model(self):
        """모델 학습 버튼 이벤트 처리"""
        if self.on_train_model:
            self.on_train_model()
    
    def handle_capture_screenshot(self):
        """스크린샷 버튼 이벤트 처리"""
        if self.on_capture_screenshot:
            self.on_capture_screenshot()
    
    def handle_state_selected(self):
        """상태 선택 콤보박스 이벤트 처리"""
        if self.on_state_selected:
            selected = self.state_combo.get()
            state_code = int(selected[0])  # 첫 글자(숫자)만 추출
            self.on_state_selected(state_code)
    
    def handle_max_samples_selected(self):
        """최대 샘플 수 콤보박스 이벤트 처리"""
        if self.on_max_samples_selected:
            try:
                max_samples = int(self.max_samples_combo.get())
                self.on_max_samples_selected(max_samples)
            except ValueError:
                pass
    
    def handle_open_data_folder(self):
        """데이터 폴더 열기 버튼 이벤트 처리"""
        if self.on_open_data_folder:
            self.on_open_data_folder()
    
    def handle_closing(self):
        """창 닫기 이벤트 처리"""
        if self.on_closing:
            self.on_closing()
        
        if not self.external_root:
            self.root.destroy()
    
    def show_webcam_error(self):
        """웹캠 오류 메시지 표시"""
        error_msg = tk.Label(self.webcam_display, text="웹캠을 사용할 수 없습니다.\n카메라 연결을 확인하세요.",
                             fg="white", bg="black", font=("Helvetica", 16))
        error_msg.pack(expand=True)
    
    def show_message_box(self, title, message, is_error=False):
        """메시지 박스 표시"""
        from tkinter import messagebox
        
        if is_error:
            messagebox.showerror(title, message)
        else:
            messagebox.showinfo(title, message)
    
    def run(self):
        """GUI 실행"""
        if not self.external_root:
            self.root.mainloop()