#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import time

class FacialStateGUI:
    def __init__(self):
        # 메인 윈도우 설정
        self.root = tk.Tk()
        self.root.title("얼굴 상태 감지 프로그램")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # 이벤트 콜백 초기화
        self.on_start_stop = None
        self.on_closing = None
        
        # UI 구성요소 생성
        self.create_widgets()
        
        # 윈도우 종료 이벤트 연결
        self.root.protocol("WM_DELETE_WINDOW", self.on_window_close)
        
        # 상태 변수
        self.face_detected_status = False
        
    def create_widgets(self):
        """GUI 레이아웃 구성"""
        # 프레임 생성
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 상단 프레임 (제목 및 설명)
        self.header_frame = ttk.Frame(self.main_frame, padding=5)
        self.header_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(
            self.header_frame, 
            text="얼굴 상태 감지 및 음악 재생 프로그램", 
            font=("Arial", 16, "bold")
        ).pack()
        
        ttk.Label(
            self.header_frame, 
            text="정면 얼굴/얼굴 측면/하품을 감지하여 상황에 맞는 음악을 재생합니다.",
            font=("Arial", 10)
        ).pack(pady=5)
        
        # 컨트롤 프레임 (버튼 등)
        self.control_frame = ttk.Frame(self.main_frame, padding=5)
        self.control_frame.pack(fill=tk.X, pady=5)
        
        # 시작/중지 버튼
        self.start_stop_button = ttk.Button(
            self.control_frame, 
            text="시작",
            width=15,
            command=self.start_stop_clicked
        )
        self.start_stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 상태 표시 라벨
        self.status_label = ttk.Label(
            self.control_frame, 
            text="대기 중...",
            font=("Arial", 10)
        )
        self.status_label.pack(side=tk.LEFT, padx=20, pady=5)
        
        # 얼굴 감지 상태 표시
        self.face_status_label = ttk.Label(
            self.control_frame, 
            text="얼굴 감지 상태: 없음",
            font=("Arial", 10)
        )
        self.face_status_label.pack(side=tk.RIGHT, padx=20, pady=5)
        
        # 웹캠 화면 프레임
        self.webcam_frame = ttk.Frame(self.main_frame, padding=5, relief=tk.GROOVE, borderwidth=2)
        self.webcam_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 웹캠 표시 라벨
        self.webcam_label = ttk.Label(self.webcam_frame)
        self.webcam_label.pack(fill=tk.BOTH, expand=True)
        
        # 정보 프레임 (하단 상태 메시지)
        self.info_frame = ttk.Frame(self.main_frame, padding=5)
        self.info_frame.pack(fill=tk.X, pady=5)
        
        # 상태 메시지 텍스트 상자
        self.status_text = tk.Text(self.info_frame, height=10, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # 스크롤바
        self.status_scrollbar = ttk.Scrollbar(self.info_frame, command=self.status_text.yview)
        self.status_scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.status_text.config(yscrollcommand=self.status_scrollbar.set)
        
        # 상태 텍스트 초기화
        self.status_text.insert(tk.END, "프로그램이 시작되었습니다.\n")
        self.status_text.insert(tk.END, "시작 버튼을 눌러 얼굴 감지를 시작하세요.\n")
        self.status_text.config(state=tk.DISABLED)  # 읽기 전용으로 설정
        
    def start_stop_clicked(self):
        """시작/중지 버튼 클릭 처리"""
        if self.on_start_stop:
            self.on_start_stop()
    
    def set_start_stop_button_state(self, is_running):
        """시작/중지 버튼 상태 설정"""
        if is_running:
            self.start_stop_button.config(text="중지")
        else:
            self.start_stop_button.config(text="시작")
    
    def update_status(self, message, warning=False):
        """상태 메시지 업데이트"""
        # 텍스트 위젯 편집 가능하게 설정
        self.status_text.config(state=tk.NORMAL)
        
        # 현재 시간
        current_time = time.strftime("%H:%M:%S", time.localtime())
        
        # 메시지 포맷팅
        full_message = f"[{current_time}] {message}\n"
        
        # 경고 메시지인 경우 태그 추가
        if warning:
            self.status_text.insert(tk.END, full_message, "warning")
            self.status_text.tag_configure("warning", foreground="red")
        else:
            self.status_text.insert(tk.END, full_message)
        
        # 스크롤을 항상 최신 메시지로
        self.status_text.see(tk.END)
        
        # 다시 읽기 전용으로 설정
        self.status_text.config(state=tk.DISABLED)
        
        # 메인 상태 라벨 업데이트
        self.status_label.config(text=message)
    
    def update_face_status(self, detected):
        """얼굴 감지 상태 업데이트"""
        if detected != self.face_detected_status:
            self.face_detected_status = detected
            if detected:
                self.face_status_label.config(text="얼굴 감지 상태: 감지됨")
            else:
                self.face_status_label.config(text="얼굴 감지 상태: 없음")
    
    def update_webcam_display(self, frame):
        """웹캠 이미지 업데이트"""
        if frame is not None:
            # OpenCV BGR에서 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # PIL 이미지로 변환
            pil_image = Image.fromarray(rgb_frame)
            
            # 라벨 크기에 맞게 이미지 크기 조정
            label_width = self.webcam_label.winfo_width()
            label_height = self.webcam_label.winfo_height()
            
            if label_width > 1 and label_height > 1:  # 유효한 크기인 경우
                pil_image = pil_image.resize((label_width, label_height), Image.LANCZOS)
            
            # Tkinter PhotoImage로 변환
            tk_image = ImageTk.PhotoImage(image=pil_image)
            
            # 라벨에 이미지 표시
            self.webcam_label.config(image=tk_image)
            self.webcam_label.image = tk_image  # 참조 유지
    
    def show_webcam_error(self):
        """웹캠 에러 메시지 표시"""
        self.update_status("웹캠을 사용할 수 없습니다. 연결을 확인하세요.", warning=True)
        
        # 에러 이미지 표시
        error_text = "웹캠을 사용할 수 없습니다\n연결을 확인하세요"
        
        # 검은 배경 이미지 생성
        error_img = Image.new('RGB', (640, 480), color=(0, 0, 0))
        tk_error_img = ImageTk.PhotoImage(error_img)
        
        # 라벨에 이미지 표시
        self.webcam_label.config(image=tk_error_img)
        self.webcam_label.image = tk_error_img
    
    def on_window_close(self):
        """윈도우 종료 처리"""
        if self.on_closing:
            self.on_closing()
        self.root.destroy()
        
    def run(self):
        """GUI 실행"""
        self.root.mainloop()