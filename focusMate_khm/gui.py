import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import time

class FacialStateGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("얼굴 상태 감지 프로그램")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")

        self.on_start_stop = None
        self.on_closing = None

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_window_close)

        self.face_detected_status = False

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.header_frame = ttk.Frame(self.main_frame, padding=5)
        self.header_frame.pack(fill=tk.X, pady=10)

        ttk.Label(
            self.header_frame,
            text="얼굴 상태 감지 및 음악 재생 프로그램",
            font=("Helvetica", 16, "bold")
        ).pack()

        ttk.Label(
            self.header_frame,
            text="정면 얼굴/측면/하품 감지 후 음악 반응",
            font=("Helvetica", 10)
        ).pack(pady=5)

        self.control_frame = ttk.Frame(self.main_frame, padding=5)
        self.control_frame.pack(fill=tk.X, pady=5)

        self.start_stop_button = ttk.Button(
            self.control_frame,
            text="시작",
            width=15,
            command=self.start_stop_clicked
        )
        self.start_stop_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.status_label = ttk.Label(
            self.control_frame,
            text="대기 중...",
            font=("Helvetica", 10)
        )
        self.status_label.pack(side=tk.LEFT, padx=20, pady=5)

        self.face_status_label = ttk.Label(
            self.control_frame,
            text="얼굴 감지 상태: 없음",
            font=("Helvetica", 10)
        )
        self.face_status_label.pack(side=tk.RIGHT, padx=20, pady=5)

        self.webcam_frame = ttk.LabelFrame(self.main_frame, text="실시간 웹캠 화면", padding=10)
        self.webcam_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.webcam_label = ttk.Label(self.webcam_frame)
        self.webcam_label.pack(pady=10)

        self.info_frame = ttk.Frame(self.main_frame, padding=5)
        self.info_frame.pack(fill=tk.BOTH, expand=True)

        self.status_text = tk.Text(self.info_frame, height=8, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        self.status_scrollbar = ttk.Scrollbar(self.info_frame, command=self.status_text.yview)
        self.status_scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.status_text.config(yscrollcommand=self.status_scrollbar.set)

        self.status_text.insert(tk.END, "프로그램이 시작되었습니다.\n")
        self.status_text.insert(tk.END, "시작 버튼을 눌러 얼굴 감지를 시작하세요.\n")
        self.status_text.config(state=tk.DISABLED)

    def start_stop_clicked(self):
        if self.on_start_stop:
            self.on_start_stop()

    def set_start_stop_button_state(self, is_running):
        self.start_stop_button.config(text="중지" if is_running else "시작")

    def update_status(self, message, warning=False):
        self.status_text.config(state=tk.NORMAL)
        current_time = time.strftime("%H:%M:%S", time.localtime())
        full_message = f"[{current_time}] {message}\n"
        tag = "warning" if warning else None
        self.status_text.insert(tk.END, full_message, tag)
        if warning:
            self.status_text.tag_configure("warning", foreground="red")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
        self.status_label.config(text=message)

    def update_face_status(self, detected):
        if detected != self.face_detected_status:
            self.face_detected_status = detected
            self.face_status_label.config(
                text="얼굴 감지 상태: 감지됨" if detected else "얼굴 감지 상태: 없음"
            )

    def update_webcam_display(self, frame):
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_image = pil_image.resize((640, 480), Image.LANCZOS)
            tk_image = ImageTk.PhotoImage(image=pil_image)
            self.webcam_label.config(image=tk_image)
            self.webcam_label.image = tk_image

    def show_webcam_error(self):
        self.update_status("웹캠을 사용할 수 없습니다. 연결을 확인하세요.", warning=True)
        error_img = Image.new('RGB', (640, 480), color=(0, 0, 0))
        tk_error_img = ImageTk.PhotoImage(error_img)
        self.webcam_label.config(image=tk_error_img)
        self.webcam_label.image = tk_error_img

    def on_window_close(self):
        if self.on_closing:
            self.on_closing()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

