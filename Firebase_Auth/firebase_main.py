import sys
import cv2
import os
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QFrame, QMessageBox
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer
from firebase_auth import signup_user, login_user


class OverlayLoginApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("카메라 로그인 앱")

        self.logged_in = False
        self.capture = cv2.VideoCapture(0)

        self.cam_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[카메라 해상도] width: {self.cam_width}, height: {self.cam_height}")

        self.resize(self.cam_width, self.cam_height)
        print(f"[초기 윈도우 창 크기] width: {self.width()}, height: {self.height()}")

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.background_label = QLabel(self.central_widget)
        self.background_label.setGeometry(0, 0, self.cam_width, self.cam_height)
        self.background_label.setScaledContents(True)

        self.login_frame = QFrame(self.central_widget)
        self.login_frame.setFixedSize(300, 200)
        self.login_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 200);
                border-radius: 10px;
            }
            QLineEdit, QPushButton {
                font-size: 14px;
                padding: 8px;
            }
        """)
        self.center_login_frame()

        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("이메일")

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("비밀번호")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)

        self.login_button = QPushButton("로그인")
        self.login_button.clicked.connect(self.login)

        self.signup_button = QPushButton("회원가입")
        self.signup_button.clicked.connect(self.signup)

        self.email_input.returnPressed.connect(self.login)
        self.password_input.returnPressed.connect(self.login)

        form_layout = QVBoxLayout()
        form_layout.addWidget(self.email_input)
        form_layout.addWidget(self.password_input)
        form_layout.addWidget(self.login_button)
        form_layout.addWidget(self.signup_button)
        self.login_frame.setLayout(form_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.capture_timer = QTimer()
        self.capture_timer.setInterval(5000)
        self.capture_timer.timeout.connect(self.save_capture)

    def center_login_frame(self):
        win_w = self.width()
        win_h = self.height()
        frm_w = self.login_frame.width()
        frm_h = self.login_frame.height()
        self.login_frame.move((win_w - frm_w) // 2, (win_h - frm_h) // 2)

    def resizeEvent(self, event):
        self.background_label.setGeometry(0, 0, self.width(), self.height())
        self.center_login_frame()
        print(f"[윈도우 크기 변경됨] width: {self.width()}, height: {self.height()}")

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return

        if not self.logged_in:
            frame = cv2.GaussianBlur(frame, (45, 45), 0)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.background_label.width(),
            self.background_label.height(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding
        )
        self.background_label.setPixmap(pixmap)

    def login(self):
        email = self.email_input.text()
        password = self.password_input.text()

        success, msg = login_user(email, password)  # Firebase 인증 요청
        if success:
            self.logged_in = True
            self.login_frame.hide()
            print("[로그인 성공] 영상 캡처 시작됨 (5초마다 저장)")
            self.capture_timer.start()
        else:
            self.show_message(f"로그인 실패: {msg}")

    def signup(self):
        email = self.email_input.text()
        password = self.password_input.text()
        success, msg = signup_user(email, password)
        if success:
            self.email_input.clear()
            self.password_input.clear()
        self.show_message(msg)

    def save_capture(self):
        ret, frame = self.capture.read()
        if not ret:
            print("❌ 캡처 실패 (카메라 프레임 없음)")
            return

        if not os.path.exists("captures"):
            os.makedirs("captures")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captures/capture_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[✔ 저장됨] {filename}")

    def show_message(self, text):
        QMessageBox.information(self, "알림", text)

    def closeEvent(self, event):
        self.capture.release()
        self.capture_timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OverlayLoginApp()
    window.show()
    sys.exit(app.exec())
