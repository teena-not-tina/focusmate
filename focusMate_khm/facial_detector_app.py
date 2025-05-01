#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import time

from face_detector import FaceDetector
from gui import FacialStateGUI
from music_player import MusicPlayer

class FacialDetectorApp:
    def __init__(self):
        self.gui = FacialStateGUI()
        self.register_event_handlers()
        self.face_detector = FaceDetector()
        self.music_player = MusicPlayer(self.gui)
        
        self.init_webcam()
        self.init_detection_variables()
        
    def init_webcam(self):
        """웹캠 초기화"""
        self.webcam_available = False
        for camera_index in range(3):
            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ret, test_frame = self.cap.read()
                if ret:
                    self.webcam_available = True
                    break
                else:
                    self.cap.release()

        if not self.webcam_available:
            self.gui.show_webcam_error()

    def init_detection_variables(self):
        """감지 관련 변수 초기화"""
        self.is_running = False
        self.current_frame = None
        self.processed_frame = None
        
        # 얼굴 정면 감지 관련 변수
        self.face_detected = False
        self.face_detection_start_time = None
        self.face_occupancy_threshold = 0.30  # 화면의 30%
        self.face_time_threshold = 2.0  # 2초
        self.face_threshold_met = False
        
        # 하품 감지 관련 변수
        self.yawn_start_time = None
        self.is_yawning = False
        self.yawn_detected = False
        self.yawn_threshold = 0.02  # 입 벌어짐 정규화된 값
        self.yawn_time_threshold = 3.0  # 3초
        
        # 얼굴 측면 감지 관련 변수
        self.face_side_detected = False
        self.face_side_start_time = None
        self.face_side_threshold_met = False
        self.head_yaw_threshold = 0.30  # 머리 회전 임계값
        self.face_side_time_threshold = 2.0  # 2초

    def register_event_handlers(self):
        """GUI 이벤트 핸들러 등록"""
        self.gui.on_start_stop = self.toggle_start_stop
        self.gui.on_closing = self.on_closing

    def toggle_start_stop(self):
        """시작/중지 버튼 동작 처리"""
        if self.is_running:
            self.is_running = False
            self.gui.set_start_stop_button_state(False)
            self.music_player.close_browser()
            self.gui.update_status("중지됨")
            self.reset_detection_state()
        else:
            if not self.webcam_available:
                self.gui.update_status("웹캠을 사용할 수 없습니다.", warning=True)
                return
                
            self.is_running = True
            self.gui.set_start_stop_button_state(True)
            self.gui.update_status("실행 중... 얼굴 감지 대기중")
            self.reset_detection_state()
            self.update_webcam()
    
    def reset_detection_state(self):
        """감지 상태 초기화"""
        # 얼굴 정면 감지 상태 초기화
        self.face_detected = False
        self.face_detection_start_time = None
        self.face_threshold_met = False
        
        # 하품 감지 상태 초기화
        self.yawn_detected = False
        self.is_yawning = False
        self.yawn_start_time = None
        
        # 얼굴 측면 감지 상태 초기화
        self.face_side_detected = False
        self.face_side_start_time = None
        self.face_side_threshold_met = False
        
        # 음악 재생 상태 초기화
        self.music_player.reset_music_state()

    def update_webcam(self):
        """
        웹캠 프레임 업데이트 및 얼굴/하품/측면 감지
        음악이 재생 중이면 감지를 중지함
        """
        if not self.is_running or not self.webcam_available:
            return
        
        # 음악이 재생 중이면 감지 중지 (화면 업데이트만 계속)
        if self.music_player.music_playing:
            try:
                ret, self.current_frame = self.cap.read()
                if not ret:
                    self.gui.update_status("웹캠에서 프레임을 읽을 수 없습니다.", warning=True)
                    self.is_running = False
                    self.gui.set_start_stop_button_state(False)
                    return
                    
                self.current_frame = cv2.flip(self.current_frame, 1)  # 좌우 반전
                
                # 상태 메시지 표시
                music_type = self.music_player.current_search_term if self.music_player.current_search_term else "음악"
                cv2.putText(self.current_frame, f"{music_type} 재생 중 - 얼굴 감지 일시 중지됨", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                           
                # 프레임 표시
                self.gui.update_webcam_display(self.current_frame)
                
                # 재귀 호출
                self.gui.root.after(10, self.update_webcam)
            except Exception as e:
                self.gui.root.after(100, self.update_webcam)
            return
        
        # 음악이 재생 중이 아닐 때만 얼굴 감지 수행
        try:
            ret, self.current_frame = self.cap.read()
            if not ret:
                self.gui.update_status("웹캠에서 프레임을 읽을 수 없습니다.", warning=True)
                self.is_running = False
                self.gui.set_start_stop_button_state(False)
                return
                
            self.current_frame = cv2.flip(self.current_frame, 1)  # 좌우 반전
            frame_height, frame_width, _ = self.current_frame.shape

            # 얼굴 감지
            face_detected, frame_with_face, face_data = self.face_detector.detect_face(self.current_frame)
            self.processed_frame = frame_with_face
            
            # 현재 시간
            current_time = time.time()
            
            # GUI 상태 업데이트
            self.gui.update_face_status(face_detected)
            
            # 얼굴이 감지되면 관련 로직 적용
            if face_detected:
                # 얼굴 메트릭 데이터 분석
                face_metrics = face_data.get('face_metrics', {})
                face_box = face_data.get('face_box', [0, 0, 0, 0])  # [x, y, w, h]
                mouth_open_height = face_metrics.get('mouth_open_height', 0)
                head_pose_yaw = face_metrics.get('head_pose_yaw', 0)
                
                # 얼굴 점유 비율 계산
                face_area = face_box[2] * face_box[3]  # 얼굴 영역 (너비 * 높이)
                frame_area = frame_width * frame_height  # 전체 프레임 영역
                face_occupancy = face_area / frame_area  # 얼굴이 차지하는 비율
                
                # 얼굴 정면 여부 확인 (머리 회전 각도로 판단)
                is_face_front = head_pose_yaw < self.head_yaw_threshold
                is_face_side = head_pose_yaw >= self.head_yaw_threshold
                
                # 1. 얼굴 정면 점유 상태 확인 (화면의 30% 이상)
                if is_face_front:
                    self.detect_face_front(face_occupancy, current_time)
                else:
                    # 정면이 아니면 정면 감지 상태 초기화
                    self.face_detected = False
                    self.face_detection_start_time = None
                
                # 2. 하품 감지 상태 업데이트
                self.detect_yawn(mouth_open_height, current_time)
                
                # 3. 얼굴 측면 감지
                self.detect_face_side(head_pose_yaw, current_time)
                
                # 화면에 상태 표시
                self.display_detection_state(face_occupancy, mouth_open_height, head_pose_yaw, current_time)
            else:
                # 얼굴이 감지되지 않으면 모든 타이머 초기화
                self.face_detected = False
                self.face_detection_start_time = None
                self.is_yawning = False
                self.yawn_start_time = None
                self.face_side_detected = False
                self.face_side_start_time = None
            
            # 최종 처리된 프레임 표시
            self.gui.update_webcam_display(self.processed_frame)
            
            # 재귀 호출 (10ms 후 다시 업데이트)
            self.gui.root.after(10, self.update_webcam)
        except Exception as e:
            self.gui.update_status("웹캠 업데이트 중 오류 발생", warning=True)
            self.gui.root.after(100, self.update_webcam)  # 오류 발생 시 더 긴 간격으로 재시도

    def detect_face_front(self, face_occupancy, current_time):
        """얼굴 정면이 화면에서 차지하는 비율을 확인하고 조건 충족 시 알파파 음악 재생"""
        # 이미 알파파 음악이 재생 중이면 추가 감지 중단
        if self.face_threshold_met and self.music_player.current_search_term == "알파파":
            return
        
        # 얼굴이 기준치 이상 차지하는지 확인
        if face_occupancy >= self.face_occupancy_threshold:
            # 얼굴 감지 시작 시간 기록
            if not self.face_detected:
                self.face_detected = True
                self.face_detection_start_time = current_time
                self.gui.update_status(f"정면 얼굴 감지: 화면의 {face_occupancy:.1%} 차지")
            
            # 얼굴 감지 지속 시간 계산
            if self.face_detection_start_time:
                face_duration = current_time - self.face_detection_start_time
                
                # 얼굴이 기준 시간 이상 감지되면 조건 충족
                if face_duration >= self.face_time_threshold and not self.face_threshold_met:
                    self.face_threshold_met = True
                    self.gui.update_status(f"정면 얼굴이 화면의 30% 이상 차지, 지속 시간: {face_duration:.1f}초")
                    
                    # 알파파 음악 재생
                    self.music_player.play_music("gamma")
        else:
            # 얼굴이 작아지면 감지 상태 초기화
            if self.face_detected:
                self.face_detected = False
                self.face_detection_start_time = None
                self.gui.update_status("얼굴 크기가 감소함")

    def detect_yawn(self, mouth_height, current_time):
        """하품 감지 (입이 일정 수준 이상 벌어진 상태가 3초 이상 지속되는지 확인)"""
        # 이미 모짜르트 음악이 재생 중이면 추가 감지 중단
        if self.yawn_detected and self.music_player.current_search_term == "모짜르트":
            return
        
        # 입이 기준치 이상 벌어졌는지 확인
        if mouth_height > self.yawn_threshold:
            # 하품 시작 시간 기록
            if not self.is_yawning:
                self.is_yawning = True
                self.yawn_start_time = current_time
                self.gui.update_status(f"입이 벌어짐 감지")
            
            # 하품 지속 시간 계산
            if self.yawn_start_time:
                yawn_duration = current_time - self.yawn_start_time
                
                # 하품이 기준 시간 이상 지속되면 피곤함으로 판단
                if yawn_duration >= self.yawn_time_threshold and not self.yawn_detected:
                    self.yawn_detected = True
                    self.gui.update_status(f"하품 감지! 지속 시간: {yawn_duration:.1f}초")
                    
                    # 모짜르트 음악 재생
                    self.music_player.play_music("모짜르트")
        else:
            # 입이 다시 닫히면 하품 상태 초기화
            if self.is_yawning:
                self.is_yawning = False
                self.gui.update_status("입이 닫힘 감지")
                
    def detect_face_side(self, head_yaw, current_time):
        """얼굴 측면 감지 (머리를 옆으로 돌린 상태 감지)"""
        # 이미 알파파 음악이 재생 중이면 추가 감지 중단
        if self.face_side_threshold_met and self.music_player.current_search_term == "알파파":
            return
            
        # 머리 회전 각도가 기준치 이상인지 확인
        if head_yaw >= self.head_yaw_threshold:
            # 측면 감지 시작 시간 기록
            if not self.face_side_detected:
                self.face_side_detected = True
                self.face_side_start_time = current_time
                self.gui.update_status(f"얼굴 측면 감지: 회전 각도 {head_yaw:.2f}")
            
            # 측면 감지 지속 시간 계산
            if self.face_side_start_time:
                side_duration = current_time - self.face_side_start_time
                
                # 측면이 기준 시간 이상 지속되면 조건 충족
                if side_duration >= self.face_side_time_threshold and not self.face_side_threshold_met:
                    self.face_side_threshold_met = True
                    self.gui.update_status(f"얼굴 측면 감지! 지속 시간: {side_duration:.1f}초")
                    
                    # 알파파 음악 재생
                    self.music_player.play_music("alpha")
        else:
            # 머리를 다시 정면으로 돌리면 측면 감지 상태 초기화
            if self.face_side_detected:
                self.face_side_detected = False
                self.face_side_start_time = None
                self.gui.update_status("얼굴이 정면으로 돌아옴")

    def display_detection_state(self, face_occupancy, mouth_height, head_yaw, current_time):
        """현재 감지 상태를 화면에 표시"""
        # 얼굴 크기 표시
        face_text = f"얼굴 점유율: {face_occupancy:.1%}"
        cv2.putText(self.processed_frame, face_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 얼굴 회전 각도 표시
        yaw_text = f"머리 회전: {head_yaw:.2f}"
        cv2.putText(self.processed_frame, yaw_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 현재 y 위치
        y_pos = 90
        
        # 1. 얼굴 정면 감지 상태 표시
        if self.face_detected and not self.face_threshold_met:
            face_duration = current_time - self.face_detection_start_time
            duration_text = f"정면 감지중: {face_duration:.1f}초 / {self.face_time_threshold}초"
            cv2.putText(self.processed_frame, duration_text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            y_pos += 30
        
        # 알파파 음악 재생 상태 표시
        if self.face_threshold_met and self.music_player.current_search_term == "알파파":
            beta_text = "정면 얼굴 감지됨! 알파파 음악 재생 중"
            cv2.putText(self.processed_frame, beta_text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y_pos += 30
        
        # 2. 하품 감지 중 상태 표시
        if self.is_yawning:
            yawn_duration = current_time - self.yawn_start_time
            yawn_text = f"하품 감지중: {yawn_duration:.1f}초 / {self.yawn_time_threshold}초"
            cv2.putText(self.processed_frame, yawn_text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            y_pos += 30
        
        # 모짜르트 음악 재생 상태 표시
        if self.yawn_detected and self.music_player.current_search_term == "모짜르트":
            mozart_text = "하품 감지됨! 모짜르트 음악 재생 중"
            cv2.putText(self.processed_frame, mozart_text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_pos += 30
        
        # 3. 얼굴 측면 감지 상태 표시
        if self.face_side_detected and not self.face_side_threshold_met:
            side_duration = current_time - self.face_side_start_time
            side_text = f"측면 감지중: {side_duration:.1f}초 / {self.face_side_time_threshold}초"
            cv2.putText(self.processed_frame, side_text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            y_pos += 30
        
        # 알파파 음악 재생 상태 표시
        if self.face_side_threshold_met and self.music_player.current_search_term == "알파파":
            gamma_text = "얼굴 측면 감지됨! 알파파 음악 재생 중"
            cv2.putText(self.processed_frame, gamma_text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    def on_closing(self):
        """프로그램 종료 처리"""
        self.is_running = False
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'face_detector'):
            self.face_detector.close()
            
        # 브라우저 종료
        self.music_player.close_browser()

    def run(self):
        """애플리케이션 실행"""
        self.gui.run()