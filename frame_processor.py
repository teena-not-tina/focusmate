from face_detector import FacialFeatureAnalyzer
from eyelid_detector import EyelidDistanceDetector
from mouth_detector import MouthDistanceDetector
import cv2
import numpy as np
import time
from collections import deque
from datetime import datetime
import os
import config

class FacialStateTracker:
    """
    Tracks facial states (eyes closed, yawning) over time and provides alerts
    based on configurable thresholds.
    """
    
    def __init__(self, 
                 eye_history_seconds=15,
                 mouth_history_seconds=15, 
                 eyes_closed_alert_threshold=10,
                 yawning_alert_threshold=4,
                 max_eye_alerts=2,
                 max_yawn_alerts=2,
                 eye_threshold=0.05,
                 mouth_threshold=0.1,
                 providers=['CPUExecutionProvider']):
        """
        Initialize the facial state tracker.
        
        Args:
            eye_history_seconds: Number of seconds to keep eye state history
            mouth_history_seconds: Number of seconds to keep mouth state history
            eyes_closed_alert_threshold: Seconds of continuous eye closure to trigger alert
            yawning_alert_threshold: Seconds of continuous mouth opening to consider as yawning
            max_eye_alerts: Maximum number of eye closure alerts to send
            max_yawn_alerts: Maximum number of yawning alerts to send
            eye_threshold: Threshold ratio for eye openness
            mouth_threshold: Threshold ratio for mouth openness
            providers: List of providers for InsightFace
        """
        # Initialize detectors
        self.eye_detector = EyelidDistanceDetector(providers)
        self.mouth_detector = MouthDistanceDetector(providers)
        
        # Create temp directory for frame processing
        self.temp_dir = config.TEMP_DIR
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Configuration
        self.eye_threshold = eye_threshold
        self.mouth_threshold = mouth_threshold
        
        # Alert thresholds
        self.eyes_closed_alert_threshold = eyes_closed_alert_threshold
        self.yawning_alert_threshold = yawning_alert_threshold
        
        # Max alerts
        self.max_eye_alerts = max_eye_alerts
        self.max_yawn_alerts = max_yawn_alerts
        self.eye_alerts_sent = 0
        self.yawn_alerts_sent = 0
        
        # History tracking
        self.fps = config.PROCESSING_FPS
        self.eye_history_length = int(eye_history_seconds * self.fps)
        self.mouth_history_length = int(mouth_history_seconds * self.fps)
        
        # State history: each entry is a tuple (timestamp, state)
        self.eye_states = deque(maxlen=self.eye_history_length)
        self.mouth_states = deque(maxlen=self.mouth_history_length)
        
        # Current state tracking
        self.current_eyes_closed_duration = 0
        self.current_yawning_duration = 0
        self.last_process_time = None
        
        print(f"Facial State Tracker initialized with:")
        print(f"  - Eye alert threshold: {eyes_closed_alert_threshold} seconds")
        print(f"  - Yawn alert threshold: {yawning_alert_threshold} seconds")
        print(f"  - Max alerts: {max_eye_alerts} (eyes), {max_yawn_alerts} (yawning)")
    
    def process_frame(self, frame):
        now = time.time()
        dt = now - self.last_time if self.last_time else 0
        self.last_time = now

        temp_path = os.path.join(self.temp_dir, "temp.jpg")
        cv2.imwrite(temp_path, frame)

        try:
            eye_results, eye_frame = self.eye_detector.detect_eye_state(temp_path, visualize=True, threshold_ratio=config.EYE_THRESHOLD)
            mouth_results, mouth_frame = self.mouth_detector.detect_mouth_state(temp_path, visualize=True, threshold_ratio=config.MOUTH_THRESHOLD)
            if not eye_results or not mouth_results:
                return frame, {}
        except Exception as e:
            print(f"Error: {e}")
            return frame, {}

        eye_closed = eye_results[0]['both_eyes_closed']
        yawning = mouth_results[0]['mouth']['is_yawning']

        self.eye_history.append((datetime.now(), eye_closed))
        self.mouth_history.append((datetime.now(), yawning))

        self.eye_closed_time = self.eye_closed_time + dt if eye_closed else 0
        self.yawn_time = self.yawn_time + dt if yawning else 0

        alerts = {}
        if self.eye_closed_time >= config.EYES_CLOSED_ALERT_THRESHOLD and self.eye_alerts < config.MAX_EYE_ALERTS:
            alerts['eyes_closed'] = f"Eyes closed for {self.eye_closed_time:.1f}s"
            self.eye_alerts += 1
        if self.yawn_time >= config.YAWNING_ALERT_THRESHOLD and self.yawn_alerts < config.MAX_YAWN_ALERTS:
            alerts['yawning'] = f"Yawning for {self.yawn_time:.1f}s"
            self.yawn_alerts += 1

        annotated = self._draw_overlay(eye_frame, self.eye_closed_time, self.yawn_time, self.eye_alerts, self.yawn_alerts)
        return annotated, alerts

    def _draw_overlay(self, frame, eye_time, yawn_time, eye_alerts, yawn_alerts):
        y = 120
        if eye_time:
            cv2.putText(frame, f"Eyes closed: {eye_time:.1f}s", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        y += 30
        if yawn_time:
            cv2.putText(frame, f"Yawning: {yawn_time:.1f}s", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
        y += 30
        cv2.putText(frame, f"Alerts: {eye_alerts}/{config.MAX_EYE_ALERTS} (eyes), {yawn_alerts}/{config.MAX_YAWN_ALERTS} (yawn)", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128,0,128), 2)
        return frame

    def reset_alerts(self):
        self.eye_alerts = 0
        self.yawn_alerts = 0

    def get_state_statistics(self):
        eye_closed_pct = 100 * sum(1 for _, s in self.eye_history if s) / len(self.eye_history) if self.eye_history else 0
        yawn_pct = 100 * sum(1 for _, s in self.mouth_history if s) / len(self.mouth_history) if self.mouth_history else 0
        return {
            'eye_closed_percentage': eye_closed_pct,
            'yawning_percentage': yawn_pct,
            'current_eyes_closed_duration': self.eye_closed_time,
            'current_yawning_duration': self.yawn_time,
            'eye_alerts_sent': self.eye_alerts,
            'yawn_alerts_sent': self.yawn_alerts
        }