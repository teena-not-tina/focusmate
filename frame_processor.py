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
        
        self.eye_event_active = False
        self.yawn_event_active = False

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
        """
        Process a video frame to analyze facial features and update state history.
        
        Args:
            frame: OpenCV image frame (numpy array)
            
        Returns:
            tuple: (annotated_frame, alerts_dict)
            - annotated_frame: Frame with visual annotations
            - alerts_dict: Dictionary with any triggered alerts
        """
        current_time = time.time()
        time_delta = 0
        
        if self.last_process_time is not None:
            time_delta = current_time - self.last_process_time
        
        self.last_process_time = current_time
        
        # Save the frame temporarily for analysis
        temp_filename = os.path.join(self.temp_dir, "temp_frame.jpg")
        cv2.imwrite(temp_filename, frame)
        
        # Analyze the frame for eye and mouth state
        try:
            # Process eye state
            eye_results, eye_frame = self.eye_detector.detect_eye_state(
                temp_filename, 
                visualize=True,
                threshold_ratio=self.eye_threshold
            )
            
            # Process mouth state
            mouth_results, mouth_frame = self.mouth_detector.detect_mouth_state(
                temp_filename, 
                visualize=True,
                threshold_ratio=self.mouth_threshold
            )
            
            # Check if both analyses were successful
            if not eye_results or not mouth_results:
                print("No valid face detected in the frame.")
                return frame, {}
                
        except Exception as e:
            print(f"Error analyzing frame: {e}")
            return frame, {}
        
        # Combine results (use eye_frame as base and add mouth information)
        combined_frame = eye_frame.copy() if eye_frame is not None else frame.copy()
        
        # Get first face data (assuming single face processing for simplicity)
        eye_data = eye_results[0]
        mouth_data = mouth_results[0]
        
        # Check if we're looking at the same face
        if eye_data['face_idx'] != mouth_data['face_idx']:
            print("Warning: Eye and mouth data may be from different faces")
        
        # Extract state information
        both_eyes_closed = eye_data['both_eyes_closed']
        is_yawning = mouth_data['mouth']['is_yawning']
        
        # Add current states to history
        current_timestamp = datetime.now()
        self.eye_states.append((current_timestamp, both_eyes_closed))
        self.mouth_states.append((current_timestamp, is_yawning))
        
        # Update durations
        if both_eyes_closed:
            self.current_eyes_closed_duration += time_delta
        else:
            self.current_eyes_closed_duration = 0
            
        if is_yawning:
            self.current_yawning_duration += time_delta
        else:
            self.current_yawning_duration = 0
        
        alerts = {}
    
        # Eyes closed event logic
        if both_eyes_closed:
            if (self.current_eyes_closed_duration >= self.eyes_closed_alert_threshold and
                not self.eye_event_active and
                self.eye_alerts_sent < self.max_eye_alerts):
                alerts['eyes_closed'] = {
                    'duration': self.current_eyes_closed_duration,
                    'message': f"Alert: Eyes have been closed for {self.current_eyes_closed_duration:.1f} seconds. Please wake up!"
                }
                self.eye_alerts_sent += 1
                self.eye_event_active = True
        else:
            self.eye_event_active = False

        # Yawning event logic
        if is_yawning:
            if (self.current_yawning_duration >= self.yawning_alert_threshold and
                not self.yawn_event_active and
                self.yawn_alerts_sent < self.max_yawn_alerts):
                alerts['yawning'] = {
                    'duration': self.current_yawning_duration,
                    'message': f"Alert: You've been yawning for {self.current_yawning_duration:.1f} seconds. Do you want to take a break?"
                }
                self.yawn_alerts_sent += 1
                self.yawn_event_active = True
        else:
            self.yawn_event_active = False

        # Draw overlay info
        y_start = 60
        line_height = 30

        cv2.putText(
            combined_frame, 
            f"Eyes closed: {self.current_eyes_closed_duration:.1f}s", 
            (20, y_start + line_height), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 0, 255) if self.current_eyes_closed_duration >= self.eyes_closed_alert_threshold else (255, 0, 0), 
            2
        )
        cv2.putText(
            combined_frame, 
            f"Yawning: {self.current_yawning_duration:.1f}s", 
            (20, y_start + 2 * line_height), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 165, 255) if self.current_yawning_duration >= self.yawning_alert_threshold else (255, 165, 0), 
            2
        )
        cv2.putText(
            combined_frame,
            f"Alerts: {self.eye_alerts_sent}/{self.max_eye_alerts} (eyes), {self.yawn_alerts_sent}/{self.max_yawn_alerts} (yawn)",
            (20, y_start + 3 * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (128, 0, 128),
            2
        )

        return combined_frame, alerts
        
    def reset_alerts(self):
        """Reset alert counters to allow new alerts to be triggered"""
        self.eye_alerts_sent = 0
        self.yawn_alerts_sent = 0
        print("Alert counters reset")
    
    def get_state_statistics(self):
        """
        Calculate statistics about the facial states over the tracked history.
        
        Returns:
            Dictionary with statistics
        """
        # Count eye closed frames
        eye_closed_frames = sum(1 for _, state in self.eye_states if state)
        eye_closed_percentage = 0
        if len(self.eye_states) > 0:
            eye_closed_percentage = (eye_closed_frames / len(self.eye_states)) * 100
        
        # Count yawning frames
        yawning_frames = sum(1 for _, state in self.mouth_states if state)
        yawning_percentage = 0
        if len(self.mouth_states) > 0:
            yawning_percentage = (yawning_frames / len(self.mouth_states)) * 100
        
        return {
            'eye_history_length': len(self.eye_states),
            'mouth_history_length': len(self.mouth_states),
            'eye_closed_percentage': eye_closed_percentage,
            'yawning_percentage': yawning_percentage,
            'current_eyes_closed_duration': self.current_eyes_closed_duration,
            'current_yawning_duration': self.current_yawning_duration,
            'eye_alerts_sent': self.eye_alerts_sent,
            'yawn_alerts_sent': self.yawn_alerts_sent
        }