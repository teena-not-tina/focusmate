#!/usr/bin/env python3
"""
Simple script to run the facial state tracking system.
This will monitor for frames in the temp_frames directory and alert when yawning or sleeping is detected.
"""
import os
import sys
import time
import cv2
import numpy as np
import pygame
from datetime import datetime
import argparse

# Add the parent directory to the path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import our modules
from frame_processor import FacialStateTracker, FrameMonitor
import config

def init_sounds():
    """Initialize alert sounds"""
    # Initialize pygame for sound
    pygame.mixer.init()
    
    # Create sounds directory
    sound_dir = "sounds"
    os.makedirs(sound_dir, exist_ok=True)
    
    # Sound file paths
    yawn_sound_path = os.path.join(sound_dir, "yawn_alert.mp3")
    sleep_sound_path = os.path.join(sound_dir, "sleep_alert.mp3")
    
    # Download sounds if they don't exist
    if not os.path.exists(yawn_sound_path):
        try:
            import urllib.request
            print("Downloading yawn alert sound...")
            yawn_url = "https://cdn.pixabay.com/download/audio/2022/10/30/audio_92b6c97289.mp3"
            urllib.request.urlretrieve(yawn_url, yawn_sound_path)
            print(f"Downloaded yawn alert sound to {yawn_sound_path}")
        except Exception as e:
            print(f"Failed to download yawn sound: {e}")
            yawn_sound_path = None
    
    if not os.path.exists(sleep_sound_path):
        try:
            import urllib.request
            print("Downloading sleep alert sound...")
            sleep_url = "https://cdn.pixabay.com/download/audio/2022/03/15/audio_95b42c0ffe.mp3"
            urllib.request.urlretrieve(sleep_url, sleep_sound_path)
            print(f"Downloaded sleep alert sound to {sleep_sound_path}")
        except Exception as e:
            print(f"Failed to download sleep sound: {e}")
            sleep_sound_path = None
    
    return yawn_sound_path, sleep_sound_path

def play_sound(sound_path):
    """Play a sound file"""
    if sound_path:
        try:
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Failed to play sound: {e}")

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Facial State Tracker")
    parser.add_argument("--frames-dir", type=str, default="./StartPage/temp_frames",
                       help="Directory containing frames to process")
    parser.add_argument("--yawn-threshold", type=float, default=config.YAWNING_ALERT_THRESHOLD,
                       help=f"Seconds of yawning to trigger alert (default: {config.YAWNING_ALERT_THRESHOLD})")
    parser.add_argument("--sleep-threshold", type=float, default=config.EYES_CLOSED_ALERT_THRESHOLD,
                       help=f"Seconds of eye closure to trigger alert (default: {config.EYES_CLOSED_ALERT_THRESHOLD})")
    parser.add_argument("--max-alerts", type=int, default=config.MAX_EYE_ALERTS,
                       help=f"Maximum number of alerts per type (default: {config.MAX_EYE_ALERTS})")
    parser.add_argument("--no-sound", action="store_true",
                       help="Disable alert sounds")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Focusmate Alertness Tracker".center(60))
    print("="*60)
    print(f"Monitoring frames in: {args.frames_dir}")
    print(f"Yawn threshold: {args.yawn_threshold} seconds")
    print(f"Sleep threshold: {args.sleep_threshold} seconds")
    print(f"Maximum alerts per type: {args.max_alerts}")
    print("-"*60)
    
    # Initialize sounds if enabled
    yawn_sound, sleep_sound = None, None
    if not args.no_sound:
        yawn_sound, sleep_sound = init_sounds()
    
    # Create facial state tracker
    tracker = FacialStateTracker(
        eye_history_seconds=15,
        mouth_history_seconds=15,
        eyes_closed_alert_threshold=args.sleep_threshold,
        yawning_alert_threshold=args.yawn_threshold,
        max_eye_alerts=args.max_alerts,
        max_yawn_alerts=args.max_alerts
    )
    
    # Create frame monitor
    monitor = FrameMonitor(frames_dir=args.frames_dir, tracker=tracker)
    
    # Start monitoring
    monitor.start()
    
    # Display initial status
    print("\nMonitoring started.")
    print("The system will track:")
    print(f"  - Yawning: Alert after {args.yawn_threshold} seconds of open mouth")
    print(f"  - Sleeping: Alert after {args.sleep_threshold} seconds of closed eyes")
    print(f"  - Maximum of {args.max_alerts} alerts per type")
    print("\nPress Ctrl+C to exit.\n")
    
    # Variables to track last alert times
    last_yawn_alert_time = 0
    last_sleep_alert_time = 0
    
    try:
        # Main loop to check for alerts and display status
        while True:
            # Get the latest alert
            alert = monitor.get_latest_alert()
            
            if alert:
                alert_type = alert['type']
                message = alert['message']
                
                if alert_type == 'sleep':
                    # Only play sound at most every 8 seconds for sleep alerts
                    current_time = time.time()
                    if current_time - last_sleep_alert_time > 8:
                        last_sleep_alert_time = current_time
                        # Print with red color
                        print(f"\033[91m{message}\033[0m")
                        # Play alert sound
                        if not args.no_sound:
                            play_sound(sleep_sound)
                
                elif alert_type == 'yawn':
                    # Only play sound at most every 8 seconds for yawn alerts
                    current_time = time.time()
                    if current_time - last_yawn_alert_time > 8:
                        last_yawn_alert_time = current_time
                        # Print with yellow color
                        print(f"\033[93m{message}\033[0m")
                        # Play alert sound
                        if not args.no_sound:
                            play_sound(yawn_sound)
            
            # Get current state statistics
            stats = tracker.get_state_statistics()
            
            # Clear line and print current state
            print("\r" + " " * 100, end="\r")  # Clear line
            
            # Format status text based on current state
            status_text = "Status: "
            
            # Eye state
            if stats['current_eyes_closed_duration'] > 0:
                eye_text = f"Eyes closed: \033[91m{stats['current_eyes_closed_duration']:.1f}s\033[0m"
                if stats['is_sleeping']:
                    eye_text += " \033[91m[SLEEPING!]\033[0m"
            else:
                eye_text = "\033[92mEyes open\033[0m"
            
            # Mouth state
            if stats['current_yawning_duration'] > 0:
                mouth_text = f"Yawning: \033[93m{stats['current_yawning_duration']:.1f}s\033[0m"
                if stats['is_yawning']:
                    mouth_text += " \033[93m[YAWNING!]\033[0m"
            else:
                mouth_text = "\033[92mNot yawning\033[0m"
            
            # Alert counts
            alert_text = f"Alerts: {stats['eye_alerts_sent']}/{args.max_alerts} (sleep), {stats['yawn_alerts_sent']}/{args.max_alerts} (yawn)"
            
            # Print status
            status_text += f"{eye_text} | {mouth_text} | {alert_text}"
            print(status_text, end="\r")
            
            # Sleep briefly to prevent CPU hogging
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nStopping monitoring...")
    finally:
        # Stop the monitor
        monitor.stop()
        print("Monitoring stopped. Exiting.")


if __name__ == "__main__":
    main()