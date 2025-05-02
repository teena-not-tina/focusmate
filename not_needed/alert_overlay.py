#!/usr/bin/env python3
"""
Transparent overlay window that displays alerts when yawning or sleeping is detected.
This is designed to be non-intrusive while using Focusmate.
"""
import os
import sys
import time
import threading
import pygame
import tkinter as tk
from tkinter import font as tkfont
from datetime import datetime
import argparse

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import our modules
from frame_processor import FacialStateTracker, FrameMonitor
import config

class AlertOverlay:
    """
    Creates a transparent window overlay that shows alerts when triggered.
    """
    def __init__(self, frames_dir=None, yawn_threshold=None, sleep_threshold=None, 
                max_alerts=None, no_sound=False, position='top'):
        """
        Initialize the alert overlay.
        
        Args:
            frames_dir: Directory containing frames to monitor
            yawn_threshold: Seconds of yawning to trigger alert
            sleep_threshold: Seconds of eye closure to trigger alert
            max_alerts: Maximum number of alerts per type
            no_sound: Whether to disable alert sounds
            position: Where to position the overlay ('top', 'bottom', 'top-right', etc.)
        """
        # Configuration
        self.frames_dir = frames_dir or os.path.join("StartPage", "temp_frames")
        self.yawn_threshold = yawn_threshold or config.YAWNING_ALERT_THRESHOLD
        self.sleep_threshold = sleep_threshold or config.EYES_CLOSED_ALERT_THRESHOLD
        self.max_alerts = max_alerts or config.MAX_EYE_ALERTS
        self.no_sound = no_sound
        self.position = position
        
        # Create output directory for processed frames
        self.output_dir = "output_frames"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize sounds
        pygame.mixer.init()
        self.sounds = self._init_sounds()
        
        # Create the tracker and monitor
        self.tracker = FacialStateTracker(
            eye_history_seconds=15,
            mouth_history_seconds=15,
            eyes_closed_alert_threshold=self.sleep_threshold,
            yawning_alert_threshold=self.yawn_threshold,
            max_eye_alerts=self.max_alerts,
            max_yawn_alerts=self.max_alerts
        )
        
        self.monitor = FrameMonitor(frames_dir=self.frames_dir, tracker=self.tracker)
        
        # Initialize Tkinter
        self.root = tk.Tk()
        self.root.title("Focusmate Alert Overlay")
        
        # Make the window transparent
        self.root.attributes("-alpha", 0.85)  # 85% opacity
        self.root.attributes("-topmost", True)  # Keep on top
        
        # Get screen dimensions
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        # Set window size and position based on position parameter
        self._configure_window()
        
        # Configure window border and background
        self.root.overrideredirect(True)  # Remove window decorations
        self.root.configure(bg='black')
        
        # Create alert label
        self.alert_font = tkfont.Font(family="Arial", size=16, weight="bold")
        self.alert_label = tk.Label(
            self.root,
            text="",
            font=self.alert_font,
            bg="black",
            fg="white",
            padx=20,
            pady=10
        )
        self.alert_label.pack(fill=tk.BOTH, expand=True)
        
        # State tracking
        self.should_hide = True  # Start hidden
        self.last_activity = time.time()
        self.currently_visible = False
        
        # Force initial hiding
        self.hide_window()
        
        # Variables to track alert timestamps
        self.last_yawn_alert = 0
        self.last_sleep_alert = 0
        
        # Start monitoring in a separate thread
        self.monitoring = False
    
    def _configure_window(self):
        """Configure window size and position based on position parameter"""
        # Window dimensions
        window_width = int(self.screen_width * 0.5)  # 50% of screen width
        window_height = 60  # Fixed height
        
        # Position based on parameter
        if self.position == 'top':
            x = (self.screen_width - window_width) // 2
            y = 20
        elif self.position == 'bottom':
            x = (self.screen_width - window_width) // 2
            y = self.screen_height - window_height - 60  # 60px from bottom
        elif self.position == 'top-left':
            x = 20
            y = 20
        elif self.position == 'top-right':
            x = self.screen_width - window_width - 20
            y = 20
        elif self.position == 'bottom-left':
            x = 20
            y = self.screen_height - window_height - 60
        elif self.position == 'bottom-right':
            x = self.screen_width - window_width - 20
            y = self.screen_height - window_height - 60
        else:  # Default to top
            x = (self.screen_width - window_width) // 2
            y = 20
        
        # Set window size and position
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    def _init_sounds(self):
        """Initialize alert sounds"""
        sound_dir = "sounds"
        os.makedirs(sound_dir, exist_ok=True)
        
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
        
        return {
            'yawn': yawn_sound_path,
            'sleep': sleep_sound_path
        }
    
    def start(self):
        """Start monitoring and show the overlay"""
        if not self.monitoring:
            # Start the frame monitor
            self.monitor.start()
            
            # Start the alert checking thread
            self.monitoring = True
            self.check_thread = threading.Thread(target=self._check_alerts)
            self.check_thread.daemon = True
            self.check_thread.start()
            
            print(f"Alert overlay started. Monitoring frames in {self.frames_dir}")
            print(f"Thresholds: {self.yawn_threshold}s (yawn), {self.sleep_threshold}s (sleep)")
            
            # Start the Tkinter main loop
            self.root.mainloop()
    
    def stop(self):
        """Stop monitoring and close the overlay"""
        self.monitoring = False
        if self.monitor:
            self.monitor.stop()
        self.root.destroy()
    
    def _check_alerts(self):
        """Background thread to check for alerts"""
        while self.monitoring:
            try:
                # Get the latest alert
                alert = self.monitor.get_latest_alert()
                
                if alert:
                    alert_type = alert['type']
                    message = alert['message']
                    
                    current_time = time.time()
                    
                    if alert_type == 'sleep':
                        # Only process sleep alerts at most every 8 seconds
                        if current_time - self.last_sleep_alert > 8:
                            self.last_sleep_alert = current_time
                            self._show_alert(message, "red")
                            
                            # Play sound if enabled
                            if not self.no_sound and self.sounds['sleep']:
                                self._play_sound(self.sounds['sleep'])
                    
                    elif alert_type == 'yawn':
                        # Only process yawn alerts at most every 8 seconds
                        if current_time - self.last_yawn_alert > 8:
                            self.last_yawn_alert = current_time
                            self._show_alert(message, "orange")
                            
                            # Play sound if enabled
                            if not self.no_sound and self.sounds['yawn']:
                                self._play_sound(self.sounds['yawn'])
                
                # Check current states for minimal display
                stats = self.tracker.get_state_statistics()
                
                # Show mini state indicators if significant durations but not yet alerts
                if (stats['current_eyes_closed_duration'] > 2 and 
                    stats['current_eyes_closed_duration'] < self.sleep_threshold):
                    # Eyes are closed but not long enough for full alert
                    mini_message = f"Eyes closed: {stats['current_eyes_closed_duration']:.1f}s"
                    self._show_mini_state(mini_message, "yellow")
                
                elif (stats['current_yawning_duration'] > 1 and 
                      stats['current_yawning_duration'] < self.yawn_threshold):
                    # Yawning but not long enough for full alert
                    mini_message = f"Yawning: {stats['current_yawning_duration']:.1f}s"
                    self._show_mini_state(mini_message, "yellow")
                
                elif self.currently_visible:
                    # Auto-hide after no significant states
                    elapsed = time.time() - self.last_activity
                    if elapsed > 3:  # Hide after 3 seconds of no significant activity
                        self.hide_window()
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in alert checking thread: {e}")
                time.sleep(0.5)
    
    def _show_alert(self, message, color):
        """Show an alert message in the overlay"""
        self.last_activity = time.time()
        self.should_hide = False
        
        # Update label
        self.alert_label.config(text=message, fg=color, font=self.alert_font)
        
        # Show the window
        if not self.currently_visible:
            self.show_window()
        
        # Schedule auto-hide after 5 seconds
        self.root.after(5000, self._auto_hide)
    
    def _show_mini_state(self, message, color):
        """Show a mini state indication in the overlay"""
        self.last_activity = time.time()
        
        # Smaller font for mini state
        mini_font = tkfont.Font(family="Arial", size=14)
        
        # Update label
        self.alert_label.config(text=message, fg=color, font=mini_font)
        
        # Show the window if not visible
        if not self.currently_visible:
            self.show_window()
    
    def _auto_hide(self):
        """Auto-hide the window after alerts"""
        elapsed = time.time() - self.last_activity
        if elapsed >= 4.9:  # Close enough to 5 seconds
            self.hide_window()
    
    def show_window(self):
        """Make the window visible"""
        self.root.deiconify()
        self.currently_visible = True
    
    def hide_window(self):
        """Hide the window"""
        self.root.withdraw()
        self.currently_visible = False
    
    def _play_sound(self, sound_path):
        """Play an alert sound"""
        try:
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Failed to play sound: {e}")


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Focusmate Alert Overlay")
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
    parser.add_argument("--position", type=str, default="top",
                       choices=["top", "bottom", "top-left", "top-right", "bottom-left", "bottom-right"],
                       help="Position of the overlay on screen")
    
    args = parser.parse_args()
    
    # Create and start the overlay
    overlay = AlertOverlay(
        frames_dir=args.frames_dir,
        yawn_threshold=args.yawn_threshold,
        sleep_threshold=args.sleep_threshold,
        max_alerts=args.max_alerts,
        no_sound=args.no_sound,
        position=args.position
    )
    
    try:
        overlay.start()
    except KeyboardInterrupt:
        print("\nStopping overlay...")
    finally:
        overlay.stop()
        print("Overlay stopped. Exiting.")


if __name__ == "__main__":
    main()