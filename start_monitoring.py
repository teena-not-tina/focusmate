#!/usr/bin/env python3
"""
Combined script to start the Focusmate alertness monitoring system.
This script starts the server for frame capture and runs the alertness monitoring
with configurable options.
"""
import os
import sys
import time
import threading
import subprocess
import argparse
import signal
import atexit

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the path
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import config to get default values
try:
    import config
except ImportError:
    # Define some defaults if config can't be imported
    class DefaultConfig:
        YAWNING_ALERT_THRESHOLD = 4.0
        EYES_CLOSED_ALERT_THRESHOLD = 10.0
        MAX_EYE_ALERTS = 2
        MAX_YAWN_ALERTS = 2
    config = DefaultConfig()

# Process management
processes = []

def start_server():
    """Start the server for frame capture"""
    server_path = os.path.join(script_dir, "StartPage", "server.py")
    if not os.path.exists(server_path):
        print(f"Server script not found at {server_path}")
        return None
    
    print("Starting server for frame capture...")
    try:
        server_process = subprocess.Popen([sys.executable, server_path], 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE)
        processes.append(server_process)
        print("Server started.")
        return server_process
    except Exception as e:
        print(f"Failed to start server: {e}")
        return None

def start_alert_overlay(args):
    """Start the alert overlay"""
    overlay_path = os.path.join(script_dir, "alert_overlay.py")
    if not os.path.exists(overlay_path):
        print(f"Alert overlay script not found at {overlay_path}")
        return None
    
    print("Starting alert overlay...")
    try:
        # Build command with arguments
        cmd = [sys.executable, overlay_path]
        
        if args.frames_dir:
            cmd.extend(["--frames-dir", args.frames_dir])
        
        if args.yawn_threshold:
            cmd.extend(["--yawn-threshold", str(args.yawn_threshold)])
        
        if args.sleep_threshold:
            cmd.extend(["--sleep-threshold", str(args.sleep_threshold)])
        
        if args.max_alerts:
            cmd.extend(["--max-alerts", str(args.max_alerts)])
        
        if args.no_sound:
            cmd.append("--no-sound")
        
        if args.position:
            cmd.extend(["--position", args.position])
        
        overlay_process = subprocess.Popen(cmd, 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE)
        processes.append(overlay_process)
        print("Alert overlay started.")
        return overlay_process
    except Exception as e:
        print(f"Failed to start alert overlay: {e}")
        return None

def start_console_monitor(args):
    """Start the console-based monitoring"""
    monitor_path = os.path.join(script_dir, "run_tracker.py")
    if not os.path.exists(monitor_path):
        print(f"Console monitor script not found at {monitor_path}")
        return None
    
    print("Starting console-based monitoring...")
    try:
        # Build command with arguments
        cmd = [sys.executable, monitor_path]
        
        if args.frames_dir:
            cmd.extend(["--frames-dir", args.frames_dir])
        
        if args.yawn_threshold:
            cmd.extend(["--yawn-threshold", str(args.yawn_threshold)])
        
        if args.sleep_threshold:
            cmd.extend(["--sleep-threshold", str(args.sleep_threshold)])
        
        if args.max_alerts:
            cmd.extend(["--max-alerts", str(args.max_alerts)])
        
        if args.no_sound:
            cmd.append("--no-sound")
        
        monitor_process = subprocess.Popen(cmd, 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE)
        processes.append(monitor_process)
        print("Console monitor started.")
        return monitor_process
    except Exception as e:
        print(f"Failed to start console monitor: {e}")
        return None

def cleanup():
    """Terminate all processes on exit"""
    print("\nCleaning up processes...")
    for process in processes:
        if process and process.poll() is None:
            try:
                process.terminate()
                # Give it a moment to terminate gracefully
                time.sleep(0.5)
                if process.poll() is None:
                    # If still running, force kill
                    process.kill()
            except Exception as e:
                print(f"Error terminating process: {e}")
    print("All processes terminated.")

def handle_signal(sig, frame):
    """Handle interrupt signals"""
    print("\nReceived signal to terminate...")
    cleanup()
    sys.exit(0)

def main():
    """Main function"""
    # Set up cleanup on exit
    atexit.register(cleanup)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Focusmate Alertness Monitoring System")
    parser.add_argument("--frames-dir", type=str, default="./StartPage/temp_frames",
                       help="Directory for frame storage")
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
    parser.add_argument("--console-only", action="store_true",
                       help="Run console monitor instead of overlay")
    parser.add_argument("--no-server", action="store_true",
                       help="Don't start the server (use if it's already running)")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Focusmate Alertness Monitoring".center(60))
    print("="*60)
    print(f"Frame directory: {args.frames_dir}")
    print(f"Yawn threshold: {args.yawn_threshold} seconds")
    print(f"Sleep threshold: {args.sleep_threshold} seconds")
    print(f"Maximum alerts: {args.max_alerts}")
    print(f"Sound alerts: {'Disabled' if args.no_sound else 'Enabled'}")
    if not args.console_only:
        print(f"Overlay position: {args.position}")
    print("-"*60)
    
    # Create frames directory if it doesn't exist
    os.makedirs(args.frames_dir, exist_ok=True)
    
    # Start the server if not disabled
    if not args.no_server:
        server_process = start_server()
        if not server_process:
            print("Failed to start server. Exiting.")
            return 1
        
        # Give the server a moment to start
        print("Waiting for server to start...")
        time.sleep(2)
    
    # Start either the overlay or console monitor
    if args.console_only:
        monitor_process = start_console_monitor(args)
        if not monitor_process:
            print("Failed to start console monitor. Exiting.")
            return 1
        
        # Stream console output for the monitor
        print("\nConsole monitor output:")
        print("-"*60)
        
        try:
            while True:
                output = monitor_process.stdout.readline()
                if output:
                    print(output.decode().strip())
                
                if monitor_process.poll() is not None:
                    break
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
    else:
        overlay_process = start_alert_overlay(args)
        if not overlay_process:
            print("Failed to start alert overlay. Exiting.")
            return 1
        
        # Keep the main thread running
        print("\nMonitoring system running. Press Ctrl+C to exit.")
        
        try:
            while True:
                # Check if processes are still running
                all_running = True
                for process in processes:
                    if process.poll() is not None:
                        all_running = False
                        print(f"Process terminated with code {process.returncode}")
                
                if not all_running:
                    print("One or more processes terminated unexpectedly. Exiting.")
                    break
                
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())