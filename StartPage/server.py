from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import threading
import time
import os
import signal
import sys
import numpy as np
from datetime import datetime
from queue import Queue
from threading import Lock
import config
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Now you can import modules from the parent directory
from frame_processor import FacialStateTracker
import config  # This assumes config.py is also in the StartPage directory

# Try to import config, but define defaults if it fails
try:
    import config
except (ImportError, AttributeError):
    # Define a simple config class with default values
    class ConfigDefaults:
        TEMP_DIR = "temp_frames"
        SAVE_DIR = "saved_faces"
        # Add other missing settings here
    
    # Check if config exists but is missing attributes
    if 'config' in sys.modules:
        # Add the missing attributes
        if not hasattr(config, 'TEMP_DIR'):
            config.TEMP_DIR = "temp_frames"
        if not hasattr(config, 'SAVE_DIR'):
            config.SAVE_DIR = "saved_faces"
        # Add other missing settings here
    else:
        # No config imported at all, use our defaults
        config = ConfigDefaults()


# Flask server initialization
app = Flask(__name__)
CORS(app)

# Create necessary directories
os.makedirs(config.TEMP_DIR, exist_ok=True)
os.makedirs(config.SAVE_DIR, exist_ok=True)

# Global variables
camera = None
frame = None
processed_frame = None
running = True
last_seen_time = time.time()
facial_state_tracker = None
alerts = {}
saving_frames = False

# Locks for thread-safe access
frame_lock = Lock()
alerts_lock = Lock()
save_queue = Queue(maxsize=100)  # Limit queue size to prevent memory issues

# Ctrl+C handler
def signal_handler(sig, frame):
    global running
    print("\n[🔚 Shutdown requested] Releasing camera...")
    if camera:
        camera.release()
    running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Initialize the system
def initialize_system():
    global camera, facial_state_tracker
    try:
        # Initialize camera
        camera = cv2.VideoCapture(config.CAMERA_ID, cv2.CAP_AVFOUNDATION)  # For macOS
        time.sleep(1)  # Give the camera a moment to initialize

        if not camera.isOpened():
            print(f"[❌ Error] Failed to open camera {config.CAMERA_ID}")
            return False
            
        # Initialize facial state tracker with configuration parameters
        facial_state_tracker = FacialStateTracker(
            eye_history_seconds=config.EYE_HISTORY_SECONDS,
            mouth_history_seconds=config.MOUTH_HISTORY_SECONDS,
            eyes_closed_alert_threshold=config.EYES_CLOSED_ALERT_THRESHOLD,
            yawning_alert_threshold=config.YAWNING_ALERT_THRESHOLD,
            max_eye_alerts=config.MAX_EYE_ALERTS,
            max_yawn_alerts=config.MAX_YAWN_ALERTS,
            eye_threshold=config.EYE_THRESHOLD,
            mouth_threshold=config.MOUTH_THRESHOLD
        )
        
        print("[✅ System initialized successfully]")
        return True
    except Exception as e:
        print(f"[❌ Initialization error] {str(e)}")
        return False

# Thread for capturing frames
def capture_frame():
    global frame, camera
    
    if not camera or not camera.isOpened():
        print("[❌ Error] Camera not initialized in capture thread")
        return
        
    print("[🎥 Frame capture thread started]")
    frame_interval = 1.0 / 30  # Try to capture at 30 FPS
    
    while running:
        start_time = time.time()
        success, captured_frame = camera.read()
        if success:
            with frame_lock:
                frame = np.copy(captured_frame)
        else:
            print("[⚠️ Warning] Failed to capture frame")
            # Try to reinitialize camera if it fails
            camera.release()
            time.sleep(1)
            camera = cv2.VideoCapture(config.CAMERA_ID, cv2.CAP_DSHOW)
            
        # Calculate time to sleep to maintain target FPS
        elapsed = time.time() - start_time
        sleep_time = max(0, frame_interval - elapsed)
        time.sleep(sleep_time)

# Thread for processing frames
def process_frames():
    global frame, processed_frame, facial_state_tracker, alerts
    
    if not facial_state_tracker:
        print("[❌ Error] Facial state tracker not initialized")
        return
        
    print("[🔍 Frame processing thread started]")
    process_interval = 1.0 / config.PROCESSING_FPS
    
    while running:
        start_time = time.time()
        current_frame = None
        
        # Get the current frame
        with frame_lock:
            if frame is not None:
                current_frame = np.copy(frame)
        
        # Process if we have a frame
        if current_frame is not None:
            try:
                # Process the frame
                new_processed_frame, new_alerts = facial_state_tracker.process_frame(current_frame)
                
                # Update processed frame
                with frame_lock:
                    processed_frame = new_processed_frame
                
                # Update alerts if there are any new ones
                if new_alerts:
                    with alerts_lock:
                        alerts.update(new_alerts)
                        
                # Add to save queue if saving is enabled
                if saving_frames and not save_queue.full():
                    if new_processed_frame is not None:
                        save_queue.put((np.copy(new_processed_frame), datetime.now()))
                    
            except Exception as e:
                print(f"[❌ Error processing frame] {str(e)}")
        
        # Calculate time to sleep to maintain target FPS
        elapsed = time.time() - start_time
        sleep_time = max(0, process_interval - elapsed)
        time.sleep(sleep_time)

# Thread for saving frames
def save_frames_worker():
    print("[💾 Frame saving thread started]")
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), config.STREAM_QUALITY]
    
    while running:
        if not save_queue.empty():
            try:
                frame_data, timestamp = save_queue.get()
                
                # Generate filename with timestamp
                filename = f"frame_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                filepath = os.path.join(config.SAVE_DIR, filename)
                
                # Save the frame
                success = cv2.imwrite(filepath, frame_data, encode_param)
                if success:
                    print(f"[✅] Saved frame to {filename}")
                else:
                    print(f"[❌] Failed to save frame {filename}")
            except Exception as e:
                print(f"[❌ Error saving frame] {str(e)}")
                
        time.sleep(0.01)  # Check queue frequently

# Camera stream route - provides processed video feed
@app.route('/camera_stream')
def camera_stream():
    def generate():
        global processed_frame
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), config.STREAM_QUALITY]
        stream_interval = 1.0 / config.STREAMING_FPS
        
        while running:
            start_time = time.time()
            output_frame = None
            
            # Get the latest processed frame, or fall back to raw frame
            with frame_lock:
                if processed_frame is not None:
                    output_frame = np.copy(processed_frame)
                elif frame is not None:
                    output_frame = np.copy(frame)
            
            # Skip if no frame available
            if output_frame is None:
                time.sleep(0.03)
                continue
            
            # Encode frame for streaming
            success, buffer = cv2.imencode('.jpg', output_frame, encode_param)
            if not success:
                continue
                
            # Send frame
            frame_bytes = buffer.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )
            
            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, stream_interval - elapsed)
            time.sleep(sleep_time)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# API: Get alerts
@app.route('/get_alerts')
def get_alerts():
    global alerts
    
    with alerts_lock:
        current_alerts = alerts.copy()
        alerts = {}  # Clear after retrieving
    
    return jsonify({'alerts': current_alerts})

# API: Reset alert counters
@app.route('/reset_alerts', methods=['POST'])
def reset_alerts():
    global facial_state_tracker
    
    if facial_state_tracker:
        facial_state_tracker.reset_alerts()
        return jsonify({'status': 'success', 'message': 'Alert counters reset'})
    else:
        return jsonify({'status': 'error', 'message': 'Facial state tracker not initialized'})

# API: Get state statistics
@app.route('/get_statistics')
def get_statistics():
    global facial_state_tracker
    
    if facial_state_tracker:
        stats = facial_state_tracker.get_state_statistics()
        return jsonify({'status': 'success', 'statistics': stats})
    else:
        return jsonify({'status': 'error', 'message': 'Facial state tracker not initialized'})

# API: Start saving frames
@app.route('/start_saving', methods=['GET', 'POST'])
def start_saving():
    global saving_frames
    saving_frames = True
    print("[▶️] Frame saving started")
    return jsonify({'status': 'started'})

# API: Stop saving frames
@app.route('/stop_saving', methods=['GET', 'POST'])
def stop_saving():
    global saving_frames
    saving_frames = False
    print("[⏹️] Frame saving stopped")
    return jsonify({'status': 'stopped'})

# API: Heartbeat to keep system running
@app.route('/heartbeat')
def heartbeat():
    global last_seen_time
    last_seen_time = time.time()
    return jsonify({'status': 'ok'})

# API: Process input from external source
@app.route('/process_frame', methods=['POST'])
def process_external_frame():
    if not request.files or 'frame' not in request.files:
        return jsonify({'status': 'error', 'message': 'No frame provided'})
    
    try:
        # Read the uploaded frame
        frame_file = request.files['frame']
        frame_data = frame_file.read()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Update the current frame
        with frame_lock:
            global frame
            frame = img
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# Status info endpoint
@app.route('/status')
def get_status():
    status = {
        'running': running,
        'camera_initialized': camera is not None and camera.isOpened(),
        'tracker_initialized': facial_state_tracker is not None,
        'saving_frames': saving_frames,
        'last_heartbeat_seconds_ago': time.time() - last_seen_time,
        'config': {
            'eye_threshold': config.EYE_THRESHOLD,
            'mouth_threshold': config.MOUTH_THRESHOLD,
            'eyes_closed_alert_threshold': config.EYES_CLOSED_ALERT_THRESHOLD,
            'yawning_alert_threshold': config.YAWNING_ALERT_THRESHOLD,
            'max_eye_alerts': config.MAX_EYE_ALERTS,
            'max_yawn_alerts': config.MAX_YAWN_ALERTS
        }
    }
    return jsonify(status)

# Server startup
if __name__ == '__main__':
    print("="*50)
    print("Starting Driver Alertness Monitoring System")
    print("="*50)
    
    if initialize_system():
        # Start background threads
        threading.Thread(target=capture_frame, daemon=True).start()
        threading.Thread(target=process_frames, daemon=True).start()
        threading.Thread(target=save_frames_worker, daemon=True).start()
        
        print(f"Starting server on {config.HOST}:{config.PORT}")
        # Start Flask server
        app.run(host=config.HOST, port=config.PORT, threaded=True)
    else:
        print("[❌ Failed to initialize system. Exiting.]")
        sys.exit(1)