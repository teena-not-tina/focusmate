# Configuration settings for the driver alertness monitoring system

# Detection thresholds
EYE_THRESHOLD = 0.05  # Ratio threshold for eye closure detection (smaller = more sensitive)
MOUTH_THRESHOLD = 0.1  # Ratio threshold for mouth closure detection (smaller = more sensitive)

# Alert thresholds
EYES_CLOSED_ALERT_THRESHOLD = 10  # Seconds of continuous eye closure to trigger alert
YAWNING_ALERT_THRESHOLD = 4  # Seconds of continuous yawning to trigger alert

# Maximum number of alerts
MAX_EYE_ALERTS = 2  # Maximum number of eye closure alerts
MAX_YAWN_ALERTS = 2  # Maximum number of yawning alerts

# History tracking (in seconds)
EYE_HISTORY_SECONDS = 15  # Number of seconds to keep eye state history
MOUTH_HISTORY_SECONDS = 15  # Number of seconds to keep mouth state history

# Camera settings
CAMERA_ID = 0  # Camera device ID
STREAM_QUALITY = 90  # JPEG quality for streaming (1-100)

# Server settings
HOST = '0.0.0.0'  # Host address (0.0.0.0 for all interfaces)
PORT = 8080  # Server port

# File paths
TEMP_DIR = "temp_frames"  # Directory for temporary frame storage
SAVE_DIR = "saved_faces"  # Directory for saved face images

# Processing settings
PROCESSING_FPS = 10  # Target FPS for facial analysis
STREAMING_FPS = 25  # Target FPS for video streaming