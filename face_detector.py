import cv2
import insightface
from insightface.app import FaceAnalysis

class FaceDetector:
    """Base class for face detection using InsightFace"""
    
    def __init__(self, providers=['CPUExecutionProvider']):
        """Initialize face detector with InsightFace"""
        # Initialize FaceAnalysis with landmark detection
        self.app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'], 
                               providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("Face detector initialized")
    
    def detect_faces(self, image_path):
        """Detect faces and landmarks in an image"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Detect faces
        faces = self.app.get(img)
        
        if len(faces) == 0:
            print("No faces detected")
            return None, None
            
        return faces, img