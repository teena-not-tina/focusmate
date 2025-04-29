import cv2
import numpy as np
import argparse
import os

# Import our new modules
from eye_detection import EyeDetector
from drowsiness import DrowsinessDetector

def process_image(image_path, mode='face'):
    """
    Process an image using different detection modes
    
    Args:
        image_path: Path to the input image
        mode: Detection mode ('face', 'eye', or 'drowsiness')
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"Processing image: {image_path} in {mode} mode")
    
    if mode == 'face':
        # Use your original face detection code
        import insightface
        from insightface.app import FaceAnalysis
        
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        img = cv2.imread(image_path)
        faces = app.get(img)
        
        if len(faces) == 0:
            print("No faces detected")
            return
        
        # Draw face bounding boxes
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        
        # Save result
        output_path = f"output_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, img)
        print(f"Face detection result saved to {output_path}")
        print(f"Detected {len(faces)} faces")
        
        # Display result (optional)
        cv2.imshow("Face Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif mode == 'eye':
        # Use our new eye detection module
        detector = EyeDetector()
        try:
            eye_statuses, result_image = detector.detect_eye_status(image_path, visualize=True)
            
            if eye_statuses:
                print("Eye Detection Results:")
                for i, status in enumerate(eye_statuses):
                    print(f"Face #{i+1}:")
                    print(f"  Left eye EAR: {status['left_ear']:.2f} - {'CLOSED' if status['left_eye_closed'] else 'OPEN'}")
                    print(f"  Right eye EAR: {status['right_ear']:.2f} - {'CLOSED' if status['right_eye_closed'] else 'OPEN'}")
                    print(f"  Both eyes closed: {status['both_eyes_closed']}")
                
                # Output saved by the detector itself
                print("Eye detection result saved to eye_status_result.jpg")
                
                # Display result (optional)
                cv2.imshow("Eye Status Detection", result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("No faces detected for eye status analysis")
                
        except Exception as e:
            print(f"Error in eye detection: {e}")
            
    elif mode == 'drowsiness':
        # Use our drowsiness detection module
        detector = DrowsinessDetector(ear_threshold=0.2, consecutive_frames=3)
        try:
            output_img, alert, stats = detector.process_image(image_path)
            
            print("Drowsiness Detection Results:")
            print(f"  Average EAR: {stats['avg_ear']:.2f}")
            print(f"  Eyes closed: {stats['eyes_closed']}")
            print(f"  Drowsiness alert: {alert}")
            
            # Output saved by the detector itself
            print("Drowsiness detection result saved to drowsiness_detection_result.jpg")
            
            # Display result (optional)
            cv2.imshow("Drowsiness Detection", output_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error in drowsiness detection: {e}")
    
    elif mode == 'webcam':
        # Run real-time drowsiness detection on webcam
        detector = DrowsinessDetector(ear_threshold=0.2, consecutive_frames=3)
        detector.run_webcam()
    
    else:
        print(f"Unknown mode: {mode}")

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Face, Eye, and Drowsiness Detection')
    parser.add_argument('--image', type=str, default='lady.jpg', help='Path to input image')
    parser.add_argument('--mode', type=str, default='face', 
                       choices=['face', 'eye', 'drowsiness', 'webcam'],
                       help='Detection mode')
    
    args = parser.parse_args()
    
    # Process the image or run webcam
    if args.mode == 'webcam':
        process_image(None, mode='webcam')
    else:
        process_image(args.image, mode=args.mode)