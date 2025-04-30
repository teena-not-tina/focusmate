import cv2
import numpy as np
import os
from face_detector import FaceDetector

class EyelidDistanceDetector(FaceDetector):
    def __init__(self, providers=['CPUExecutionProvider']):
        """
        Initialize detector to measure eyelid distance ratio
        to determine if eyes are open or closed.
        """
        super().__init__(providers)
        
        # Key point indices for eyelid detection
        # Upper eyelid points and lower eyelid points for each eye
        self.RIGHT_EYE_UPPER = 33  # Upper eyelid (right eye)
        self.RIGHT_EYE_LOWER = 40  # Lower eyelid (right eye)
        self.LEFT_EYE_UPPER = 87   # Upper eyelid (left eye)
        self.LEFT_EYE_LOWER = 94   # Lower eyelid (left eye)
        
        # Store all eye landmark indices for visualization
        self.RIGHT_EYE_INDICES = list(range(33, 43))
        self.LEFT_EYE_INDICES = list(range(87, 97))
        
        print("Eyelid distance detector initialized with key points:")
        print(f"  Right eye: upper point {self.RIGHT_EYE_UPPER}, lower point {self.RIGHT_EYE_LOWER}")
        print(f"  Left eye: upper point {self.LEFT_EYE_UPPER}, lower point {self.LEFT_EYE_LOWER}")
    
    def calculate_eye_state(self, landmarks, threshold_ratio=0.05):
        """
        Calculate if eyes are open or closed based on eyelid distance ratio.
        
        Args:
            landmarks: Facial landmarks (106 points)
            threshold_ratio: Ratio threshold below which eyes are considered closed
            
        Returns:
            Dictionary with eye state information
        """
        # Extract key points
        right_upper = landmarks[self.RIGHT_EYE_UPPER]
        right_lower = landmarks[self.RIGHT_EYE_LOWER]
        left_upper = landmarks[self.LEFT_EYE_UPPER]
        left_lower = landmarks[self.LEFT_EYE_LOWER]
        
        # Calculate vertical distances (eyelid openness)
        right_eye_distance = np.linalg.norm(right_upper - right_lower)
        left_eye_distance = np.linalg.norm(left_upper - left_lower)
        
        # To normalize for face size, we need reference measurements
        # Face width is a good normalizing factor
        face_width = np.linalg.norm(landmarks[33] - landmarks[96])  # Approximate face width
        
        # Calculate ratios
        right_eye_ratio = right_eye_distance / face_width if face_width > 0 else 0
        left_eye_ratio = left_eye_distance / face_width if face_width > 0 else 0
        
        # Determine if eyes are closed based on ratio threshold
        right_eye_closed = right_eye_ratio < threshold_ratio
        left_eye_closed = left_eye_ratio < threshold_ratio
        
        return {
            'right_eye': {
                'distance': right_eye_distance,
                'ratio': right_eye_ratio,
                'is_closed': right_eye_closed,
                'upper_point': tuple(right_upper),
                'lower_point': tuple(right_lower)
            },
            'left_eye': {
                'distance': left_eye_distance,
                'ratio': left_eye_ratio,
                'is_closed': left_eye_closed,
                'upper_point': tuple(left_upper),
                'lower_point': tuple(left_lower)
            },
            'face_width': face_width,
            'both_eyes_closed': right_eye_closed and left_eye_closed
        }
    
    def detect_eye_state(self, image_path, visualize=True, threshold_ratio=0.05):
        """
        Detect the state of eyes (open or closed) in an image.
        
        Args:
            image_path: Path to the input image
            visualize: Whether to create a visualization of the results
            threshold_ratio: Ratio threshold below which eyes are considered closed
            
        Returns:
            List of dictionaries with eye state information for each face
        """
        # Detect faces
        faces, img = self.detect_faces(image_path)
        
        if faces is None:
            return None, None
        
        output_img = img.copy() if visualize else None
        results = []
        
        # Process each face
        for i, face in enumerate(faces):
            landmarks = face.landmark_2d_106
            landmarks = np.round(landmarks).astype(int)
            
            # Calculate eye state based on eyelid distance
            eye_state = self.calculate_eye_state(landmarks, threshold_ratio)
            
            # Add face index to the result
            eye_state['face_idx'] = i
            
            # Store result
            results.append(eye_state)
            
            # Visualize if requested
            if visualize:
                self.draw_visualization(output_img, face, landmarks, eye_state)
        
        # Save the visualization
        if visualize and output_img is not None:
            output_path = os.path.splitext(os.path.basename(image_path))[0] + "_eye_state.jpg"
            cv2.imwrite(output_path, output_img)
            print(f"Visualization saved to {output_path}")
        
        return results, output_img
    
    def draw_visualization(self, img, face, landmarks, eye_state):
        """Draw visualization of eye state detection"""
        # Draw face bounding box
        bbox = face.bbox.astype(int)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        
        # Draw all eye landmarks
        for idx in self.RIGHT_EYE_INDICES:
            cv2.circle(img, tuple(landmarks[idx]), 1, (0, 255, 0), -1)
        
        for idx in self.LEFT_EYE_INDICES:
            cv2.circle(img, tuple(landmarks[idx]), 1, (0, 255, 0), -1)
        
        # Highlight the key points used for measurement
        # Right eye
        right_upper = eye_state['right_eye']['upper_point']
        right_lower = eye_state['right_eye']['lower_point']
        cv2.circle(img, right_upper, 3, (255, 0, 0), -1)  # Upper (red)
        cv2.circle(img, right_lower, 3, (0, 0, 255), -1)  # Lower (blue)
        cv2.line(img, right_upper, right_lower, (255, 0, 255), 1)  # Line connecting points
        
        # Left eye
        left_upper = eye_state['left_eye']['upper_point']
        left_lower = eye_state['left_eye']['lower_point']
        cv2.circle(img, left_upper, 3, (255, 0, 0), -1)  # Upper (red)
        cv2.circle(img, left_lower, 3, (0, 0, 255), -1)  # Lower (blue)
        cv2.line(img, left_upper, left_lower, (255, 0, 255), 1)  # Line connecting points
        
        # Add eye state information
        right_status = "CLOSED" if eye_state['right_eye']['is_closed'] else "OPEN"
        left_status = "CLOSED" if eye_state['left_eye']['is_closed'] else "OPEN"
        
        y_offset = 30
        
        # Right eye information
        r_ratio = eye_state['right_eye']['ratio']
        cv2.putText(img, f"Right eye: {r_ratio:.3f} - {right_status}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 0, 255) if eye_state['right_eye']['is_closed'] else (0, 255, 0), 2)
        y_offset += 30
        
        # Left eye information
        l_ratio = eye_state['left_eye']['ratio']
        cv2.putText(img, f"Left eye: {l_ratio:.3f} - {left_status}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 0, 255) if eye_state['left_eye']['is_closed'] else (0, 255, 0), 2)


def print_eye_state_results(results):
    """Pretty print eye state detection results"""
    if not results:
        print("No results to display")
        return
    
    print("\n===== EYELID DISTANCE DETECTION RESULTS =====")
    for face in results:
        print(f"\nFace #{face['face_idx'] + 1}:")
        
        # Face size information
        print(f"  Face width: {face['face_width']:.2f} pixels")
        
        # Right eye
        right_eye = face['right_eye']
        print("\n  RIGHT EYE:")
        print(f"    Status: {'CLOSED' if right_eye['is_closed'] else 'OPEN'}")
        print(f"    Eyelid distance: {right_eye['distance']:.2f} pixels")
        print(f"    Distance/Face width ratio: {right_eye['ratio']:.4f}")
        print(f"    Upper eyelid point: {right_eye['upper_point']}")
        print(f"    Lower eyelid point: {right_eye['lower_point']}")
        
        # Left eye
        left_eye = face['left_eye']
        print("\n  LEFT EYE:")
        print(f"    Status: {'CLOSED' if left_eye['is_closed'] else 'OPEN'}")
        print(f"    Eyelid distance: {left_eye['distance']:.2f} pixels")
        print(f"    Distance/Face width ratio: {left_eye['ratio']:.4f}")
        print(f"    Upper eyelid point: {left_eye['upper_point']}")
        print(f"    Lower eyelid point: {left_eye['lower_point']}")
        
        # Overall status
        print(f"\n  Both eyes closed: {face['both_eyes_closed']}")
    
    print("\n============================================")