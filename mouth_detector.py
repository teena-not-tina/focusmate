import cv2
import numpy as np
import os
from face_detector import FaceDetector

class MouthDistanceDetector(FaceDetector):
    def __init__(self, providers=['CPUExecutionProvider']):
        """
        Initialize detector to measure mouth distance ratio
        to determine if mouth is open or closed.
        """
        super().__init__(providers)
        
        # Key point indices for mouth detection - updated for better vertical measurement
        self.UPPER_LIP = 62  # Upper lip top center
        self.LOWER_LIP = 60  # Lower lip bottom center
        
        # Store lip landmark indices for visualization
        # Using the range 52-71 for all lip landmarks
        self.OUTER_LIP_INDICES = list(range(52, 59))  # Outer lip contour
        self.UPPER_LIP_INDICES = list(range(61, 67))  # Upper lip landmarks
        self.LOWER_LIP_INDICES = list(range(67, 72))  # Lower lip landmarks
        
        print("Mouth distance detector initialized with key points:")
        print(f"  Mouth: upper point {self.UPPER_LIP} (upper lip top center), lower point {self.LOWER_LIP} (lower lip bottom center)")
    
    def calculate_mouth_state(self, landmarks, threshold_ratio=0.05):
        """
        Calculate if mouth is open, closed, or yawning based on lip distance ratio.
        
        Args:
            landmarks: Facial landmarks (106 points)
            threshold_ratio: Ratio threshold below which mouth is considered closed
            
        Returns:
            Dictionary with mouth state information
        """
        # Extract key points
        upper_lip = landmarks[self.UPPER_LIP]
        lower_lip = landmarks[self.LOWER_LIP]
        
        # Calculate vertical distance (mouth openness)
        mouth_distance = np.linalg.norm(upper_lip - lower_lip)
        
        # To normalize for face size, we need reference measurements
        # Face width is a good normalizing factor
        face_width = np.linalg.norm(landmarks[33] - landmarks[96])  # Approximate face width
        
        # Calculate ratio
        mouth_ratio = mouth_distance / face_width if face_width > 0 else 0
        
        # Determine mouth state based on ratio thresholds
        # Closed: ratio < threshold_ratio
        # Yawning: ratio > threshold_ratio * 3 (significantly open)
        # Open but not yawning: in between
        mouth_closed = mouth_ratio < threshold_ratio
        yawning = mouth_ratio > threshold_ratio * 3
        
        mouth_state = "CLOSED" if mouth_closed else ("YAWNING" if yawning else "OPEN")
        
        return {
            'mouth': {
                'distance': mouth_distance,
                'ratio': mouth_ratio,
                'is_closed': mouth_closed,
                'is_yawning': yawning,
                'state': mouth_state,
                'upper_point': tuple(upper_lip),
                'lower_point': tuple(lower_lip)
            },
            'face_width': face_width
        }
    
    def detect_mouth_state(self, image_path, visualize=True, threshold_ratio=0.05):
        """
        Detect the state of mouth (closed, open, or yawning) in an image.
        
        Args:
            image_path: Path to the input image
            visualize: Whether to create a visualization of the results
            threshold_ratio: Ratio threshold below which mouth is considered closed
            
        Returns:
            List of dictionaries with mouth state information for each face
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
            
            # Calculate mouth state based on lip distance
            mouth_state = self.calculate_mouth_state(landmarks, threshold_ratio)
            
            # Add face index to the result
            mouth_state['face_idx'] = i
            
            # Store result
            results.append(mouth_state)
            
            # Visualize if requested
            if visualize:
                self.draw_visualization(output_img, face, landmarks, mouth_state)
        
        # Save the visualization
        if visualize and output_img is not None:
            output_dir = "output_pics"
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_mouth_state.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, output_img)
            print(f"Visualization saved to {output_path}")
        
        return results, output_img
    
    def draw_visualization(self, img, face, landmarks, mouth_state):
        """Draw visualization of mouth state detection"""
        # Draw face bounding box
        # bbox = face.bbox.astype(int)
        # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        
        # # Draw outer lip contour
        # for idx in self.OUTER_LIP_INDICES:
        #     cv2.circle(img, tuple(landmarks[idx]), 1, (0, 255, 255), -1)  # Cyan for outer contour
            
        # # Draw upper lip landmarks
        # for idx in self.UPPER_LIP_INDICES:
        #     cv2.circle(img, tuple(landmarks[idx]), 1, (0, 255, 0), -1)  # Green for upper lip
            
        # # Draw lower lip landmarks
        # for idx in self.LOWER_LIP_INDICES:
        #     cv2.circle(img, tuple(landmarks[idx]), 1, (255, 255, 0), -1)  # Yellow for lower lip
        
        # # Highlight the key points used for measurement
        # upper_lip = mouth_state['mouth']['upper_point']
        # lower_lip = mouth_state['mouth']['lower_point']
        # cv2.circle(img, upper_lip, 3, (255, 0, 0), -1)  # Upper (red)
        # cv2.circle(img, lower_lip, 3, (0, 0, 255), -1)  # Lower (blue)
        # cv2.line(img, upper_lip, lower_lip, (255, 0, 255), 1)  # Line connecting points
        
        # # Add mouth state information
        # mouth_info = mouth_state['mouth']
        
        # y_offset = 30
        
        # # Get color based on state
        # if mouth_info['is_closed']:
        #     color = (0, 0, 255)  # Red for closed
        # elif mouth_info['is_yawning']:
        #     color = (0, 165, 255)  # Orange for yawning
        # else:
        #     color = (0, 255, 0)  # Green for normal open
        
        # # Mouth information
        # m_ratio = mouth_info['ratio']
        # cv2.putText(img, f"Mouth: {m_ratio:.3f} - {mouth_info['state']}", 
        #            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def print_mouth_state_results(results):
    """Pretty print mouth state detection results"""
    if not results:
        print("No results to display")
        return
    
    print("\n===== MOUTH DISTANCE DETECTION RESULTS =====")
    for face in results:
        print(f"\nFace #{face['face_idx'] + 1}:")
        
        # Face size information
        print(f"  Face width: {face['face_width']:.2f} pixels")
        
        # Mouth
        mouth = face['mouth']
        print("\n  MOUTH:")
        print(f"    State: {mouth['state']}")
        print(f"    Lip distance: {mouth['distance']:.2f} pixels")
        print(f"    Distance/Face width ratio: {mouth['ratio']:.4f}")
        print(f"    Upper lip point: {mouth['upper_point']}")
        print(f"    Lower lip point: {mouth['lower_point']}")
        print(f"    Is yawning: {mouth['is_yawning']}")
    
    print("\n============================================")