import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import argparse

class EyeDetector:
    def __init__(self, left_eye_indices=None, right_eye_indices=None, providers=['CPUExecutionProvider']):
        """
        Initialize EyeDetector with flexible eye landmark indices
        
        Args:
            left_eye_indices: Custom indices for left eye. Default is range(87, 97)
            right_eye_indices: Custom indices for right eye. Default is range(33, 43)
            providers: List of execution providers for InsightFace
        """
        # Initialize FaceAnalysis with landmark detection
        self.app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'], 
                               providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Set eye landmark indices - using the user-suggested indices by default
        self.LEFT_EYE_INDICES = list(range(87, 97)) if left_eye_indices is None else left_eye_indices
        self.RIGHT_EYE_INDICES = list(range(33, 43)) if right_eye_indices is None else right_eye_indices
        
        print(f"Using landmark indices:")
        print(f"  Left eye: {self.LEFT_EYE_INDICES}")
        print(f"  Right eye: {self.RIGHT_EYE_INDICES}")
    
    def visualize_landmark_groups(self, image_path, output_path=None):
        """
        Create a visualization that shows different landmark groups with different colors
        to help identify the correct indices for eyes.
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Detect faces
        faces = self.app.get(img)
        if len(faces) == 0:
            print("No faces detected")
            return None
        
        # Make a copy of the image for visualization
        output_img = img.copy()
        
        # Define color groups for visualization
        color_groups = [
            {"indices": range(0, 33), "color": (150, 150, 150), "name": "Jawline (0-32)"},  # Grey
            {"indices": range(33, 43), "color": (0, 165, 255), "name": "Group 33-42"},  # Orange
            {"indices": range(43, 53), "color": (0, 255, 255), "name": "Group 43-52"},  # Yellow
            {"indices": range(53, 72), "color": (255, 0, 0), "name": "Group 53-71"},   # Blue
            {"indices": range(72, 84), "color": (0, 255, 0), "name": "Group 72-83"},   # Green
            {"indices": range(84, 96), "color": (255, 0, 255), "name": "Group 84-95"}, # Purple
            {"indices": range(96, 106), "color": (128, 0, 128), "name": "Group 96-105"}, # Dark purple
            {"indices": range(87, 97), "color": (255, 255, 0), "name": "Group 87-96"}, # Cyan
        ]
        
        # Draw all landmarks with their group colors
        for face in faces:
            landmarks = face.landmark_2d_106
            landmarks = np.round(landmarks).astype(int)
            
            # Draw face bounding box
            bbox = face.bbox.astype(int)
            cv2.rectangle(output_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 2)
            
            # Draw landmarks by group
            for group in color_groups:
                for i in group["indices"]:
                    if i < len(landmarks):
                        cv2.circle(output_img, tuple(landmarks[i]), 2, group["color"], -1)
            
            # Add legend
            y_offset = 30
            for group in color_groups:
                cv2.putText(output_img, group["name"], (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, group["color"], 2)
                y_offset += 30
        
        # Save or return the output
        if output_path is None:
            output_path = os.path.splitext(os.path.basename(image_path))[0] + "_landmark_groups.jpg"
        
        cv2.imwrite(output_path, output_img)
        print(f"Landmark group visualization saved to {output_path}")
        
        return output_img
    
    def get_eye_landmarks(self, image_path):
        """Extract eye landmarks from an image."""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Detect faces
        faces = self.app.get(img)
        
        if len(faces) == 0:
            print("No faces detected")
            return None, None, img
            
        # Get eye landmarks for each face
        results = []
        for face in faces:
            landmarks = face.landmark_2d_106
            landmarks = np.round(landmarks).astype(int)
            
            # Extract eye landmarks
            left_eye = landmarks[self.LEFT_EYE_INDICES]
            right_eye = landmarks[self.RIGHT_EYE_INDICES]
            
            results.append({
                'left_eye': left_eye,
                'right_eye': right_eye,
                'face': face
            })
            
        return results, faces, img
    
    def calculate_ear(self, eye_points):
        """
        Calculate Eye Aspect Ratio (EAR) to determine if eye is open or closed.
        """
        if len(eye_points) >= 5:  # Need at least a few points
            # Find top and bottom points
            top_idx = np.argmin(eye_points[:, 1])
            bottom_idx = np.argmax(eye_points[:, 1])
            top_point = eye_points[top_idx]
            bottom_point = eye_points[bottom_idx]
            
            # Find leftmost and rightmost points
            left_idx = np.argmin(eye_points[:, 0])
            right_idx = np.argmax(eye_points[:, 0])
            left_point = eye_points[left_idx]
            right_point = eye_points[right_idx]
            
            # Calculate vertical distance (height)
            v = np.linalg.norm(top_point - bottom_point)
            
            # Calculate horizontal distance (width)
            h = np.linalg.norm(left_point - right_point)
            
            # Calculate EAR
            if h > 0:
                ear = v / h
            else:
                ear = 0.0
                
            return ear
        return 0.0
    
    def is_eye_closed(self, eye_points, threshold=0.2):
        """Determine if eye is closed based on EAR threshold."""
        ear = self.calculate_ear(eye_points)
        return ear < threshold, ear
    
    def get_extremity_points(self, eye_points):
        """Get top, bottom, left, and right points of the eye"""
        top_idx = np.argmin(eye_points[:, 1])
        bottom_idx = np.argmax(eye_points[:, 1])
        left_idx = np.argmin(eye_points[:, 0])
        right_idx = np.argmax(eye_points[:, 0])
        
        return {
            "top": tuple(eye_points[top_idx]),
            "bottom": tuple(eye_points[bottom_idx]),
            "left": tuple(eye_points[left_idx]),
            "right": tuple(eye_points[right_idx])
        }
    
    def detect_eyes(self, image_path, visualize=True, threshold=0.2):
        """
        Detect eyes and their open/closed state in an image
        
        Args:
            image_path: Path to the input image
            visualize: Whether to create a visualization of the results
            threshold: EAR threshold for determining if eyes are closed
            
        Returns:
            Dictionary with eye detection results
        """
        # Get landmarks
        results, faces, img = self.get_eye_landmarks(image_path)
        
        if results is None:
            return None, None
        
        output_img = img.copy() if visualize else None
        processed_results = []
        
        # Process each face
        for face_idx, result in enumerate(results):
            left_eye = result['left_eye']
            right_eye = result['right_eye']
            face = result['face']
            
            # Calculate EAR for both eyes
            left_closed, left_ear = self.is_eye_closed(left_eye, threshold)
            right_closed, right_ear = self.is_eye_closed(right_eye, threshold)
            
            # Get extremity points
            left_extremities = self.get_extremity_points(left_eye)
            right_extremities = self.get_extremity_points(right_eye)
            
            # Calculate eye dimensions
            left_height = np.linalg.norm(np.array(left_extremities["top"]) - np.array(left_extremities["bottom"]))
            left_width = np.linalg.norm(np.array(left_extremities["left"]) - np.array(left_extremities["right"]))
            right_height = np.linalg.norm(np.array(right_extremities["top"]) - np.array(right_extremities["bottom"]))
            right_width = np.linalg.norm(np.array(right_extremities["left"]) - np.array(right_extremities["right"]))
            
            # Create result dictionary
            face_result = {
                'face_idx': face_idx,
                'left_eye': {
                    'landmarks': left_eye.tolist(),
                    'ear': left_ear,
                    'is_closed': left_closed,
                    'extremities': left_extremities,
                    'height': left_height,
                    'width': left_width
                },
                'right_eye': {
                    'landmarks': right_eye.tolist(),
                    'ear': right_ear,
                    'is_closed': right_closed,
                    'extremities': right_extremities,
                    'height': right_height,
                    'width': right_width
                },
                'both_eyes_closed': left_closed and right_closed
            }
            
            processed_results.append(face_result)
            
            # Visualize if requested
            if visualize:
                # Draw face bounding box
                bbox = face.bbox.astype(int)
                cv2.rectangle(output_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                
                # Draw eye landmarks
                for point in left_eye:
                    cv2.circle(output_img, tuple(point), 2, (0, 255, 0), -1)  # Green
                for point in right_eye:
                    cv2.circle(output_img, tuple(point), 2, (0, 255, 0), -1)  # Green
                
                # Draw extremity points with different colors
                # Left eye
                cv2.circle(output_img, left_extremities["top"], 3, (255, 0, 0), -1)  # Top (red)
                cv2.circle(output_img, left_extremities["bottom"], 3, (0, 0, 255), -1)  # Bottom (blue)
                cv2.circle(output_img, left_extremities["left"], 3, (255, 0, 255), -1)  # Left (purple)
                cv2.circle(output_img, left_extremities["right"], 3, (0, 255, 255), -1)  # Right (yellow)
                
                # Right eye
                cv2.circle(output_img, right_extremities["top"], 3, (255, 0, 0), -1)  # Top (red)
                cv2.circle(output_img, right_extremities["bottom"], 3, (0, 0, 255), -1)  # Bottom (blue)
                cv2.circle(output_img, right_extremities["left"], 3, (255, 0, 255), -1)  # Left (purple)
                cv2.circle(output_img, right_extremities["right"], 3, (0, 255, 255), -1)  # Right (yellow)
                
                # Add color legend
                cv2.putText(output_img, "Eye points (green)", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(output_img, "Top points (red)", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(output_img, "Bottom points (blue)", (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Add eye status info
                left_status = "CLOSED" if left_closed else "OPEN"
                right_status = "CLOSED" if right_closed else "OPEN"
                
                cv2.putText(output_img, f"Left Eye: {left_ear:.2f} - {left_status}", 
                           (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.putText(output_img, f"Right Eye: {right_ear:.2f} - {right_status}", 
                           (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add height/width ratios
                if left_width > 0:
                    hw_ratio_left = left_height / left_width
                    cv2.putText(output_img, f"L H/W Ratio: {hw_ratio_left:.2f}", 
                               (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if right_width > 0:
                    hw_ratio_right = right_height / right_width
                    cv2.putText(output_img, f"R H/W Ratio: {hw_ratio_right:.2f}", 
                               (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Save the output image
        if visualize and output_img is not None:
            output_path = os.path.splitext(os.path.basename(image_path))[0] + "_eye_detection.jpg"
            cv2.imwrite(output_path, output_img)
            print(f"Visualization saved to {output_path}")
        
        return processed_results, output_img

def print_eye_results(results):
    """Pretty print eye detection results with coordinates"""
    if not results:
        print("No results to display")
        return
    
    print("\n===== EYE DETECTION RESULTS =====")
    for face in results:
        print(f"\nFace #{face['face_idx'] + 1}:")
        
        # Left eye
        left_eye = face['left_eye']
        print("\n  LEFT EYE:")
        print(f"    Status: {'CLOSED' if left_eye['is_closed'] else 'OPEN'}")
        print(f"    Eye Aspect Ratio (EAR): {left_eye['ear']:.4f}")
        print(f"    Height: {left_eye['height']:.2f} pixels")
        print(f"    Width: {left_eye['width']:.2f} pixels")
        print(f"    Height/Width Ratio: {left_eye['height']/left_eye['width']:.4f}" if left_eye['width'] > 0 else "    Height/Width Ratio: N/A")
        
        print("\n    Extremity Coordinates:")
        print("      Top point (red):", left_eye['extremities']['top'])
        print("      Bottom point (blue):", left_eye['extremities']['bottom'])
        print("      Left point (purple):", left_eye['extremities']['left'])
        print("      Right point (yellow):", left_eye['extremities']['right'])
        
        print("\n    All Landmark Points:")
        for i, point in enumerate(left_eye['landmarks']):
            print(f"      Point {i}: {tuple(point)}")
        
        # Right eye
        right_eye = face['right_eye']
        print("\n  RIGHT EYE:")
        print(f"    Status: {'CLOSED' if right_eye['is_closed'] else 'OPEN'}")
        print(f"    Eye Aspect Ratio (EAR): {right_eye['ear']:.4f}")
        print(f"    Height: {right_eye['height']:.2f} pixels")
        print(f"    Width: {right_eye['width']:.2f} pixels")
        print(f"    Height/Width Ratio: {right_eye['height']/right_eye['width']:.4f}" if right_eye['width'] > 0 else "    Height/Width Ratio: N/A")
        
        print("\n    Extremity Coordinates:")
        print("      Top point (red):", right_eye['extremities']['top'])
        print("      Bottom point (blue):", right_eye['extremities']['bottom'])
        print("      Left point (purple):", right_eye['extremities']['left'])
        print("      Right point (yellow):", right_eye['extremities']['right'])
        
        print("\n    All Landmark Points:")
        for i, point in enumerate(right_eye['landmarks']):
            print(f"      Point {i}: {tuple(point)}")
    
    print("\n=================================")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Eye Detection with Customizable Landmark Indices')
    parser.add_argument('--image', type=str, default='lady.jpg', help='Path to input image')
    parser.add_argument('--visualize-groups', action='store_true', help='Visualize different landmark groups')
    parser.add_argument('--left-eye', type=str, default=None, help='Comma-separated list of indices for left eye')
    parser.add_argument('--right-eye', type=str, default=None, help='Comma-separated list of indices for right eye')
    parser.add_argument('--threshold', type=float, default=0.2, help='EAR threshold for closed eyes')
    
    args = parser.parse_args()
    
    # Parse eye indices if provided
    left_eye_indices = None
    right_eye_indices = None
    
    if args.left_eye:
        left_eye_indices = [int(idx) for idx in args.left_eye.split(',')]
    
    if args.right_eye:
        right_eye_indices = [int(idx) for idx in args.right_eye.split(',')]
    
    # Initialize detector with custom indices if provided
    detector = EyeDetector(left_eye_indices=left_eye_indices, right_eye_indices=right_eye_indices)
    
    try:
        # Visualize landmark groups if requested
        if args.visualize_groups:
            result_image = detector.visualize_landmark_groups(args.image)
            cv2.imshow("Landmark Groups", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            # Run eye detection
            results, result_image = detector.detect_eyes(args.image, threshold=args.threshold)
            
            if results:
                # Print detailed results including coordinates
                print_eye_results(results)
                
                # Print simplified summary
                print("\nSUMMARY:")
                for i, status in enumerate(results):
                    print(f"Face #{i+1}:")
                    print(f"  Left eye: {status['left_eye']['ear']:.2f} - {'CLOSED' if status['left_eye']['is_closed'] else 'OPEN'}")
                    print(f"  Right eye: {status['right_eye']['ear']:.2f} - {'CLOSED' if status['right_eye']['is_closed'] else 'OPEN'}")
                    print(f"  Both eyes closed: {status['both_eyes_closed']}")
                    print()
                
                # Display the result
                cv2.imshow("Eye Status Detection", result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")