import cv2
import argparse
import os
from face_detector import FaceDetector
from eyelid_detector import EyelidDistanceDetector, print_eye_state_results
from mouth_detector import MouthDistanceDetector, print_mouth_state_results

class FacialFeatureAnalyzer:
    """
    Combined analyzer for facial features including eyes and mouth
    """
    
    def __init__(self, providers=['CPUExecutionProvider']):
        """Initialize the facial feature analyzer"""
        self.eye_detector = EyelidDistanceDetector(providers)
        self.mouth_detector = MouthDistanceDetector(providers)
        print("Facial Feature Analyzer initialized")
    
    def analyze_image(self, image_path, eye_threshold=0.05, mouth_threshold=0.1, visualize=True):
        """
        Analyze facial features in an image
        
        Args:
            image_path: Path to the input image
            eye_threshold: Threshold for eye openness
            mouth_threshold: Threshold for mouth openness
            visualize: Whether to create visualizations
            
        Returns:
            Combined results dictionary
        """
        # Get eye analysis
        eye_results, eye_img = self.eye_detector.detect_eye_state(
            image_path, visualize=visualize, threshold_ratio=eye_threshold
        )
        
        # Get mouth analysis
        mouth_results, mouth_img = self.mouth_detector.detect_mouth_state(
            image_path, visualize=visualize, threshold_ratio=mouth_threshold
        )
        
        # If no faces were detected, return None
        if eye_results is None or mouth_results is None:
            print("No valid analysis results")
            return None, None
        
        # Combine results
        combined_results = []
        
        for i in range(min(len(eye_results), len(mouth_results))):
            eye_data = eye_results[i]
            mouth_data = mouth_results[i]
            
            # Make sure we're looking at the same face
            if eye_data['face_idx'] == mouth_data['face_idx']:
                result = {
                    'face_idx': eye_data['face_idx'],
                    'face_width': eye_data['face_width'],
                    'right_eye': eye_data['right_eye'],
                    'left_eye': eye_data['left_eye'],
                    'both_eyes_closed': eye_data['both_eyes_closed'],
                    'mouth': mouth_data['mouth']
                }
                combined_results.append(result)
        
        # Create combined visualization if requested
        if visualize and eye_img is not None and mouth_img is not None:
            # We'll use the eye_img as base and add mouth information
            combined_img = eye_img.copy()
            
            # Get face detection data from mouth analysis
            for i, result in enumerate(combined_results):
                # Add mouth information at an offset from eye information
                y_offset = 90
                
                # Mouth information
                mouth = result['mouth']
                
                # Get color based on state
                if mouth['is_closed']:
                    color = (0, 0, 255)  # Red for closed
                elif mouth.get('is_yawning', False):
                    color = (0, 165, 255)  # Orange for yawning
                else:
                    color = (0, 255, 0)  # Green for normal open
                
                # Get mouth state display text
                if 'state' in mouth:
                    state_text = mouth['state']
                else:
                    state_text = "CLOSED" if mouth['is_closed'] else "OPEN"
                
                cv2.putText(combined_img, f"Mouth: {mouth['ratio']:.3f} - {state_text}", 
                           (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw mouth landmarks and measurement line
                upper_lip = mouth['upper_point']
                lower_lip = mouth['lower_point']
                cv2.circle(combined_img, upper_lip, 3, (255, 0, 0), -1)  # Upper (red)
                cv2.circle(combined_img, lower_lip, 3, (0, 0, 255), -1)  # Lower (blue)
                cv2.line(combined_img, upper_lip, lower_lip, (255, 0, 255), 1)  # Line connecting points
            
            # Save the combined visualization
            output_path = os.path.splitext(os.path.basename(image_path))[0] + "_combined_analysis.jpg"
            cv2.imwrite(output_path, combined_img)
            print(f"Combined visualization saved to {output_path}")
            
            return combined_results, combined_img
        
        return combined_results, None


def print_combined_results(results):
    """Pretty print combined facial feature analysis results"""
    if not results:
        print("No results to display")
        return
    
    print("\n===== COMBINED FACIAL FEATURE ANALYSIS =====")
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
        
        # Left eye
        left_eye = face['left_eye']
        print("\n  LEFT EYE:")
        print(f"    Status: {'CLOSED' if left_eye['is_closed'] else 'OPEN'}")
        print(f"    Eyelid distance: {left_eye['distance']:.2f} pixels")
        print(f"    Distance/Face width ratio: {left_eye['ratio']:.4f}")
        
        # Overall eye status
        print(f"\n  Both eyes closed: {face['both_eyes_closed']}")
        
        # Mouth
        mouth = face['mouth']
        print("\n  MOUTH:")
        
        # Get mouth state display text
        if 'state' in mouth:
            mouth_state = mouth['state']
        else:
            mouth_state = "CLOSED" if mouth['is_closed'] else "OPEN"
            
        print(f"    Status: {mouth_state}")
        print(f"    Lip distance: {mouth['distance']:.2f} pixels")
        print(f"    Distance/Face width ratio: {mouth['ratio']:.4f}")
        
        # Check if yawning info is available
        if 'is_yawning' in mouth:
            print(f"    Is yawning: {mouth['is_yawning']}")
        
        # Expression inference
        expression = "neutral"
        
        if face['both_eyes_closed']:
            if mouth.get('is_yawning', False):
                expression = "eyes closed, mouth wide open (yawning)"
            elif mouth['is_closed']:
                expression = "eyes and mouth closed (possibly sleeping/resting)"
            else:
                expression = "eyes closed, mouth open (possibly speaking with eyes closed)"
        else:
            if mouth.get('is_yawning', False):
                expression = "eyes open, mouth wide open (yawning or surprised)"
            elif mouth['is_closed']:
                expression = "eyes open, mouth closed (neutral/attentive)"
            else:
                expression = "eyes open, mouth open (possibly speaking/expressing)"
        
        print(f"\n  Possible expression: {expression}")
    
    print("\n============================================")


# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facial Feature Analysis (Eyes and Mouth)')
    parser.add_argument('--image', type=str, default='lady.jpg', help='Path to input image')
    parser.add_argument('--eye-threshold', type=float, default=0.05, 
                   help='Ratio threshold for closed eyes (smaller ratio = closed)')
    parser.add_argument('--mouth-threshold', type=float, default=0.1, 
                   help='Ratio threshold for closed mouth (smaller ratio = closed)')
    parser.add_argument('--mode', type=str, choices=['eyes', 'mouth', 'combined'], default='combined',
                   help='Analysis mode: eyes, mouth, or combined')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'eyes':
            # Initialize eye detector
            detector = EyelidDistanceDetector()
            
            # Detect eye state
            results, result_image = detector.detect_eye_state(args.image, threshold_ratio=args.eye_threshold)
            
            if results:
                # Print detailed results
                print_eye_state_results(results)
                
                # Display the result
                cv2.imshow("Eye State Detection", result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        elif args.mode == 'mouth':
            # Initialize mouth detector
            detector = MouthDistanceDetector()
            
            # Detect mouth state
            results, result_image = detector.detect_mouth_state(args.image, threshold_ratio=args.mouth_threshold)
            
            if results:
                # Print detailed results
                print_mouth_state_results(results)
                
                # Display the result
                cv2.imshow("Mouth State Detection", result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        else:  # combined mode
            # Initialize combined analyzer
            analyzer = FacialFeatureAnalyzer()
            
            # Analyze facial features
            results, result_image = analyzer.analyze_image(
                args.image, 
                eye_threshold=args.eye_threshold,
                mouth_threshold=args.mouth_threshold
            )
            
            if results:
                # Print detailed results
                print_combined_results(results)
                
                # Display the result
                cv2.imshow("Facial Feature Analysis", result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
    except Exception as e:
        print(f"Error: {e}")