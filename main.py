import cv2
import argparse
import os
from face_detector import FaceDetector, FacialFeatureAnalyzer, print_combined_results
from eyelid_detector import EyelidDistanceDetector, print_eye_state_results
from mouth_detector import MouthDistanceDetector, print_mouth_state_results

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