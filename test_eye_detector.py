"""
Test script for eye detection.
"""
import os
import cv2
import sys
from eye_detection import EyeDetector, print_eye_results

def test_eye_detection(image_path):
    """Test eye detection on a specific image"""
    print(f"Testing eye detection on: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    try:
        # Initialize the detector
        detector = EyeDetector()
        
        # Run eye detection using the correct method name
        results, result_image = detector.detect_eyes(image_path)
        
        if results:
            # Print detailed results with all coordinates
            print_eye_results(results)
            
            # Display the image
            cv2.imshow("Eye Detection Results", result_image)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No faces detected in the image")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Use command line argument if provided, otherwise use default image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default image path - replace with your image
        image_path = "lady.jpg"
    
    test_eye_detection(image_path)