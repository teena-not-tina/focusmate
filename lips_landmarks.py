import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import argparse

def visualize_landmarks(image_path, output_path=None, show_all=True, focus_mouth=True):
    """
    Visualize facial landmarks with their index numbers.
    
    Args:
        image_path: Path to input image
        output_path: Path to save the output image (default: based on input filename)
        show_all: Whether to show all 106 landmarks
        focus_mouth: Whether to highlight mouth region landmarks
    
    Returns:
        Annotated image with landmark indices
    """
    # Initialize the face analyzer
    app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'], 
                       providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Detect faces
    faces = app.get(img)
    
    if len(faces) == 0:
        print("No faces detected")
        return None
    
    # Create a copy for visualization
    vis_img = img.copy()
    
    # Define mouth region landmark indices
    mouth_indices = list(range(52, 77))  # Landmarks 52-76 are around the mouth region
    
    # Process each detected face
    for face in faces:
        landmarks = face.landmark_2d_106
        landmarks = np.round(landmarks).astype(int)
        
        # Draw face bounding box
        bbox = face.bbox.astype(int)
        cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        
        # Draw all landmarks and their indices
        for i, (x, y) in enumerate(landmarks):
            # Determine color and size based on region
            if i in mouth_indices and focus_mouth:
                # Mouth region - highlight with larger circles
                color = (0, 165, 255)  # Orange
                size = 2
                thickness = -1  # Filled circle
                font_scale = 0.4
                font_thickness = 1
            else:
                # Other landmarks
                color = (0, 255, 0)  # Green
                size = 1
                thickness = -1
                font_scale = 0.3
                font_thickness = 1
            
            # Skip non-mouth landmarks if not showing all
            if not show_all and i not in mouth_indices:
                continue
                
            # Draw the landmark point
            cv2.circle(vis_img, (x, y), size, color, thickness)
            
            # Add the landmark index number
            text = str(i)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            
            # Position the text to not overlap with the point
            text_x = x - text_size[0] // 2
            
            # Alternate text position above/below point to avoid overlap
            if i % 2 == 0:
                text_y = y - 5
            else:
                text_y = y + 15
                
            cv2.putText(vis_img, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
    
    # Save the visualization
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_landmarks.jpg"
    
    cv2.imwrite(output_path, vis_img)
    print(f"Visualization with landmark indices saved to {output_path}")
    
    return vis_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize facial landmarks with index numbers')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default=None, help='Path to save output image')
    parser.add_argument('--all', action='store_true', help='Show all 106 landmarks')
    parser.add_argument('--mouth-only', action='store_false', dest='all', help='Focus on mouth region only')
    
    args = parser.parse_args()
    
    try:
        vis_img = visualize_landmarks(args.image, args.output, show_all=args.all)
        
        # Display the result
        if vis_img is not None:
            cv2.imshow("Facial Landmarks", vis_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Error: {e}")