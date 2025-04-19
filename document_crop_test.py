import sys
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ExifTags
from deskew import determine_skew
from doctr.io import DocumentFile
from doctr.models import detection_predictor
from PIL import ImageDraw

# Output folder for saving debug images
OUTPUT_DIR = Path("doctr_crop_results")
OUTPUT_DIR.mkdir(exist_ok=True)

def save_image(img, name):
    """Save an image to the output folder"""
    if isinstance(img, Image.Image):
        img.save(OUTPUT_DIR / f"{name}.jpg")
    else:
        img = Image.fromarray(img)
        img.save(OUTPUT_DIR / f"{name}.jpg")
    print(f"Saved: {name}.jpg")

def get_image_orientation(image_path):
    """Get the true orientation of an image using EXIF data"""
    try:
        image = Image.open(image_path)
        # Check for EXIF orientation tag
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                try:
                    exif = dict(image._getexif().items())
                    if orientation in exif:
                        print(f"[Image] Found EXIF orientation: {exif[orientation]}")
                        if exif[orientation] > 4:  # Values 5-8 indicate vertical orientation
                            return "vertical"
                except (AttributeError, KeyError, IndexError):
                    pass
        
        # Fallback to dimension check if no EXIF data
        width, height = image.size
        return "vertical" if height > width else "horizontal"
    except Exception as e:
        print(f"[Warning] Error checking orientation: {e}")
        return "unknown"

def enhanced_contour_crop(image, debug_output=False):
    """Enhanced contour-based document detection with improved cropping"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    # Get dimensions
    height, width = img_gray.shape
    
    # Apply bilateral filter to reduce noise while preserving edges
    blurred = cv2.bilateralFilter(img_gray, 9, 75, 75)
    
    # Apply adaptive thresholding first (more reliable for documents)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Dilate to connect components
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    if debug_output:
        debug_dir = Path("debug_crops")
        debug_dir.mkdir(exist_ok=True)
        Image.fromarray(thresh).save(debug_dir / "threshold.jpg")
        Image.fromarray(dilated).save(debug_dir / "dilated.jpg")
    
    # Find contours from thresholded image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, try Canny edge detection
    if not contours:
        edges = cv2.Canny(blurred, 30, 100)  # More sensitive thresholds
        dilated_edges = cv2.dilate(edges, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if debug_output:
            Image.fromarray(edges).save(debug_dir / "edges.jpg")
            Image.fromarray(dilated_edges).save(debug_dir / "dilated_edges.jpg")
    
    # Sort by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if debug_output and contours:
        debug_img = img_array.copy()
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 3)
        Image.fromarray(debug_img).save(debug_dir / "contours.jpg")
    
    # Process large contours
    for contour in contours[:3]:
        area = cv2.contourArea(contour)
        img_area = width * height
        
        # Lower threshold to 10% of image area
        if area < (img_area * 0.1):
            continue
        
        # Try to get a more precise polygon approximation
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If we have a quadrilateral, use it directly
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
        else:
            # For non-quadrilateral, try to find the largest rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            x, y, w, h = cv2.boundingRect(box)
        
        # Skip extreme aspect ratios (more lenient now)
        aspect_ratio = max(w/h, h/w)
        if aspect_ratio > 8:  # More lenient aspect ratio
            continue
        
        # Add larger margin (3%)
        margin_x = int(w * 0.03)
        margin_y = int(h * 0.03)
        
        # Apply margin with bounds checking
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(width - x, w + 2 * margin_x)
        h = min(height - y, h + 2 * margin_y)
        
        # Crop image
        cropped = img_array[y:y+h, x:x+w]
        
        if debug_output:
            crop_debug = img_array.copy()
            cv2.rectangle(crop_debug, (x, y), (x+w, y+h), (0, 0, 255), 3)
            Image.fromarray(crop_debug).save(debug_dir / "final_crop.jpg")
        
        return Image.fromarray(cropped)
    
    # If still no suitable contours found, try OTSU thresholding
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dilated_otsu = cv2.dilate(thresh_otsu, kernel, iterations=3)
    contours_otsu, _ = cv2.findContours(dilated_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug_output:
        Image.fromarray(thresh_otsu).save(debug_dir / "otsu.jpg")
    
    # Sort by area
    contours_otsu = sorted(contours_otsu, key=cv2.contourArea, reverse=True)
    
    # Try to find a suitable contour
    for contour in contours_otsu[:2]:
        x, y, w, h = cv2.boundingRect(contour)
        # Lower threshold to 15% of image
        if (w*h) > (width*height*0.15):
            # Add margin
            margin_x = int(w * 0.03)
            margin_y = int(h * 0.03)
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            w = min(width - x, w + 2 * margin_x)
            h = min(height - y, h + 2 * margin_y)
            # Crop image
            cropped = img_array[y:y+h, x:x+w]
            return Image.fromarray(cropped)
    
    print("[Warning] Could not detect document boundaries")
    return image

def doctr_crop(image_path):
    """Crop document using DocTR and enhanced contour detection"""
    # Get true orientation first
    true_orientation = get_image_orientation(image_path)
    print(f"[Image] True orientation: {true_orientation}")
    
    # Load and preprocess image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    print(f"[Image] Dimensions: {image.size[0]}x{image.size[1]}")
    
    # Save original for debugging
    image.save("1_original.jpg")
    
    # Convert to numpy array for DocTR
    img_array = np.array(image)
    
    try:
        # Try DocTR detection first with more aggressive settings
        model = detection_predictor(pretrained=True)
        # Normalize the image to improve text detection
        img_norm = img_array.astype(np.float32) / 255.0
        result = model([img_norm])
        
        # Get detected regions from DocTR output
        regions = []
        if result and len(result) > 0:
            # Extract word boxes from first page
            for pred in result[0]:
                if isinstance(pred, dict) and 'geometry' in pred:
                    regions.append(pred['geometry'])
        
        print(f"[DocTR] Found {len(regions)} word regions")
        
        if regions:
            # Draw boxes for debugging
            debug_img = image.copy()
            draw = ImageDraw.Draw(debug_img)
            for box in regions:
                points = box_to_points(box, image.size)
                draw.rectangle(points, outline='red', width=2)
            debug_img.save("2_debug_box.jpg")
            
            # Get bounding box of all regions
            boxes = np.array([box_to_points(r, image.size) for r in regions])
            min_x = np.min(boxes[:, [0, 2]])
            min_y = np.min(boxes[:, [1, 3]])
            max_x = np.max(boxes[:, [0, 2]])
            max_y = np.max(boxes[:, [1, 3]])
            
            # Add larger margins (7%)
            margin_x = int((max_x - min_x) * 0.07)
            margin_y = int((max_y - min_y) * 0.07)
            
            min_x = max(0, min_x - margin_x)
            min_y = max(0, min_y - margin_y)
            max_x = min(image.size[0], max_x + margin_x)
            max_y = min(image.size[1], max_y + margin_y)
            
            # Crop the image
            cropped = image.crop((min_x, min_y, max_x, max_y))
        else:
            # Fallback to enhanced contour detection
            print("[DocTR] No text regions found, falling back to contour detection")
            cropped = enhanced_contour_crop(image, debug_output=True)
        
        # Check if we need to rotate the crop
        print(f"[Image] Cropped dimensions: {cropped.size[0]}x{cropped.size[1]}")
        crop_orientation = "vertical" if cropped.size[1] > cropped.size[0] else "horizontal"
        
        if true_orientation == "vertical" and crop_orientation == "horizontal":
            print("[Image] Rotating crop to match original orientation")
            cropped = cropped.rotate(90, expand=True)
            print(f"[Image] Final dimensions: {cropped.size[0]}x{cropped.size[1]}")
        
        cropped.save("3_cropped.jpg")
        return cropped
        
    except Exception as e:
        print(f"[DocTR] Error: {e}, falling back to contour detection")
        return enhanced_contour_crop(image, debug_output=True)

def box_to_points(box, size):
    """Convert relative coordinates to absolute pixels"""
    return (
        int(box[0] * size[0]),  # x0
        int(box[1] * size[1]),  # y0
        int(box[2] * size[0]),  # x1
        int(box[3] * size[1])   # y1
    )

def main():
    if len(sys.argv) != 2:
        print("Usage: python doctr_cropper.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: File not found - {image_path}")
        sys.exit(1)

    print(f"[Running] Processing {image_path}")
    result = doctr_crop(image_path)
    result.show()

if __name__ == "__main__":
    main()