import typer
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
from pdf2image import convert_from_path
from datetime import datetime
from utils.batch import BatchProcessor
from utils.processor import process_file
import srsly
from PIL import ExifTags
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YOLOv8 model
model = YOLO('yolov8x.pt')  # Using x model for better accuracy

# Image processing functions specific to cropping
def is_likely_ruler(w: int, h: int, max_ruler_aspect_ratio: float = 15, max_ruler_width_ratio: float = 0.05) -> bool:
    """Check if the contour is likely a ruler based on its aspect ratio and width."""
    aspect_ratio = max(w / h, h / w)
    return aspect_ratio > max_ruler_aspect_ratio or w < max_ruler_width_ratio * h

def is_predominantly_black(image: Image.Image, threshold: float = 0.70) -> bool:
    """Check if the image is predominantly black."""
    img_array = np.array(image.convert("L"))
    black_pixels = np.sum(img_array < 100)  # Increased threshold from 50 to 100
    total_pixels = img_array.size
    return (black_pixels / total_pixels) > threshold

def get_image_orientation(image_path: Path) -> tuple[str, int, dict]:
    """Get the true orientation of an image using EXIF data and required rotation angle.
    Returns (orientation, rotation_angle, details) where:
    - orientation is "vertical" or "horizontal"
    - rotation_angle is the degrees needed to correct the image
    - details is a dict with EXIF and processing information"""
    details = {
        "exif_orientation": None,
        "original_dimensions": None,
        "rotation_applied": None,
        "reason": None
    }
    
    try:
        image = Image.open(image_path)
        width, height = image.size
        details["original_dimensions"] = {"width": width, "height": height}
        
        # Check for EXIF orientation tag
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                try:
                    exif = dict(image._getexif().items())
                    if orientation in exif:
                        exif_orientation = exif[orientation]
                        details["exif_orientation"] = exif_orientation
                        
                        # EXIF orientation values and their meanings:
                        # 1: Normal (0°)
                        # 2: Mirrored (0°)
                        # 3: Upside down (180°)
                        # 4: Mirrored upside down (180°)
                        # 5: Mirrored and rotated 90° CCW (90°)
                        # 6: Rotated 90° CW (270°)
                        # 7: Mirrored and rotated 90° CW (270°)
                        # 8: Rotated 90° CCW (90°)
                        
                        if exif_orientation in [5, 6, 7, 8]:  # Vertical orientations
                            # Calculate required rotation angle
                            if exif_orientation == 6:  # 90° CW
                                details["rotation_applied"] = 270
                                details["reason"] = "EXIF orientation 6 (90° CW) requires 270° rotation to correct"
                                return "vertical", 270, details
                            elif exif_orientation == 8:  # 90° CCW
                                details["rotation_applied"] = 90
                                details["reason"] = "EXIF orientation 8 (90° CCW) requires 90° rotation to correct"
                                return "vertical", 90, details
                            elif exif_orientation == 5:  # Mirrored and rotated 90° CCW
                                details["rotation_applied"] = 270
                                details["reason"] = "EXIF orientation 5 (Mirrored 90° CCW) requires 270° rotation to correct"
                                return "vertical", 270, details
                            elif exif_orientation == 7:  # Mirrored and rotated 90° CW
                                details["rotation_applied"] = 90
                                details["reason"] = "EXIF orientation 7 (Mirrored 90° CW) requires 90° rotation to correct"
                                return "vertical", 90, details
                except (AttributeError, KeyError, IndexError) as e:
                    details["reason"] = f"No valid EXIF data found: {str(e)}"
        
        # Fallback to dimension check if no EXIF data
        if height > width:
            details["reason"] = "No EXIF data, using dimensions (height > width) to determine vertical orientation"
            return "vertical", 0, details
        else:
            details["reason"] = "No EXIF data, using dimensions (width >= height) to determine horizontal orientation"
            return "horizontal", 0, details
    except Exception as e:
        details["reason"] = f"Error checking orientation: {str(e)}"
        return "unknown", 0, details

def contour_crop(image: Image.Image, image_path: Path = None) -> tuple[Image.Image, dict]:
    """Crop image using YOLOv8 document detection."""
    details = {
        "pre_processing": {},
        "detection": {},
        "final_crop": None
    }
    
    # Get true orientation if image path is provided
    true_orientation = None
    rotation_angle = 0
    if image_path:
        true_orientation, rotation_angle, orientation_details = get_image_orientation(image_path)
        details["orientation"] = orientation_details
        logger.info(f"[Image] True orientation: {true_orientation}, Rotation needed: {rotation_angle}°")
        
        # Apply initial rotation if needed
        if rotation_angle > 0:
            image = image.rotate(rotation_angle, expand=True)
            details["rotation_applied"] = rotation_angle
    
    # Convert image to numpy array for YOLO
    img_array = np.array(image)
    
    # Get dimensions
    height, width = img_array.shape[:2]
    details["pre_processing"]["original_dimensions"] = {"width": width, "height": height}
    
    # Run YOLO detection with very low confidence threshold to catch all potential boxes
    results = model(img_array, conf=0.001)
    
    # Process detections
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        best_box = None
        max_score = 0
        
        logger.info(f"Found {len(boxes)} potential document detections")
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            box_width = x2 - x1
            box_height = y2 - y1
            area = box_width * box_height
            conf = box.conf[0].cpu().numpy()
            area_ratio = area / (width * height)
            
            # Skip tiny or huge detections
            if area_ratio < 0.1 or area_ratio > 0.98:
                continue
                
            # Calculate center position of the box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Calculate position scores (prefer boxes closer to center)
            # More lenient on horizontal position to handle rulers
            center_score_x = 1 - abs(center_x - width/2) / (width/2)
            center_score_y = 1 - abs(center_y - height/2) / (height/2)
            position_score = (center_score_x * 0.3 + center_score_y * 0.7)  # Weight vertical position more
            
            # Calculate aspect ratio score (prefer document-like aspect ratios)
            aspect_ratio = box_width / box_height
            # Prefer aspect ratios close to common document formats (A4, letter, etc.)
            aspect_score = 1 - min(abs(aspect_ratio - 0.707), abs(aspect_ratio - 1.414))
            
            # Calculate box fill ratio (how much of the bounding box is filled)
            # This helps identify solid rectangular shapes vs sparse detections
            fill_ratio = area / (box_width * box_height)
            fill_score = fill_ratio
            
            # Calculate size preference score
            # Prefer larger boxes that don't touch the edges
            edge_margin = 10  # pixels
            touches_edge = (x1 <= edge_margin or y1 <= edge_margin or 
                          x2 >= width - edge_margin or y2 >= height - edge_margin)
            size_score = area_ratio * (0.8 if touches_edge else 1.0)
            
            # Calculate final score with adjusted weights
            score = (
                position_score * 0.35 +  # Heavily weight position
                aspect_score * 0.25 +    # Document-like shape is important
                size_score * 0.25 +      # Prefer larger detections
                fill_score * 0.15        # Solid shapes preferred
            )
            
            logger.info(f"Detection: area={area:.2f}, conf={conf:.2f}, area_ratio={area_ratio:.2f}, "
                       f"position_score={position_score:.2f}, aspect_score={aspect_score:.2f}, "
                       f"size_score={size_score:.2f}, fill_score={fill_score:.2f}, final_score={score:.2f}")
            
            if score > max_score:
                max_score = score
                best_box = box
        
        if best_box is not None:
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
            conf = best_box.conf[0].cpu().numpy()
            
            logger.info(f"Selected best detection: score={max_score:.2f}, conf={conf:.2f}")
            
            # No padding to keep tight crops
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(width, int(x2))
            y2 = min(height, int(y2))
            
            # Crop image
            cropped = img_array[y1:y2, x1:x2]
            cropped_image = Image.fromarray(cropped)
            
            # Check if we need to rotate the crop
            if true_orientation:
                crop_orientation = "vertical" if cropped_image.size[1] > cropped_image.size[0] else "horizontal"
                if true_orientation == "vertical" and crop_orientation == "horizontal":
                    logger.info("[Image] Rotating crop to match original orientation")
                    cropped_image = cropped_image.rotate(90, expand=True)
                    details["final_rotation"] = 90
            
            details["detection"] = {
                "confidence": float(conf),
                "coordinates": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                },
                "dimensions": {
                    "width": cropped_image.size[0],
                    "height": cropped_image.size[1]
                },
                "area_ratio": (x2 - x1) * (y2 - y1) / (width * height),
                "score_components": {
                    "position_score": float(position_score),
                    "aspect_score": float(aspect_score),
                    "size_score": float(size_score),
                    "fill_score": float(fill_score)
                }
            }
            
            details["final_crop"] = {
                "coordinates": {
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1)
                },
                "dimensions": {
                    "width": cropped_image.size[0],
                    "height": cropped_image.size[1]
                },
                "area_ratio": (x2 - x1) * (y2 - y1) / (width * height)
            }
            
            return cropped_image, details
    else:
        logger.warning("No document detections found")
    
    # If no detection found, return original image
    details["final_crop"] = {
        "status": "no_detection",
        "reason": "No document detected by YOLO"
    }
    return image, details

def process_image(file_path: Path, out_path: Path) -> dict:
    """Process a single image file"""
    details = {}
    
    # Convert any image format to RGB
    img = Image.open(file_path)
    if img.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', img.size, 'white')
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Crop and save as JPG
    cropped, crop_details = contour_crop(img, file_path)
    details.update(crop_details)
    
    # Print detailed JSON output
    print(srsly.json_dumps(details, indent=2))
    
    # Ensure output path has .jpg extension
    out_path = out_path.with_suffix('.jpg')
    cropped.save(out_path, "JPEG", quality=100)
    
    details["original_size"] = img.size
    details["cropped_size"] = cropped.size
    return {"details": details}

def process_pdf(file_path: Path, out_path: Path) -> dict:
    """Process a PDF file"""
    outputs = []
    details = {}
    
    # Convert and process each page
    images = convert_from_path(file_path, dpi=300)
    
    # Create directory for PDF pages
    pdf_dir = out_path.parent / f"{out_path.stem}"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the relative path from the base directory
    rel_base = out_path.relative_to(out_path.parents[1])
    
    # Get relative source path
    source_path = Path(file_path).relative_to(Path(file_path).parents[2])  # Remove /documents from path
    
    for i, image in enumerate(images):
        page_name = f"{out_path.stem}_page_{i + 1}.jpg"
        page_path = pdf_dir / page_name
        cropped, crop_details = contour_crop(image)
        cropped.save(page_path, "JPEG", quality=100)
        
        # Create relative path that matches actual file structure
        rel_path = str(rel_base.parent / rel_base.stem / page_name)
        outputs.append(rel_path)
        
        details[f"page_{i + 1}"] = {
            "original_size": image.size,
            "cropped_size": cropped.size,
            "page_number": i + 1,
            "total_pages": len(images)
        }
        
        # Print manifest entry for this page (will be captured by BatchProcessor)
        print(srsly.json_dumps({
            "source": str(source_path),
            "outputs": [rel_path],
            "processed_at": datetime.now().isoformat(),
            "success": True,
            "details": details[f"page_{i + 1}"]
        }))
    
    # Return None since we handled the output directly
    return None

def process_document(file_path: str, output_folder: Path, method: str = 'enhanced',
                    background: str = 'messy', edges: str = 'complex',
                    deskew: bool = True, debug: bool = False) -> dict:
    """Process a single document file"""
    file_path = Path(file_path)
    
    supported_types = {
        '.pdf': process_pdf,
        '.jpg': process_image,
        '.jpeg': process_image,
        '.tif': process_image,
        '.tiff': process_image,
        '.png': process_image
    }
    
    # All processing through process_file
    return process_file(
        file_path=str(file_path),
        output_folder=output_folder,
        process_fn=lambda f, o: process_pdf(f, o) if file_path.suffix.lower() == '.pdf' else process_image(f, o),
        file_types=supported_types
    )

def crop(
    documents_folder: Path = typer.Argument(..., help="Input documents folder"),
    documents_manifest: Path = typer.Argument(..., help="Input documents manifest file"),
    crops_folder: Path = typer.Argument(..., help="Output folder for cropped images"),
    method: str = typer.Option('enhanced', help="Cropping method: basic, contour, enhanced, or doctr"),
    background: str = typer.Option('white', help="Background type: black, white, messy, or colored"),
    edges: str = typer.Option('straight', help="Edge type: straight or complex"),
    deskew: str = typer.Option('false', help="Apply deskew after cropping"),
    debug: str = typer.Option('false', help="Save debug images")
):
    """Crop images from documents"""
    # Convert string boolean parameters to actual booleans
    deskew_bool = deskew.lower() == 'true'
    debug_bool = debug.lower() == 'true'
    
    processor = BatchProcessor(
        input_manifest=documents_manifest,
        output_folder=crops_folder,
        process_name="crop",
        processor_fn=lambda f, o: process_document(f, o),
        base_folder=documents_folder,
        use_source=True
    )
    processor.process()

if __name__ == "__main__":
    typer.run(crop)