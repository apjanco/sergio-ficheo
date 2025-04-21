import typer
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
from pdf2image import convert_from_path
from datetime import datetime
import logging
from typing import Dict, Any, Optional, Tuple
import os
import json
import yaml
from utils.batch import BatchProcessor
from utils.processor import process_file
from rich.console import Console
from PIL import ExifTags

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()

# Load YOLO model
try:
    from ultralytics import YOLO
    yolo_model = YOLO("models/yolov8s-fichero.pt")  # Keep original model
    logger.info("Successfully loaded YOLO model")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    raise

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

def crop_with_yolo(image_path: Path, output_folder: Path, conf_threshold: float = 0.35) -> Optional[Tuple[Image.Image, Dict[str, Any]]]:
    """Crop image using YOLOv8 model
    Returns tuple of (cropped_image, crop_info) where crop_info contains box coordinates and confidence"""
    try:
        # Get true orientation and required rotation
        true_orientation, rotation_angle, orientation_details = get_image_orientation(image_path)
        
        # Read original image and convert to PIL
        original_pil = Image.open(image_path)
        orig_width, orig_height = original_pil.size
        
        # Apply rotation if needed
        if rotation_angle > 0:
            original_pil = original_pil.rotate(rotation_angle, expand=True)
            # Update dimensions after rotation
            orig_width, orig_height = original_pil.size
        
        # Convert to numpy array for YOLO
        original_img = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
        
        # Resize image for model prediction while maintaining aspect ratio and stride requirement
        model_size = 640
        scale = min(model_size / orig_width, model_size / orig_height)
        model_width = int(orig_width * scale)
        model_height = int(orig_height * scale)
        
        # Ensure dimensions are multiples of 32 (YOLO stride requirement)
        model_width = ((model_width + 31) // 32) * 32
        model_height = ((model_height + 31) // 32) * 32
        
        model_img = cv2.resize(original_img, (model_width, model_height))
        
        # Run prediction with optimized settings
        results = yolo_model.predict(
            source=model_img,
            conf=conf_threshold,
            imgsz=(model_width, model_height),
            iou=0.45,
            verbose=False
        )[0]
        
        if not results.boxes:
            logger.warning("No detections found")
            return None
            
        # Get the best detection (highest confidence)
        box = max(results.boxes.data, key=lambda x: x[4])
        x1, y1, x2, y2, conf = map(float, box[:5])
        
        # Scale coordinates back to original image size
        x1 = int(x1 / scale)
        y1 = int(y1 / scale)
        x2 = int(x2 / scale)
        y2 = int(y2 / scale)
        
        # Apply padding only on left and bottom
        padding = 30
        x1 = max(0, x1 - padding)  # Add padding to left
        y1 = max(0, y1 - padding)  # Add padding to top
        x2 = min(orig_width, x2)   # No padding on right
        y2 = min(orig_height, y2 + padding)  # Add padding to bottom
        
        # Crop original image at full resolution
        cropped = original_img[y1:y2, x1:x2]
        
        # Convert to PIL Image and preserve EXIF
        result = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        
        # Try to preserve EXIF data from original image
        try:
            if hasattr(original_pil, '_getexif'):
                exif = original_pil._getexif()
                if exif is not None:
                    result.info['exif'] = exif
        except Exception as e:
            logger.warning(f"Could not preserve EXIF data: {e}")
        
        # Create crop info dictionary
        crop_info = {
            "box": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            },
            "confidence": float(conf),
            "method": "yolo",
            "padding": padding,
            "original_size": [orig_width, orig_height],
            "cropped_size": [x2 - x1, y2 - y1]
        }
            
        return result, crop_info
    except Exception as e:
        logger.error(f"YOLO cropping failed: {e}")
        return None

def detect_with_contours(image_path: Path) -> Optional[Image.Image]:
    """Try to detect document using contour detection"""
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = 30
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + padding)
        h = min(img.shape[0] - y, h + padding)
        
        # Crop the image
        cropped = img[y:y+h, x:x+w]
        
        # Convert to PIL Image
        return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    except Exception as e:
        logger.warning(f"Contour detection failed: {e}")
        return None

def process_image(file_path: Path, out_path: Path) -> dict:
    """Process a single image file"""
    # Get source folder structure from input path
    source_dir = Path(file_path).parts[1:]  # Skip the first part (documents)
    
    # Verify file exists and is readable
    if not file_path.exists():
        logger.error(f"File does not exist: {file_path}")
        return {"success": False, "error": "File not found"}
    
    try:
        # Try to open the image to verify it's readable
        with Image.open(file_path) as img:
            logger.debug(f"Successfully opened image: {file_path.name} (format: {img.format})")
    except Exception as e:
        logger.error(f"Failed to open image {file_path.name}: {e}")
        return {"success": False, "error": f"Failed to open image: {e}"}
    
    attempts = []
    
    # Try YOLO with original confidence threshold
    logger.debug(f"Attempting YOLO detection with confidence 0.35 for {file_path.name}")
    result = crop_with_yolo(file_path, out_path.parent, conf_threshold=0.35)
    attempts.append({
        "method": "yolo",
        "confidence": 0.35,
        "success": bool(result)
    })
    
    # If YOLO fails, try with lower confidence
    if not result:
        logger.debug(f"Attempting YOLO detection with confidence 0.15 for {file_path.name}")
        result = crop_with_yolo(file_path, out_path.parent, conf_threshold=0.15)
        attempts.append({
            "method": "yolo",
            "confidence": 0.15,
            "success": bool(result)
        })
    
    # If YOLO still fails, try contour detection
    if not result:
        logger.debug(f"Attempting contour detection for {file_path.name}")
        result = detect_with_contours(file_path)
        attempts.append({
            "method": "contour",
            "success": bool(result)
        })
        if result:
            # For contour detection, create a simplified crop info
            crop_info = {
                "method": "contour",
                "original_size": list(Image.open(file_path).size),
                "cropped_size": list(result.size)
            }
            result = (result, crop_info)
    
    # If all detection methods fail, use original image
    if not result:
        logger.warning(f"Using original image as fallback for {file_path.name}")
        original = Image.open(file_path)
        crop_info = {
            "method": "original",
            "original_size": list(original.size),
            "cropped_size": list(original.size)
        }
        result = (original, crop_info)
        attempts.append({
            "method": "original",
            "success": True
        })
    
    # Convert to JPG if needed
    image, crop_info = result
    if image.format != 'JPEG':
        logger.debug(f"Converting {file_path.name} from {image.format} to JPEG")
        image = image.convert('RGB')
    
    # Save the result as JPG with lowercase extension
    out_path = out_path.with_suffix('.jpg')
    image.save(out_path, 'JPEG', quality=95)
    logger.debug(f"Saved cropped image to {out_path}")
    
    # Build output path preserving full source hierarchy
    # Use the same directory structure but with lowercase .jpg extension
    rel_path = Path(*source_dir[:-1]) / out_path.with_suffix('.jpg').name
    
    # Add attempts to the crop info
    crop_info["attempts"] = attempts
    
    return {
        "outputs": [str(rel_path)],
        "details": crop_info  # Include the crop info in the details
    }

def process_pdf(file_path: Path, out_path: Path) -> dict:
    """Process a PDF file"""
    # Get source folder structure from input path
    source_dir = Path(file_path).parts[-4:-1]
    
    # Convert and process each page
    images = convert_from_path(file_path, dpi=300)
    outputs = []
    details = {}
    
    # Create directory for PDF pages
    pdf_dir = out_path.parent / f"{out_path.stem}"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    for i, image in enumerate(images):
        logger.info(f"Processing page {i+1} of {len(images)} from {file_path.name}")
        # Save original page as JPG
        page_path = pdf_dir / f"page_{i + 1}.jpg"
        image.save(page_path, "JPEG", quality=95)
        
        # Process with YOLO
        result = crop_with_yolo(page_path, pdf_dir, conf_threshold=0.35)
        
        if result:
            # Save cropped page as JPG
            cropped_path = pdf_dir / f"page_{i + 1}_cropped.jpg"
            result[0].save(cropped_path, "JPEG", quality=95)
            
            # Build relative path preserving hierarchy
            rel_path = Path(*source_dir) / f"{out_path.stem}" / f"page_{i + 1}_cropped.jpg"
            outputs.append(str(rel_path))
            
            details[f"page_{i + 1}"] = result[1]  # Include the crop info
        else:
            details[f"page_{i + 1}"] = {
                "success": False,
                "error": "No detection",
                "original_size": list(image.size),
                "attempts": [{
                    "method": "yolo",
                    "confidence": 0.35,
                    "success": False
                }]
            }
    
    return {
        "outputs": outputs,
        "details": details
    }

def process_document(file_path: str, output_folder: Path) -> dict:
    """Process a single document file"""
    file_path = Path(file_path)
    
    def process_fn(f: str, o: Path) -> dict:
        return process_image(Path(f), o)
    
    return process_file(
        file_path=str(file_path),
        output_folder=output_folder,
        process_fn=process_fn,
        file_types={
            '.pdf': lambda f, o: process_pdf(Path(f), o),
            '.jpg': process_fn,
            '.jpeg': process_fn,
            '.tif': process_fn,
            '.tiff': process_fn,
            '.png': process_fn
        }
    )

def crop(
    source_folder: Path = typer.Argument(..., help="Source folder containing documents"),
    source_manifest: Path = typer.Argument(..., help="Manifest file"),
    output_folder: Path = typer.Argument(..., help="Output folder for cropped images")
):
    """Crop images from documents using YOLO detection"""
    processor = BatchProcessor(
        input_manifest=source_manifest,
        output_folder=output_folder,
        process_name="crop",
        base_folder=source_folder,  # Paths in manifest already include documents/
        processor_fn=lambda f, o: process_document(f, o)
    )
    processor.process()

if __name__ == "__main__":
    typer.run(crop)