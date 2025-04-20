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
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load YOLOv8 model
model = YOLO('yolov8x.pt')  # Using x model for better accuracy

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

def yolo_crop(image_path: str, output_path: str = None) -> Dict[str, Any]:
    # Read the image with color profile preservation
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert BGR to RGB for PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Preserve color profile if it exists
    if 'icc_profile' in pil_img.info:
        icc_profile = pil_img.info['icc_profile']
    else:
        icc_profile = None
    
    height, width = img.shape[:2]
    
    # Create a copy for debug visualization only if we'll need it
    debug_img = None
    
    # Calculate edge strip dimensions (3% of image dimensions)
    edge_strip_width = int(width * 0.03)
    edge_strip_height = int(height * 0.03)
    
    # Run YOLO detection with lower confidence threshold
    results = model(img, conf=0.1)[0]  # Lowered from 0.15
    
    # Convert detections to our format and filter
    detections = []
    for det in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Calculate area ratio
        area_ratio = ((x2 - x1) * (y2 - y1)) / (width * height)
        
        # More lenient filtering criteria
        # Minimum confidence threshold
        if score < 0.15:  # Lowered from 0.2
            logging.debug(f"Skipping detection due to low confidence: {score:.2f}")
            continue
            
        # Area ratio checks - more lenient
        if area_ratio < 0.02:  # Lowered from 0.03
            logging.debug(f"Skipping detection due to small area ratio: {area_ratio:.2f}")
            continue
        if area_ratio > 0.98:  # Increased from 0.97
            logging.debug(f"Skipping detection due to large area ratio: {area_ratio:.2f}")
            continue
            
        # More lenient ruler detection
        aspect_ratio = max((x2 - x1) / (y2 - y1), (y2 - y1) / (x2 - x1))
        if aspect_ratio > 20.0:  # Increased from 15.0
            logging.debug(f"Skipping detection due to extreme aspect ratio: {aspect_ratio:.2f}")
            continue
            
        # More lenient edge detection for rulers
        edge_margin = max(edge_strip_width, edge_strip_height)
        is_at_edge = (x1 < edge_margin or 
                     y1 < edge_margin or 
                     x2 > (width - edge_margin) or 
                     y2 > (height - edge_margin))
        
        if is_at_edge and (aspect_ratio > 10.0):  # More lenient filtering for edge objects
            logging.debug(f"Skipping edge detection with high aspect ratio: edge={is_at_edge}, aspect_ratio={aspect_ratio:.2f}")
            continue
            
        detections.append({
            'box': [x1, y1, x2, y2],
            'score': score,
            'area_ratio': area_ratio
        })
    
    # Process detections
    if len(detections) > 0:
        best_detection = max(detections, key=lambda x: x['score'])
        x1, y1, x2, y2 = best_detection['box']
        conf = best_detection['score']
        
        logger.info(f"Selected best detection: score={conf:.2f}")
        
        # Add small padding to ensure we don't cut off edges
        padding = int(min(width, height) * 0.01)  # 1% padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        # Crop image
        cropped = img[y1:y2, x1:x2]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped_image = Image.fromarray(cropped_rgb)
        
        # Restore color profile if it exists
        if icc_profile:
            cropped_image.info['icc_profile'] = icc_profile
        
        # Check if we need to rotate the crop
        true_orientation, rotation_angle, _ = get_image_orientation(Path(image_path))
        if true_orientation:
            crop_orientation = "vertical" if cropped_image.size[1] > cropped_image.size[0] else "horizontal"
            if true_orientation == "vertical" and crop_orientation == "horizontal":
                logger.info("[Image] Rotating crop to match original orientation")
                cropped_image = cropped_image.rotate(90, expand=True)
        
        details = {
            'success': True,
            'box': [x1, y1, x2, y2],
            'confidence': conf,
            'orientation': true_orientation,
            'rotation_angle': rotation_angle,
            'num_detections': len(detections)
        }
        
        # Save the cropped image if output path is provided
        if output_path:
            # Convert output path to .jpg if it's not already
            output_path = str(Path(output_path).with_suffix('.jpg'))
            cropped_image.save(output_path, 'JPEG', quality=100, icc_profile=icc_profile)
            
        return details
    else:
        # If no valid detections found, save the original image and debug visualization
        logger.warning("No valid detections found, saving original image")
        
        # Create debug visualization
        debug_img = img.copy()
        for det in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(debug_img, f"{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        if output_path:
            # Save original image
            output_path = str(Path(output_path).with_suffix('.jpg'))
            pil_img.save(output_path, 'JPEG', quality=100, icc_profile=icc_profile)
            
            # Save debug image in debug folder
            debug_path = Path(output_path).parent.parent / "debug" / "crop" / f"{Path(output_path).stem}_debug.jpg"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_path), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
            
        return {
            'success': False,
            'confidence': 0.0,
            'num_detections': 0
        }

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
    
    # Crop using YOLO detection
    crop_details = yolo_crop(str(file_path), str(out_path))
    details.update(crop_details)
    
    # Print detailed JSON output
    print(srsly.json_dumps(details, indent=2))
    
    # Add original size to details
    details["original_size"] = img.size
    
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
        
        # Save the page temporarily
        image.save(page_path, "JPEG", quality=100)
        
        # Process with YOLO
        crop_details = yolo_crop(str(page_path), str(page_path))
        
        # Create relative path that matches actual file structure
        rel_path = str(rel_base.parent / rel_base.stem / page_name)
        outputs.append(rel_path)
        
        details[f"page_{i + 1}"] = {
            "original_size": image.size,
            "page_number": i + 1,
            "total_pages": len(images),
            **crop_details
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