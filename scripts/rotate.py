import typer
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
from utils.batch import BatchProcessor
from utils.processor import process_file
from rich.console import Console

console = Console()

def hough_line_rotate(image: Image.Image, blur_kernel=(5, 5), canny_threshold1=50, canny_threshold2=150) -> tuple[Image.Image, dict]:
    """
    Rotate image based on Hough Line Transform.
    Returns (rotated_image, debug_info)
    """
    img_array = np.array(image)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray, blur_kernel, 0)
    edges = cv2.Canny(img_blurred, canny_threshold1, canny_threshold2)
    
    debug_info = {
        "found_lines": False,
        "rotation_angle": 0,
        "num_lines": 0,
        "edge_points": int(np.sum(edges > 0))
    }
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -2 <= angle <= 2:  # Ensure the angle is within a small range
                angles.append(angle)
        
        debug_info["num_lines"] = len(lines)
        
        if angles:
            median_angle = np.median(angles)
            debug_info["found_lines"] = True
            debug_info["rotation_angle"] = float(median_angle)
            debug_info["num_valid_angles"] = len(angles)
            
            center = (img_array.shape[1] // 2, img_array.shape[0] // 2)
            M_rotate = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(img_array, M_rotate, (img_array.shape[1], img_array.shape[0]), borderValue=(255, 255, 255))
            
            return Image.fromarray(rotated), debug_info
    
    return image, debug_info

def process_image(file_path: Path, out_path: Path) -> dict:
    """Process a single image file for rotation"""
    img = Image.open(file_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Get source folder structure from input path
    source_dir = Path(file_path).parts[-4:-1]
    
    # Rotate image and get debug info
    rotated, debug_info = hough_line_rotate(img)
    
    # Save rotated image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rotated.save(out_path, "JPEG", quality=100)
    
    # Build output path preserving full source hierarchy
    rel_path = Path(*source_dir) / out_path.name
    
    details = {
        "original_size": list(img.size),
        "rotated_size": list(rotated.size),
        "debug": debug_info
    }
    
    return {
        "outputs": [str(rel_path)],
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
            '.jpg': process_fn,
            '.jpeg': process_fn,
            '.tif': process_fn,
            '.tiff': process_fn,
            '.png': process_fn
        }
    )

def rotate(
    splits_folder: Path = typer.Argument(..., help="Input splits folder"),
    splits_manifest: Path = typer.Argument(..., help="Input splits manifest file"), 
    rotated_folder: Path = typer.Argument(..., help="Output folder for rotated images")
):
    """Rotate split document pages"""
    processor = BatchProcessor(
        input_manifest=splits_manifest,
        output_folder=rotated_folder,
        process_name="rotate",
        base_folder=splits_folder / "documents",  # Add /documents to match split.py's structure
        processor_fn=lambda f, o: process_document(f, o)
    )
    processor.process()

if __name__ == "__main__":
    typer.run(rotate)