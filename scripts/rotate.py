import typer
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from rich.progress import track
from rich.console import Console
import numpy as np
import cv2
from utils.batch import BatchProcessor

console = Console()

def hough_line_rotate(image: Image.Image, blur_kernel=(5, 5), canny_threshold1=50, canny_threshold2=150) -> Image.Image:
    """Rotate image based on Hough Line Transform."""
    img_array = np.array(image)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray, blur_kernel, 0)
    edges = cv2.Canny(img_blurred, canny_threshold1, canny_threshold2)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -2 <= angle <= 2:  # Ensure the angle is within a small range
                angles.append(angle)
        if angles:
            median_angle = np.median(angles)
            
            center = (img_array.shape[1] // 2, img_array.shape[0] // 2)
            M_rotate = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(img_array, M_rotate, (img_array.shape[1], img_array.shape[0]), borderValue=(0, 0, 0))
            
            return Image.fromarray(rotated)
    console.print("[red]No suitable lines found, returning original image.")
    return image

def process_document(file_path: str, output_folder: Path) -> dict:
    # Process single document for rotation
    file_path = Path(file_path)
    try:
        output_path = output_folder / file_path.with_suffix('.jpg')
        if output_path.exists():
            return {
                "source": str(file_path),
                "outputs": [str(output_path)],
                "success": True,
                "skipped": True
            }
            
        ensure_dirs(output_path)
        img = Image.open(file_path)
        rotated = hough_line_rotate(img)
        rotated.save(output_path, quality=100)
        
        return {
            "source": str(file_path),
            "outputs": [str(output_path)],
            "success": True
        }
    except Exception as e:
        return {
            "source": str(file_path),
            "error": str(e),
            "success": False
        }

def rotate(
    documents_manifest: Path = typer.Argument(..., help="Input manifest file"),
    rotated_folder: Path = typer.Argument(..., help="Output folder for rotated images")
):
    """Rotate images using manifest order"""
    processor = BatchProcessor(
        input_manifest=documents_manifest,
        output_folder=rotated_folder,
        process_name="rotate", 
        processor_fn=process_document
    )
    processor.process()

if __name__ == "__main__":
    typer.run(rotate)