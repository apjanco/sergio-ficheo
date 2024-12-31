import typer
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
from pdf2image import convert_from_path
from utils.batch import BatchProcessor
from utils.processor import process_file
from rich.console import Console

console = Console()

def detect_split_point(image: Image.Image, threshold_ratio: float = 0.15) -> tuple[bool, int]:
    """Detect if image should be split and return split point"""
    # Check if image is in landscape orientation
    width, height = image.size
    aspect_ratio = width / height
    
    # Don't split if image is portrait (aspect ratio < 1.2)
    if aspect_ratio < 1.2:
        return False, None
    
    # Convert to grayscale numpy array
    img_array = np.array(image.convert("L"))
    
    # Calculate midpoint region bounds
    mid_region_start = int(width * (0.5 - threshold_ratio))
    mid_region_end = int(width * (0.5 + threshold_ratio))
    
    # Look for darkest vertical line in middle region
    min_sum = float('inf')
    split_x = None
    
    for x in range(mid_region_start, mid_region_end):
        vertical_sum = np.sum(img_array[:, x])
        if vertical_sum < min_sum:
            min_sum = vertical_sum
            split_x = x
            
    # Determine if split is needed based on darkness of line
    avg_darkness = min_sum / height
    should_split = avg_darkness < 180  # More conservative threshold
    
    return should_split, split_x

def split_image(image: Image.Image) -> list[Image.Image]:
    """Split an image into left and right pages if needed"""
    should_split, split_point = detect_split_point(image)
    
    if not should_split:
        return [image]
        
    # Split into left and right pages
    width, height = image.size
    left_page = image.crop((0, 0, split_point, height))
    right_page = image.crop((split_point, 0, width, height))
    
    return [left_page, right_page]

def process_image(file_path: Path, out_path: Path) -> dict:
    """Process a single image file"""
    # Convert any image format to RGB
    img = Image.open(file_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Split image and save parts
    parts = split_image(img)
    outputs = []
    details = {"original_size": img.size}
    
    for i, part in enumerate(parts):
        # Create output filename
        if len(parts) > 1:
            part_path = out_path.parent / f"{out_path.stem}_part_{i+1}.jpg"
        else:
            part_path = out_path.with_suffix('.jpg')
            
        # Save split part
        part.save(part_path, "JPEG", quality=100)
        outputs.append(str(part_path.relative_to(part_path.parent.parent)))
        details[f"part_{i+1}_size"] = part.size
    
    return {
        "outputs": outputs,
        "details": details
    }

def process_pdf(file_path: Path, out_path: Path) -> dict:
    """Process a PDF file"""
    outputs = []
    details = {}
    
    # Convert PDF pages to images
    images = convert_from_path(file_path, dpi=300)
    
    for i, image in enumerate(images):
        # Process each page
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Split page if needed
        parts = split_image(image)
        
        for j, part in enumerate(parts):
            # Create output filename
            if len(parts) > 1:
                part_path = out_path.parent / f"{out_path.stem}_page_{i+1}_part_{j+1}.jpg"
            else:
                part_path = out_path.parent / f"{out_path.stem}_page_{i+1}.jpg"
                
            # Save split part
            part.save(part_path, "JPEG", quality=100)
            outputs.append(str(part_path.relative_to(part_path.parent.parent)))
            
        details[f"page_{i+1}"] = {
            "original_size": image.size,
            "parts": len(parts)
        }
    
    return {
        "outputs": outputs,
        "details": details
    }

def process_document(file_path: str, output_folder: Path) -> dict:
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
    
    # Process the file using the same path structure from crops
    return process_file(
        file_path=str(file_path),
        output_folder=output_folder,
        process_fn=process_image,  # Always use process_image since PDFs are already converted to images
        file_types=supported_types
    )

def split(
    crops_folder: Path = typer.Argument(..., help="Input crops folder"),
    crops_manifest: Path = typer.Argument(..., help="Input crops manifest file"),
    splits_folder: Path = typer.Argument(..., help="Output folder for split images")
):
    """Split cropped book pages into individual pages"""
    processor = BatchProcessor(
        input_manifest=crops_manifest,
        output_folder=splits_folder,
        process_name="split",
        processor_fn=process_document,
        base_folder=crops_folder / "documents"  # Add /documents to match crop.py's structure
    )
    processor.process()

if __name__ == "__main__":
    typer.run(split)