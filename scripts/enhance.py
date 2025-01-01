import typer
from PIL import Image, ImageEnhance
from pathlib import Path
import numpy as np
import cv2
from utils.batch import BatchProcessor
from utils.processor import process_file
from rich.console import Console

console = Console()

def enhance_image(image: Image.Image) -> tuple[Image.Image, dict]:
    """Enhance image quality with contrast and clarity improvements"""
    img_array = np.array(image)
    
    # Convert to LAB color space for better enhancement control
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE with tuned parameters
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels back
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Convert back to PIL
    enhanced = Image.fromarray(enhanced)
    
    # Fine-tune with PIL enhancements
    contrast = ImageEnhance.Contrast(enhanced)
    enhanced = contrast.enhance(1.05)
    
    brightness = ImageEnhance.Brightness(enhanced)
    enhanced = brightness.enhance(1.02)
    
    # Record enhancement parameters
    params = {
        "clahe_clip_limit": 1.0,
        "clahe_grid_size": 8,
        "contrast_factor": 1.05,
        "brightness_factor": 1.02
    }
    
    return enhanced, params

def process_image(file_path: Path, out_path: Path) -> dict:
    """Process a single image file for enhancement"""
    img = Image.open(file_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Get source folder structure from input path
    source_dir = Path(file_path).parts[-4:-1]
    
    # Enhance image and get parameters
    enhanced, params = enhance_image(img)
    
    # Save enhanced image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enhanced.save(out_path, "JPEG", quality=100)
    
    # Build output path preserving full source hierarchy
    rel_path = Path(*source_dir) / out_path.name
    
    details = {
        "original_size": list(img.size),
        "enhanced_size": list(enhanced.size),
        "enhancement_params": params
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

def enhance(
    rotated_folder: Path = typer.Argument(..., help="Input rotated images folder"),
    rotated_manifest: Path = typer.Argument(..., help="Input rotated manifest file"),
    enhanced_folder: Path = typer.Argument(..., help="Output folder for enhanced images")
):
    """Enhance image quality of rotated document pages"""
    processor = BatchProcessor(
        input_manifest=rotated_manifest,
        output_folder=enhanced_folder,
        process_name="enhance",
        base_folder=rotated_folder / "documents",  # Add /documents to match rotation's structure
        processor_fn=lambda f, o: process_document(f, o)
    )
    processor.process()

if __name__ == "__main__":
    typer.run(enhance)
