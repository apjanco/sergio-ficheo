import typer
from PIL import Image, ImageEnhance, UnidentifiedImageError
from pathlib import Path
from rich.console import Console
import numpy as np
import cv2
from utils.batch import BatchProcessor
from utils.files import ensure_dirs

console = Console()

def adjust_image(image: Image.Image) -> Image.Image:
    """Enhance handwritten text visibility and reduce background noise without radically changing the image."""
    img_array = np.array(image)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    # Split the LAB image to different channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel with more subtle parameters
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel back with A and B channels
    limg = cv2.merge((cl, a, b))
    
    # Convert back to RGB color space
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Convert back to PIL Image
    enhanced_img = Image.fromarray(enhanced_img)
    
    # Further enhance contrast using PIL with more subtle enhancement
    enhancer = ImageEnhance.Contrast(enhanced_img)
    enhanced_img = enhancer.enhance(1.05)  # Slightly increase contrast
    
    # Enhance brightness to make the background whiter with more subtle enhancement
    enhancer = ImageEnhance.Brightness(enhanced_img)
    enhanced_img = enhancer.enhance(1.02)  # Slightly increase brightness
    
    return enhanced_img

def process_document(file_path: str, output_folder: Path) -> dict:
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
        adjusted = adjust_image(img)
        adjusted.save(output_path, quality=100)
        
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

def adjust(
    documents_manifest: Path = typer.Argument(..., help="Input manifest file"),
    adjusted_folder: Path = typer.Argument(..., help="Output folder for adjusted images")
):
    """Adjust images using manifest order"""
    processor = BatchProcessor(
        input_manifest=documents_manifest,
        output_folder=adjusted_folder,
        process_name="adjust",
        processor_fn=process_document
    )
    processor.process()

if __name__ == "__main__":
    typer.run(adjust)