import typer
from PIL import Image, ImageEnhance, UnidentifiedImageError
from pathlib import Path
from rich.progress import track
from rich.console import Console
import numpy as np  # Import numpy
import cv2
import re

console = Console()

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

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

def process_images(
    collection_path: Path = typer.Argument(..., help="Path to the images", exists=True),
    out_dir: Path = typer.Argument(..., help="Output directory to save the processed images")
):
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for ext in ["*.jpg"]:
        images.extend(collection_path.glob(f"**/{ext}"))
    images = sorted(images, key=natural_sort_key)

    skip = ['SM_NPQ_C01_001.jpg',
            'SM_NPQ_C01_104.jpg',
            'SM_NPQ_C02_001.jpg',
            'SM_NPQ_C02_104.jpg',
            'SM_NPQ_C03_001.jpg',
            'SM_NPQ_C03_104.jpg',
            'SM_NPQ_C04_001.jpg',
            'SM_NPQ_C04_104.jpg',
            'SM_NPQ_C05_001.jpg',
            'SM_NPQ_C05_071.jpg'
            ]
    for image in track(images, description="Adjusting images..."):
        try:
            relative_path = image.relative_to(collection_path)
            output_path = out_dir / relative_path.parent / (relative_path.stem + ".jpg")
            output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory structure exists
            if output_path.exists():
                console.print(f"Skipping existing image: {output_path}")
                continue
            if image.name in skip:
                img = Image.open(image)
                img.save(output_path, quality=100)
            else:
                img = Image.open(image)
                if img.size == (0, 0):
                    raise ValueError("Empty image")
                adjusted_img = adjust_image(img)
                adjusted_img.save(output_path, quality=100)
        except (UnidentifiedImageError, ValueError) as e:
            console.print(f"Skipping {image.name}: {e}")

if __name__ == "__main__":
    typer.run(process_images)