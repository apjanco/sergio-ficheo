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

# Image processing functions specific to cropping
def is_likely_ruler(w: int, h: int, max_ruler_aspect_ratio: float = 15, max_ruler_width_ratio: float = 0.05) -> bool:
    """Check if the contour is likely a ruler based on its aspect ratio and width."""
    aspect_ratio = max(w / h, h / w)
    return aspect_ratio > max_ruler_aspect_ratio or w < max_ruler_width_ratio * h

def is_predominantly_black(image: Image.Image, threshold: float = 0.90) -> bool:
    """Check if the image is predominantly black."""
    img_array = np.array(image.convert("L"))
    black_pixels = np.sum(img_array < 50)
    total_pixels = img_array.size
    return (black_pixels / total_pixels) > threshold

def contour_crop(image: Image.Image) -> Image.Image:
    """Automatic contour-based cropping for documents."""
    # Convert image to grayscale for edge detection
    img_gray = np.array(image.convert("L"))
    
    # Apply Gaussian blur to reduce noise
    img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    # Use Canny edge detection (adjust thresholds for better edge detection)
    edges = cv2.Canny(img_blurred, 50, 150)
    
    # Apply dilation to enhance edges
    dilated_edges = cv2.dilate(edges, None, iterations=5)  # Increase iterations for better detection
    
    # Find contours in the edges
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip contours that are likely rulers
        if is_likely_ruler(w, h):
            continue
        
        # Add margin to the bounding box
        x_margin = 10
        y_margin = 10
        x = max(0, x - x_margin)
        y = max(0, y - y_margin)
        w = min(img_gray.shape[1] - x, w + 2 * x_margin)
        h = min(img_gray.shape[0] - y, h + 2 * y_margin)
        
        # Check if the cropped area is larger than the minimum size ratio of the original image
        min_width = img_gray.shape[1] * 0.2
        min_height = img_gray.shape[0] * 0.2
        aspect_ratio = max(w / h, h / w)
        if w >= min_width and h >= min_height and aspect_ratio <= 10:
            # Crop the original image using the bounding box
            img_array = np.array(image)
            cropped_img = img_array[y:y+h, x:x+w]
            cropped_image = Image.fromarray(cropped_img)
            # Check if the cropped image is predominantly black
            if is_predominantly_black(cropped_image):
                return image
            return cropped_image
    
    return image

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
    cropped = contour_crop(img)
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
        cropped = contour_crop(image)
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
    
    # All processing through process_file
    return process_file(
        file_path=str(file_path),
        output_folder=output_folder,
        process_fn=process_pdf if file_path.suffix.lower() == '.pdf' else process_image,
        file_types=supported_types
    )

def crop(
    documents_folder: Path = typer.Argument(..., help="Input documents folder"),
    documents_manifest: Path = typer.Argument(..., help="Input documents manifest file"),
    crops_folder: Path = typer.Argument(..., help="Output folder for cropped images")
):
    """Crop images from documents"""
    processor = BatchProcessor(
        input_manifest=documents_manifest,
        output_folder=crops_folder,
        process_name="crop",
        processor_fn=process_document,
        base_folder=documents_folder,  # Remove /documents since it will be in the paths
        use_source=True  # Use source paths from manifest
    )
    processor.process()

if __name__ == "__main__":
    typer.run(crop)