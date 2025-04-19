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
    """Automatic contour-based cropping for documents with improved handling of messy backgrounds."""
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    # Get dimensions
    height, width = img_gray.shape
    
    # Apply bilateral filter to reduce noise while preserving edges
    blurred = cv2.bilateralFilter(img_gray, 9, 75, 75)
    
    # Apply adaptive thresholding first (more reliable for documents)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Dilate to connect components
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours from thresholded image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, try Canny edge detection
    if not contours:
        edges = cv2.Canny(blurred, 30, 100)  # More sensitive thresholds
        dilated_edges = cv2.dilate(edges, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Process large contours
    for contour in contours[:3]:  # Try top 3 largest contours
        area = cv2.contourArea(contour)
        img_area = width * height
        
        # Lower threshold to 10% of image area
        if area < (img_area * 0.1):
            continue
        
        # Try to get a more precise polygon approximation
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If we have a quadrilateral, use it directly
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
        else:
            # For non-quadrilateral, try to find the largest rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            x, y, w, h = cv2.boundingRect(box)
        
        # Skip extreme aspect ratios (more lenient now)
        aspect_ratio = max(w/h, h/w)
        if aspect_ratio > 8:  # More lenient aspect ratio
            continue
        
        # Add larger margin (3%)
        margin_x = int(w * 0.03)
        margin_y = int(h * 0.03)
        
        # Apply margin with bounds checking
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(width - x, w + 2 * margin_x)
        h = min(height - y, h + 2 * margin_y)
        
        # Crop image
        cropped = img_array[y:y+h, x:x+w]
        
        # Check if the cropped area is predominantly black
        if is_predominantly_black(Image.fromarray(cropped)):
            continue
            
        return Image.fromarray(cropped)
    
    # If still no suitable contours found, try OTSU thresholding
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dilated_otsu = cv2.dilate(thresh_otsu, kernel, iterations=3)
    contours_otsu, _ = cv2.findContours(dilated_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort by area
    contours_otsu = sorted(contours_otsu, key=cv2.contourArea, reverse=True)
    
    # Try to find a suitable contour
    for contour in contours_otsu[:2]:
        x, y, w, h = cv2.boundingRect(contour)
        # Lower threshold to 15% of image
        if (w*h) > (width*height*0.15):
            # Add margin
            margin_x = int(w * 0.03)
            margin_y = int(h * 0.03)
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            w = min(width - x, w + 2 * margin_x)
            h = min(height - y, h + 2 * margin_y)
            # Crop image
            cropped = img_array[y:y+h, x:x+w]
            return Image.fromarray(cropped)
    
    # If all else fails, return original image
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