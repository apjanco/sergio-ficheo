import typer
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from rich.progress import track
from rich.console import Console
import numpy as np
import cv2
from pdf2image import convert_from_path
import re
import os  # Add this import

console = Console()

def natural_sort_key(s):
    """Improved natural sorting that handles letters and numbers correctly"""
    import re
    # Convert to string if not already
    s = str(s)
    # Split the string into parts
    parts = re.split('([0-9]+)', s.lower())
    # Convert number parts to integers for proper numerical sorting
    return [int(part) if part.isdigit() else part for part in parts]

def is_likely_ruler(w: int, h: int, max_ruler_aspect_ratio: float = 15, max_ruler_width_ratio: float = 0.05) -> bool:
    """Check if the contour is likely a ruler based on its aspect ratio and width."""
    aspect_ratio = max(w / h, h / w)
    return aspect_ratio > max_ruler_aspect_ratio or w < max_ruler_width_ratio * h

def is_predominantly_black(image: Image.Image, threshold: float = 0.90) -> bool:
    """Check if the image is predominantly black."""
    img_array = np.array(image.convert("L"))
    black_pixels = np.sum(img_array < 50)  # Increase threshold for black pixels
    total_pixels = img_array.size
    return (black_pixels / total_pixels) > threshold

def contour_crop(image: Image.Image, blur_kernel=(3, 3), canny_threshold1=50, canny_threshold2=150, margin=10, min_size_ratio=0.2, max_aspect_ratio=10) -> Image.Image:
    """Automatic contour-based cropping for documents."""
    # Convert image to grayscale for edge detection
    img_gray = np.array(image.convert("L"))
    
    # Apply Gaussian blur to reduce noise
    img_blurred = cv2.GaussianBlur(img_gray, blur_kernel, 0)
    
    # Use Canny edge detection (adjust thresholds for better edge detection)
    edges = cv2.Canny(img_blurred, canny_threshold1, canny_threshold2)
    
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
        x_margin = margin
        y_margin = margin
        x = max(0, x - x_margin)
        y = max(0, y - y_margin)
        w = min(img_gray.shape[1] - x, w + 2 * x_margin)
        h = min(img_gray.shape[0] - y, h + 2 * y_margin)
        
        # Check if the cropped area is larger than the minimum size ratio of the original image
        min_width = img_gray.shape[1] * min_size_ratio
        min_height = img_gray.shape[0] * min_size_ratio
        aspect_ratio = max(w / h, h / w)
        if w >= min_width and h >= min_height and aspect_ratio <= max_aspect_ratio:
            # Crop the original image using the bounding box
            img_array = np.array(image)
            cropped_img = img_array[y:y+h, x:x+w]
            cropped_image = Image.fromarray(cropped_img)
            # Check if the cropped image is predominantly black
            if is_predominantly_black(cropped_image):
                console.print("[red]Cropped area is predominantly black, returning original image.")
                return image
            return cropped_image
    
    console.print("[red]No suitable contours found, returning original image.")
    return image

def split_and_crop_pdf(pdf_path: Path, out_dir: Path):
    try:
        relative_path = pdf_path.relative_to(pdf_path.parent.parent)
    except ValueError:
        relative_path = Path(*pdf_path.parts[len(pdf_path.parent.parent.parts):])
    
    # Keep _pdf suffix but remove appendix subfolder
    output_dir = out_dir / relative_path.parent / f"{relative_path.stem}_pdf"
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Rest of the function remains the same
    images = convert_from_path(pdf_path, dpi=300)
    for i, image in enumerate(images):
        image_path = output_dir / f"{pdf_path.stem}_page_{i + 1}.jpg"
        if image_path.exists():
            console.print(f"Skipping existing image: {image_path}")
            continue
        console.print(f"Processing page {i + 1} of {pdf_path.name}")
        cropped_img = contour_crop(image)
        cropped_img.save(image_path, "jpeg", quality=100)  # Save with maximum quality
        console.print(f"Saved cropped image: {image_path}")

def contour_crop_images(
    collection_path: Path = typer.Argument(..., help="Path to the folder containing files", exists=True),
    out_dir: Path = typer.Argument(..., help="Output directory to save the cropped images")
):
    collection_path = collection_path.resolve()
    out_dir = out_dir.resolve()

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Get all directories and sort them naturally
    all_dirs = []
    for root, dirs, _ in os.walk(collection_path, followlinks=True):
        root_path = Path(root)
        # Sort directories naturally
        dirs.sort(key=natural_sort_key)
        all_dirs.append(root_path)
    
    # Sort all directories naturally
    all_dirs.sort(key=natural_sort_key)

    # Process each directory in sorted order
    for dir_path in all_dirs:
        # Process PDFs
        pdf_files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith('.pdf')], 
                         key=natural_sort_key)
        for pdf_file in pdf_files:
            pdf_path = dir_path / pdf_file
            split_and_crop_pdf(pdf_path, out_dir)

        # Process images
        image_files = [f for f in os.listdir(dir_path) 
                      if any(f.lower().endswith(ext) 
                            for ext in ('.jpg', '.jpeg', '.tif', '.tiff', '.png'))]
        image_files.sort(key=natural_sort_key)
        
        for image_file in image_files:
            image_path = dir_path / image_file
            try:
                relative_path = image_path.relative_to(collection_path)
                # Keep _pdf suffix but remove appendix subfolder
                output_path = out_dir / relative_path.parent / f"{relative_path.stem}_pdf.jpg"
                
                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                if output_path.exists():
                    console.print(f"Skipping existing image: {output_path}")
                    continue

                img = Image.open(image_path)
                if img.size == (0, 0):
                    raise ValueError("Empty image")
                cropped_img = contour_crop(img)
                cropped_img.save(output_path, "jpeg", quality=100)
                console.print(f"Saved: {output_path}")
            except (UnidentifiedImageError, ValueError) as e:
                console.print(f"Skipping {image_file}: {e}")

    cropped_count = 0
    skipped_count = 0

    for image in track(images, description="Cropping images..."):
        try:
            # Calculate relative path from collection_path to maintain folder structure
            try:
                relative_path = image.relative_to(collection_path)
            except ValueError:
                # If relative_to fails, use the full path structure
                relative_path = Path(*image.parts[len(collection_path.parts):])
            
            # Keep _pdf suffix but remove appendix subfolder
            output_path = out_dir / relative_path.parent / f"{relative_path.stem}_pdf.jpg"
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.exists():
                console.print(f"Skipping existing image: {output_path}")
                skipped_count += 1
                continue

            img = Image.open(image)
            if img.size == (0, 0):
                raise ValueError("Empty image")
            cropped_img = contour_crop(img)
            cropped_img.save(output_path, "jpeg", quality=100)
            cropped_count += 1
            console.print(f"Saved: {output_path}")
        except (UnidentifiedImageError, ValueError) as e:
            console.print(f"Skipping {image.name}: {e}")
            skipped_count += 1

    console.print(f"[green]Cropping completed. Total images processed: {len(images)}")
    console.print(f"[green]Images cropped: {cropped_count}")
    console.print(f"[yellow]Images skipped: {skipped_count}")

if __name__ == "__main__":
    typer.run(contour_crop_images)