import typer
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from rich.progress import track, Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
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

def split_and_crop_pdf(pdf_path: Path, out_dir: Path, collection_path: Path, progress=None):
    try:
        # Keep the relative path structure
        relative_path = pdf_path.relative_to(collection_path)
        output_dir = out_dir / relative_path.parent / f"{pdf_path.stem}_pdf"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if any pages exist already
        existing_pages = list(output_dir.glob(f"{pdf_path.stem}_page_*.jpg"))
        if existing_pages:
            console.print(f"[yellow]Skipping PDF with existing pages: {pdf_path.name}")
            return
        
        console.print(f"[blue]Converting PDF: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=300)
        
        for i, image in enumerate(images):
            image_path = output_dir / f"{pdf_path.stem}_page_{i + 1}.jpg"
            if image_path.exists():
                console.print(f"[yellow]Skipping existing: {image_path.name}")
                continue
                
            cropped_img = contour_crop(image)
            cropped_img.save(image_path, "jpeg", quality=100)
            if progress:
                progress.console.print(f"[green]Saved: {image_path.name}")
    except Exception as e:
        console.print(f"[red]Error processing PDF {pdf_path}: {e}")

def find_files(path: Path, out_dir: Path):
    """Generator that yields files from directories in sorted order"""
    try:
        # Get and sort immediate subdirectories
        subdirs = sorted([d for d in path.iterdir() if d.is_dir() and not d.resolve().is_relative_to(out_dir)],
                        key=lambda x: natural_sort_key(x.name))
        
        # Add the base directory if it contains files
        if any(f.is_file() for f in path.iterdir()):
            subdirs.insert(0, path)
        
        # Process each directory
        for current_dir in subdirs:
            console.print(f"[blue]Processing directory: {current_dir}")
            
            # Get and sort all files in current directory
            files = []
            for item in current_dir.iterdir():
                if item.is_file():
                    if item.suffix.lower() == '.pdf':
                        files.append(('pdf', item))
                    elif item.suffix.lower() in ('.jpg', '.jpeg', '.tif', '.tiff', '.png'):
                        files.append(('image', item))
            
            # Sort files by name and yield them
            for file_type, file_path in sorted(files, key=lambda x: natural_sort_key(x[1].name)):
                yield (file_type, file_path)

            # Recursively process subdirectories of current directory
            if current_dir != path:  # Avoid infinite recursion
                for item in find_files(current_dir, out_dir):
                    yield item
                    
    except PermissionError as e:
        console.print(f"[red]Permission denied accessing {path}: {e}")

def contour_crop_images(
    collection_path: Path = typer.Argument(..., help="Path to the folder containing files", exists=True),
    out_dir: Path = typer.Argument(..., help="Output directory to save the cropped images")
):
    collection_path = collection_path.resolve()
    out_dir = out_dir.resolve()

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="cyan"),
        TaskProgressColumn(),
        console=console
    ) as progress:
        pdf_count = 0
        image_count = 0
        current_task = progress.add_task("[cyan]Processing files...", total=None)

        for file_type, file_path in find_files(collection_path, out_dir):
            try:
                if file_type == 'pdf':
                    progress.update(current_task, description=f"[cyan]Processing PDF: {file_path.name}")
                    split_and_crop_pdf(file_path, out_dir, collection_path, progress)
                    pdf_count += 1
                else:
                    progress.update(current_task, description=f"[green]Processing image: {file_path.name}")
                    relative_path = file_path.relative_to(collection_path)
                    # Keep original name for non-PDF images
                    output_path = out_dir / relative_path.parent / f"{relative_path.stem}.jpg"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if output_path.exists():
                        console.print(f"[yellow]Skipping existing: {output_path.name}")
                        continue

                    img = Image.open(file_path)
                    if img.size == (0, 0):
                        raise ValueError("Empty image")
                    cropped_img = contour_crop(img)
                    cropped_img.save(output_path, "jpeg", quality=100)
                    image_count += 1
            except (UnidentifiedImageError, ValueError) as e:
                console.print(f"[red]Error processing {file_path.name}: {e}")

    console.print("[green]Processing completed!")
    console.print(f"[green]Total PDFs processed: {pdf_count}")
    console.print(f"[green]Total images processed: {image_count}")

if __name__ == "__main__":
    typer.run(contour_crop_images)