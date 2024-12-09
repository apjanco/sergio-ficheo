import typer
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from rich.progress import track
import numpy as np
import cv2
from pdf2image import convert_from_path

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

def contour_crop(image: Image.Image, blur_kernel=(5, 5), canny_threshold1=30, canny_threshold2=100, margin=50, min_size_ratio=0.2, max_aspect_ratio=15) -> Image.Image:
    """Automatic contour-based cropping for documents."""
    # Convert image to grayscale for edge detection
    img_gray = np.array(image.convert("L"))
    
    # Apply Gaussian blur to reduce noise
    img_blurred = cv2.GaussianBlur(img_gray, blur_kernel, 0)
    
    # Use Canny edge detection (adjust thresholds for better edge detection)
    edges = cv2.Canny(img_blurred, canny_threshold1, canny_threshold2)
    
    # Apply dilation to enhance edges
    dilated_edges = cv2.dilate(edges, None, iterations=10)  # Increase iterations for better detection
    
    # Find contours in the edges
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip contours that are likely rulers
        if is_likely_ruler(w, h):
            continue
        
        # Add margin to the bounding box
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img_gray.shape[1] - x, w + 2 * margin)
        h = min(img_gray.shape[0] - y, h + 2 * margin)
        
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
                print("[red]Cropped area is predominantly black, returning original image.")
                return image
            return cropped_image
    
    print("[red]No suitable contours found, returning original image.")
    return image

def split_and_crop_pdf(pdf_path: Path, out_dir: Path):
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    images = convert_from_path(pdf_path, dpi=300)  # Convert PDF to images with maximum resolution
    for i, image in enumerate(images):
        image_path = out_dir / f"{pdf_path.stem}_page_{i + 1}.jpg"
        if image_path.exists():
            print(f"Skipping existing image: {image_path}")
            continue
        print(f"Processing page {i + 1} of {pdf_path.name}")
        cropped_img = contour_crop(image)
        cropped_img.save(image_path, "JPEG")
        print(f"Saved cropped image: {image_path}")

def contour_crop_images(
    collection_path: Path = typer.Argument(..., help="Path to the folder containing files", exists=True),
    out_dir: Path = typer.Argument(..., help="Output directory to save the cropped images")
):
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Process PDF files first
    pdf_files = list(collection_path.glob("**/*.pdf"))
    for pdf_file in pdf_files:
        split_and_crop_pdf(pdf_file, out_dir)

    # Process all image files
    images = list(collection_path.glob("**/*.[jJ][pP][gG]")) + \
             list(collection_path.glob("**/*.[jJ][pP][eE][gG]")) + \
             list(collection_path.glob("**/*.[tT][iI][fF]")) + \
             list(collection_path.glob("**/*.[tT][iI][fF][fF]")) + \
             list(collection_path.glob("**/*.[pP][nN][gG]"))

    if not images:
        print("No images found to process.")
        return

    for image in track(images, description="Cropping images..."):
        try:
            img = Image.open(image)
            if img.size == (0, 0):
                raise ValueError("Empty image")
            cropped_img = contour_crop(img)  # Apply contour cropping
            cropped_img.save(out_dir / (image.stem + ".jpg"))  # Save as JPG with the original file name stem
            print(f"Cropped and saved image: {out_dir / (image.stem + '.jpg')}")
        except (UnidentifiedImageError, ValueError) as e:
            print(f"Skipping {image.name}: {e}")

if __name__ == "__main__":
    typer.run(contour_crop_images)