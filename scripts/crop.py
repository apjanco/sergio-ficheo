import typer 
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from rich.progress import track
import numpy as np
import cv2

def contour_crop(image: Image.Image, blur_kernel=(5, 5), canny_threshold1=50, canny_threshold2=150) -> Image.Image:
    """Automatic contour-based cropping for documents."""
    # Convert image to grayscale for edge detection
    img_gray = np.array(image.convert("L"))
    
    # Apply Gaussian blur to reduce noise
    img_blurred = cv2.GaussianBlur(img_gray, blur_kernel, 0)
    
    # Use Canny edge detection (adjust thresholds for better edge detection)
    edges = cv2.Canny(img_blurred, canny_threshold1, canny_threshold2)
    
    # Apply dilation to enhance edges
    dilated_edges = cv2.dilate(edges, None, iterations=2)
    
    # Find contours in the edges
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Sort contours by area and get the largest one (filtering by size)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = contours[0]
        
        # Get the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the original image using the bounding box
        img_array = np.array(image)
        cropped_img = img_array[y:y+h, x:x+w]
        return Image.fromarray(cropped_img)
    else:
        print("[red]No contours found, returning original image.")
    return image

def contour_crop_images(
    collection_path: Path = typer.Argument(..., help="Path to the collections", exists=True),
    out_dir: Path = typer.Argument(..., help="Output directory to save the cropped images")
):
    if not out_dir.exists():
         out_dir.mkdir(parents=True, exist_ok=True)

    images = list(collection_path.glob("**/*.[jJ][pP][gG]")) + \
             list(collection_path.glob("**/*.[jJ][pP][eE][gG]")) + \
             list(collection_path.glob("**/*.[tT][iI][fF]")) + \
             list(collection_path.glob("**/*.[tT][iI][fF][fF]"))
    # List of files to skip during cropping
    skip = [''
            ]
    for image in track(images, description="Cropping images..."):
        try:
            if image.name in skip:
                img = Image.open(image)
                img.save(out_dir / (image.stem + ".jpg"))
            else:
                img = Image.open(image)
                if img.size == (0, 0):
                    raise ValueError("Empty image")
                cropped_img = contour_crop(img)  # Apply contour cropping
                cropped_img.save(out_dir / (image.stem + ".jpg"))  # Save as JPG with the original file name stem
        except (UnidentifiedImageError, ValueError) as e:
            print(f"Skipping {image.name}: {e}")

if __name__ == "__main__":
    typer.run(contour_crop_images)