import typer
from PIL import Image
from pathlib import Path
from rich.progress import track
import numpy as np
import cv2

def remove_background(image: Image.Image) -> Image.Image:
    """Remove black and purple background from the image using thresholding, masking, and blurring."""
    img_array = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Apply binary threshold
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    
    # Create a mask for black and purple shades
    lower_black_purple = np.array([0, 0, 0])
    upper_black_purple = np.array([180, 255, 128])  # Aggressive range for black and purple
    mask = cv2.inRange(img_array, lower_black_purple, upper_black_purple)
    
    # Find contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for the background connected to the borders
    background_mask = np.zeros_like(mask)
    for contour in contours:
        if cv2.pointPolygonTest(contour, (0, 0), False) >= 0 or \
           cv2.pointPolygonTest(contour, (img_array.shape[1] - 1, 0), False) >= 0 or \
           cv2.pointPolygonTest(contour, (0, img_array.shape[0] - 1), False) >= 0 or \
           cv2.pointPolygonTest(contour, (img_array.shape[1] - 1, img_array.shape[0] - 1), False) >= 0:
            cv2.drawContours(background_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Dilate the mask to leave a few pixels on the edge
    border_size = 5
    kernel = np.ones((border_size, border_size), np.uint8)
    background_mask = cv2.dilate(background_mask, kernel, iterations=1)
    
    # Replace masked pixels with white
    img_array[background_mask != 0] = [255, 255, 255]
    
    # Apply Gaussian blur to the border
    blurred_img = cv2.GaussianBlur(img_array, (border_size, border_size), 0)
    img_array[background_mask != 0] = blurred_img[background_mask != 0]

    return Image.fromarray(img_array)

def process_images(
    collection_path: Path = typer.Argument(..., help="Path to the images", exists=True),
    out_dir: Path = typer.Argument(..., help="Output directory to save the processed images")
):
    if not out_dir.exists():
         out_dir.mkdir(parents=True, exist_ok=True)

    images = list(collection_path.glob("**/*.jpg"))
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
    for image in track(images, description="Removing background..."):
        if image.name in skip:
            img = Image.open(image)
            img.save(out_dir / image.name)
        else:
            img = Image.open(image)
            processed_img = remove_background(img)
            processed_img.save(out_dir / image.name)

if __name__ == "__main__":
    typer.run(process_images)