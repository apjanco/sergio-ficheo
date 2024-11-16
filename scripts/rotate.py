import typer
from PIL import Image
from pathlib import Path
from rich.progress import track
import numpy as np
import cv2

def hough_line_rotate(image: Image.Image, blur_kernel=(5, 5), canny_threshold1=50, canny_threshold2=150) -> Image.Image:
    """Rotate image based on Hough Line Transform."""
    img_array = np.array(image)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray, blur_kernel, 0)
    edges = cv2.Canny(img_blurred, canny_threshold1, canny_threshold2)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -2 <= angle <= 2:  # Ensure the angle is within a small range
                angles.append(angle)
        if angles:
            median_angle = np.median(angles)
            
            center = (img_array.shape[1] // 2, img_array.shape[0] // 2)
            M_rotate = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(img_array, M_rotate, (img_array.shape[1], img_array.shape[0]), borderValue=(0, 0, 0))
            
            return Image.fromarray(rotated)
    print("[red]No suitable lines found, returning original image.")
    return image

def rotate_images(
    collection_path: Path = typer.Argument(..., help="Path to the split images", exists=True),
    out_dir: Path = typer.Argument(..., help="Output directory to save the rotated images")
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
    for image in track(images, description="Rotating images..."):
        if image.name in skip:
            img = Image.open(image)
            img.save(out_dir / image.name)
        else:
            img = Image.open(image)
            rotated_img = hough_line_rotate(img)
            rotated_img.save(out_dir / image.name)

if __name__ == "__main__":
    typer.run(rotate_images)