import typer
from PIL import Image, ImageDraw, UnidentifiedImageError
from pathlib import Path
from rich.progress import track
from rich.console import Console
import cv2
import numpy as np
import re
import pytesseract

console = Console()

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

def remove_ruler(img):
    img1 = ImageDraw.Draw(img)
    w, h = img.size
    shape = [(0, 0), (w, 100)]  # Upper left, lower right
    img1.rectangle(shape, fill="#000000")
    return img

def should_split_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    aspect_ratio = width / height
    console.print(f"Image: {image_path}, Aspect Ratio: {aspect_ratio}")

    # Assuming notebook pages have a wider aspect ratio compared to standard documents
    if aspect_ratio > 1.2:  # Threshold to distinguish between notebook and document
        return True
    return False

def find_spiral_binding(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=10, maxLineGap=5)
    if lines is not None:
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -5 <= angle <= 5:  # Ensure the line is vertical
                vertical_lines.append((x1, y1, x2, y2))
        
        if len(vertical_lines) > 3:  # Check if there are enough vertical lines to form a spiral binding
            vertical_lines.sort(key=lambda line: line[1])  # Sort lines by their y-coordinate
            distances = [vertical_lines[i+1][1] - vertical_lines[i][1] for i in range(len(vertical_lines) - 1)]
            mean_distance = np.mean(distances)
            if all(abs(dist - mean_distance) < 15 for dist in distances):  # Check if distances are consistent
                return True
    return False

def has_clear_separation(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    center_line = width // 2

    # Check for a clear separation line in the middle
    left_half = img[:, :center_line]
    right_half = img[:, center_line:]
    left_mean = np.mean(left_half)
    right_mean = np.mean(right_half)
    separation_threshold = 5  # Adjust this threshold based on your needs

    if abs(left_mean - right_mean) > separation_threshold:
        return True
    return False

def detect_lines(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=5)
    return lines

def ocr_detect_text_blocks(image_path):
    img = cv2.imread(str(image_path))
    d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])
    text_blocks = []
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        text_blocks.append((x, y, w, h))
    return text_blocks

def split_based_on_text_blocks(image_path, text_blocks):
    img = Image.open(image_path)
    width, height = img.size
    center_line = width // 2

    left_blocks = [block for block in text_blocks if block[0] + block[2] / 2 < center_line]
    right_blocks = [block for block in text_blocks if block[0] + block[2] / 2 >= center_line]

    # Check if both sides have a significant amount of text
    left_text_area = sum(block[2] * block[3] for block in left_blocks)
    right_text_area = sum(block[2] * block[3] for block in right_blocks)
    total_text_area = left_text_area + right_text_area

    if left_text_area > 0.3 * total_text_area and right_text_area > 0.3 * total_text_area:
        # Find the best location to split based on the gaps between text blocks
        split_x = center_line
        min_gap = float('inf')
        for block in text_blocks:
            block_center = block[0] + block[2] / 2
            if center_line - 50 < block_center < center_line + 50:  # Consider blocks near the center line
                gap = abs(center_line - block_center)
                if gap < min_gap:
                    min_gap = gap
                    split_x = int(block_center)
        
        left = img.crop((0, 0, split_x, height))
        right = img.crop((split_x, 0, width, height))
        return left, right
    return None, None

def split(
    collection_path: Path = typer.Argument(..., help="Path to the collections", exists=True),
    out_dir: Path = typer.Argument(..., help="Output directory to save the split images"),
    skip_split: bool = typer.Option(False, help="Flag to skip splitting images")
):
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = ["*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.png"]
    images = []
    for ext in image_extensions:
        images.extend(collection_path.glob(f"**/{ext}"))
    images = sorted(images, key=natural_sort_key)

    # Do not split the first and last image in a notebook
    skip = [
        'SM_NPQ_C01_001.jpg', 'SM_NPQ_C01_104.jpg',
        'SM_NPQ_C02_001.jpg', 'SM_NPQ_C02_104.jpg',
        'SM_NPQ_C03_001.jpg', 'SM_NPQ_C03_104.jpg',
        'SM_NPQ_C04_001.jpg', 'SM_NPQ_C04_104.jpg',
        'SM_NPQ_C05_001.jpg', 'SM_NPQ_C05_071.jpg'
    ]

    for image in track(images, description="Processing images..."):
        try:
            relative_path = image.relative_to(collection_path)
            output_path = out_dir / relative_path.parent / (relative_path.stem + ".jpg")
            output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory structure exists
            if output_path.exists() or (out_dir / relative_path.parent / (relative_path.stem + "_left.jpg")).exists() or (out_dir / relative_path.parent / (relative_path.stem + "_right.jpg")).exists():
                console.print(f"Skipping existing image: {output_path}")
                continue
            if image.name in skip or skip_split or not should_split_image(image) or not find_spiral_binding(image) or not has_clear_separation(image):
                if should_split_image(image):  # Perform OCR text block detection only if the aspect ratio is right
                    text_blocks = ocr_detect_text_blocks(image)
                    left, right = split_based_on_text_blocks(image, text_blocks)
                    if left and right:
                        left.save(out_dir / relative_path.parent / (relative_path.stem + "_left.jpg"))
                        right.save(out_dir / relative_path.parent / (relative_path.stem + "_right.jpg"))
                    else:
                        img = Image.open(image)
                        img.save(output_path)
                else:
                    img = Image.open(image)
                    img.save(output_path)
            else:
                lines = detect_lines(image)
                if lines is not None:
                    img = Image.open(image)
                    width, height = img.size
                    center_line = width // 2
                    left = img.crop((0, 0, center_line, height))
                    right = img.crop((center_line, 0, width, height))
                    left.save(out_dir / relative_path.parent / (relative_path.stem + "_left.jpg"))
                    right.save(out_dir / relative_path.parent / (relative_path.stem + "_right.jpg"))
                else:
                    img = Image.open(image)
                    img.save(output_path)
        except (UnidentifiedImageError, ValueError) as e:
            console.print(f"Skipping {image.name}: {e}")

if __name__ == "__main__":
    typer.run(split)