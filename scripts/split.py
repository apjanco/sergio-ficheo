import typer
from PIL import Image
from pathlib import Path
import cv2
import numpy as np
import pytesseract
from datetime import datetime
from utils.batch import BatchProcessor
from utils.processor import process_file
from rich.console import Console

console = Console()

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

    if aspect_ratio > 1.2 and height > 1000 and width > 1000:
        return True
    return False

def is_large_image(image_path, threshold_area=2000000):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    area = width * height
    console.print(f"Image: {image_path}, Area: {area}")

    if area > threshold_area:
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
            if -5 <= angle <= 5:
                vertical_lines.append((x1, y1, x2, y2))
        
        if len(vertical_lines) > 3:
            vertical_lines.sort(key=lambda line: line[1])
            distances = [vertical_lines[i+1][1] - vertical_lines[i][1] for i in range(len(vertical_lines) - 1)]
            mean_distance = np.mean(distances)
            if all(abs(dist - mean_distance) < 15 for dist in distances):
                return True
    return False

def has_clear_separation(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    center_line = width // 2

    left_half = img[:, :center_line]
    right_half = img[:, center_line:]
    left_mean = np.mean(left_half)
    right_mean = np.mean(right_half)
    separation_threshold = 5

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
    console.print(f"OCR detected text blocks for {image_path}: {text_blocks}")
    return text_blocks

def find_vertical_rectangles(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > w and h > img.shape[0] * 0.5:
            rectangles.append((x, y, w, h))
    return rectangles

def split_based_on_text_blocks_and_rectangles(image_path, text_blocks):
    img = Image.open(image_path)
    width, height = img.size
    center_line = width // 2

    left_blocks = [block for block in text_blocks if block[0] + block[2] / 2 < center_line]
    right_blocks = [block for block in text_blocks if block[0] + block[2] / 2 >= center_line]

    left_text_area = sum(block[2] * block[3] for block in left_blocks)
    right_text_area = sum(block[2] * block[3] for block in right_blocks)
    total_text_area = left_text_area + right_text_area

    console.print(f"Left text area: {left_text_area}, Right text area: {right_text_area}, Total text area: {total_text_area}")

    rectangles = find_vertical_rectangles(image_path)
    if len(rectangles) == 2:
        rect1, rect2 = rectangles
        split_x = (rect1[0] + rect1[2] + rect2[0]) // 2
        left = img.crop((0, 0, split_x, height))
        right = img.crop((split_x, 0, width, height))
        console.print(f"Splitting image at {split_x} based on detected rectangles.")
        return left, right

    if left_text_area > 0.05 * total_text_area or right_text_area > 0.05 * total_text_area:
        split_x = center_line
        min_gap = float('inf')
        for block in text_blocks:
            block_center = block[0] + block[2] / 2
            if center_line - 100 < block_center < center_line + 100:
                gap = abs(center_line - block_center)
                if gap < min_gap:
                    min_gap = gap
                    split_x = int(block_center)
        
        if abs(split_x - center_line) < 50:
            split_x = center_line
        
        left = img.crop((0, 0, split_x, height))
        right = img.crop((split_x, 0, width, height))
        console.print(f"Splitting image at {split_x} based on OCR text blocks.")
        return left, right
    else:
        console.print(f"Not splitting image {image_path} due to insufficient text on one side.")
        return None, None

def process_single_image(file_path: Path, out_path: Path) -> dict:
    """Process a single image file"""
    details = {}
    should_split = should_split_image(file_path)
    has_binding = find_spiral_binding(file_path)
    has_sep = has_clear_separation(file_path)
    is_large = is_large_image(file_path)
    
    details.update({
        "should_split": should_split,
        "has_binding": has_binding,
        "has_separation": has_sep,
        "is_large": is_large
    })
    
    if should_split and (has_binding or has_sep or is_large):
        text_blocks = ocr_detect_text_blocks(file_path)
        left, right = split_based_on_text_blocks_and_rectangles(file_path, text_blocks)
        
        if left and right:
            left_path = out_path.parent / f"{out_path.stem}_left{out_path.suffix}"
            right_path = out_path.parent / f"{out_path.stem}_right{out_path.suffix}"
            
            left.save(left_path, quality=100)
            right.save(right_path, quality=100)
            
            return {
                "outputs": [
                    str(left_path.relative_to(left_path.parent.parent)),
                    str(right_path.relative_to(right_path.parent.parent))
                ],
                "split": True,
                "details": details
            }
    
    # If no split needed or splitting failed, just save original
    img = Image.open(file_path)
    img.save(out_path, quality=100)
    return {
        "outputs": [str(out_path.relative_to(out_path.parent.parent))],
        "split": False,
        "details": details
    }

def process_document(file_path: str, output_folder: Path) -> dict:
    """Process a single document file"""
    supported_types = {
        '.jpg': process_single_image,
        '.jpeg': process_single_image,
        '.tif': process_single_image,
        '.tiff': process_single_image,
        '.png': process_single_image
    }
    
    return process_file(
        file_path=file_path,
        output_folder=output_folder,
        process_fn=process_single_image,
        file_types=supported_types
    )

def split(
    crop_manifest: Path = typer.Argument(..., help="Input crop manifest file"),
    crops_folder: Path = typer.Argument(..., help="Input folder containing cropped images"),
    splits_folder: Path = typer.Argument(..., help="Output folder for split images")
):
    """Split images using crop manifest"""
    # We want to process the cropped images, not the original documents
    processor = BatchProcessor(
        input_manifest=crop_manifest,
        output_folder=splits_folder,
        process_name="split",
        processor_fn=process_document,
        base_folder=crops_folder / "documents"  # Important: crops are in crops_folder/documents
    )
    processor.process()

if __name__ == "__main__":
    typer.run(split)