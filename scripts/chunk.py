import typer
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from rich.progress import track
from rich.console import Console
import re
import pytesseract  # Add this import for OCR
import logging

console = Console()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

def get_kernel_size(image: Image.Image):
    width, height = image.size
    area = width * height
    if area < 500000:
        return 3
    elif area < 2000000:
        return 5
    return 7

def get_ocr_bounding_boxes(image: Image.Image, min_h=10):
    """Extract bounding boxes from Tesseract and skip very small boxes."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    boxes = []
    for i in range(len(data["level"])):
        h = data["height"][i]
        if h >= min_h:
            top = data["top"][i]
            bottom = top + h
            boxes.append((top, bottom))
    return boxes

def merge_overlapping_boxes(boxes, threshold=10):
    """Merge boxes that overlap or are close."""
    merged = []
    boxes = sorted(boxes, key=lambda b: b[0])  # sort by top
    for box in boxes:
        t, b = box
        if not merged or t > merged[-1][1] + threshold:
            merged.append([t, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    return merged

def adaptive_chunk_image(img: Image.Image, min_text_length=10) -> list:
    width, height = img.size
    logging.debug(f"Image size: width={width}, height={height}")

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    logging.debug(f"OCR box count: {len(data['level'])}")

    lines = {}
    for i in range(len(data["level"])):
        if data["text"][i].strip():  # Ensure the text is not empty
            line_num = data["line_num"][i]
            top = data["top"][i]
            bottom = top + data["height"][i]
            if line_num not in lines:
                lines[line_num] = [top, bottom]
            else:
                lines[line_num][1] = max(lines[line_num][1], bottom)

    logging.debug(f"Total lines detected: {len(lines)}")

    chunks = []
    for line in sorted(lines.values(), key=lambda x: x[0]):
        top, bottom = line
        roi = img.crop((0, top, width, bottom))
        ocr_result = pytesseract.image_to_string(roi).strip()
        logging.debug(f"Chunk top={top}, bottom={bottom}, text_len={len(ocr_result)}")
        if len(ocr_result) >= min_text_length:
            chunks.append(roi)

    logging.debug(f"Created {len(chunks)} full-width chunk(s) based on OCR lines.")
    return chunks

def chunk_images(
    collection_path: Path = typer.Argument(..., help="Path to the adjusted images", exists=True),
    out_dir: Path = typer.Argument(..., help="Output directory to save the chunked images"),
    min_text_length: int = typer.Option(10, help="Minimum text length to consider a chunk"),
    quality: int = typer.Option(100, help="Quality of the saved chunked images")
):
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:  # Added more common image formats
        images.extend(collection_path.glob(f"**/{ext}"))
    images = sorted(images, key=natural_sort_key)

    logging.info(f"Found {len(images)} image(s) to process.")
    for image in track(images, description="Chunking images..."):
        try:
            relative_path = image.relative_to(collection_path)
            output_path = out_dir / relative_path.parent / relative_path.stem
            output_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory structure exists

            img = Image.open(image).convert("RGB")

            # Directly use adaptive chunking without skipping logic
            chunks = adaptive_chunk_image(img, min_text_length)
            logging.debug(f"Adaptive chunking found {len(chunks)} chunk(s).")

            logging.debug(f"Final chunk count: {len(chunks)} for {relative_path}")
            for i, chunk in enumerate(chunks):
                chunk.save(output_path / f"{relative_path.stem}_chunk_{i}.jpg", quality=quality)
        except KeyboardInterrupt:
            logging.info("Operation interrupted by user.")
            return
        except UnidentifiedImageError as e:
            logging.error(f"Skipping {image.name} due to UnidentifiedImageError: {e}")
        except ValueError as e:
            logging.error(f"Skipping {image.name} due to ValueError: {e}")
        except Exception as e:
            logging.error(f"Skipping {image.name} due to unexpected error: {e}")

if __name__ == "__main__":
    typer.run(chunk_images)