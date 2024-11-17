import typer
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from rich.progress import track
import pytesseract
import cv2
import numpy as np

def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess the image to enhance text visibility."""
    img_array = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding
    processed_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return Image.fromarray(processed_img)

def detect_text_lines(image: Image.Image):
    """Detect lines of text in the image using OCR and return their y-coordinates."""
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Configure Tesseract for handwriting recognition
    custom_config = r'--oem 1 --psm 6'
    
    ocr_result = pytesseract.image_to_data(preprocessed_image, config=custom_config, output_type=pytesseract.Output.DICT)
    line_y_coords = []
    for i in range(len(ocr_result['level'])):
        if ocr_result['level'][i] == 5:  # Level 5 corresponds to line level
            y = ocr_result['top'][i]
            h = ocr_result['height'][i]
            line_y_coords.append((y, y + h))
    return line_y_coords

def find_closest_line(y_coords, target_y):
    """Find the closest line to the target y-coordinate."""
    closest_line = min(y_coords, key=lambda line: abs(line[0] - target_y))
    return closest_line

def smart_chunk_image(image: Image.Image, num_chunks: int, overlap_percentage: float):
    width, height = image.size
    line_y_coords = detect_text_lines(image)
    
    # Calculate approximate chunk height
    approx_chunk_height = height // num_chunks
    overlap = int(approx_chunk_height * (overlap_percentage / 100))
    
    chunks = []
    start_y = 0
    for i in range(num_chunks):
        target_y = start_y + approx_chunk_height
        closest_line = find_closest_line(line_y_coords, target_y)
        end_y = closest_line[0]
        chunks.append(image.crop((0, max(0, start_y - overlap), width, min(height, end_y + overlap))))
        start_y = end_y
    # Ensure the last chunk includes the remaining part of the image
    if start_y < height:
        chunks.append(image.crop((0, max(0, start_y - overlap), width, height)))
    return chunks

def chunk_images(
    collection_path: Path = typer.Argument(..., help="Path to the images", exists=True),
    out_dir: Path = typer.Argument(..., help="Output directory to save the chunked images"),
    num_chunks: int = typer.Option(3, help="Number of chunks per image"),
    overlap_percentage: float = typer.Option(5.0, help="Percentage overlap between chunks")
):
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    images = list(collection_path.glob("**/*.jpg"))
    for image in track(images, description="Chunking images..."):
        try:
            img = Image.open(image)
            if img.size == (0, 0):
                raise ValueError("Empty image")
            chunks = smart_chunk_image(img, num_chunks, overlap_percentage)
            image_folder = out_dir / image.stem
            image_folder.mkdir(parents=True, exist_ok=True)
            for idx, chunk in enumerate(chunks):
                chunk.save(image_folder / f"{image.stem}_chunk_{idx + 1}.jpg")
        except (UnidentifiedImageError, ValueError) as e:
            print(f"Skipping {image.name}: {e}")

if __name__ == "__main__":
    typer.run(chunk_images)