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

from utils.batch import BatchProcessor
from utils.processor import process_file

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

def deskew_image(pil_img: Image.Image) -> Image.Image:
    """
    Deskew an image using OpenCV's minAreaRect on the largest contour.
    This approach tries to detect the most prominent rotation in the image
    and rotate the image to correct it.
    If no rotation is found, returns the original image.
    """
    # Convert PIL to OpenCV
    cv_img = np.array(pil_img.convert('L'))  # grayscale
    # Threshold and invert to get text as white on black for better contour detection
    _, thresh = cv2.threshold(cv_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pil_img  # No contours => can't deskew

    # Pick the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    # Compute minAreaRect
    rot_rect = cv2.minAreaRect(largest_contour)
    angle = rot_rect[-1]
    # minAreaRect angle convention: if text is horizontal, angle ~ 0 or -90
    # We want to rotate it so angle becomes 0 or 180.

    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    if abs(angle) < 0.1:
        # Very small angle => no real deskew needed
        return pil_img

    (h, w) = cv_img.shape
    center = (w // 2, h // 2)
    # Compute rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Rotate
    rotated = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Convert back to PIL
    deskewed_pil = Image.fromarray(rotated).convert("RGB")
    return deskewed_pil

def get_connected_component_lines(
    img: Image.Image,
    line_threshold=10,
    min_box_width=10,
    min_box_height=10
) -> list:
    """
    Detect lines via connected components (OpenCV).
    Returns a list of (top, bottom) bounding boxes.
    """
    cv_img = np.array(img.convert('L'))  # grayscale
    # Binarize (invert so text is white on black)
    _, thresh = cv2.threshold(cv_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Optionally do morphological ops to connect broken lines:
    # kernel = np.ones((2,2), np.uint8)
    # thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box_info = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= min_box_width and h >= min_box_height:
            top = y
            bottom = y + h
            box_info.append((top, bottom))

    # Merge boxes that are close
    box_info.sort(key=lambda x: x[0])
    lines = []
    if box_info:
        current_top, current_bottom = box_info[0]
        for (t, b) in box_info[1:]:
            if t <= current_bottom + line_threshold:
                current_bottom = max(current_bottom, b)
            else:
                lines.append((current_top, current_bottom))
                current_top, current_bottom = t, b
        lines.append((current_top, current_bottom))
    return lines

def adaptive_chunk_image(img: Image.Image, min_text_length=10) -> list:
    """
    Hybrid approach that:
      1. Deskews the image (if needed).
      2. Collects bounding boxes from Tesseract.
      3. Fallback to connected-component lines if Tesseract is sparse.
      4. Merges boxes and covers every vertical region (no data lost).
      5. Subdivides large segments so chunks don't get too big.
      6. Returns a list of dicts, each with:
         { "image": cropped_chunk, "top": top_px, "bottom": bottom_px, "text_len": length_of_OCR_text }
    """
    # 1. Deskew the image
    deskewed_img = deskew_image(img)
    width, height = deskewed_img.size
    logging.debug(f"[adaptive_chunk_image] Deskewed image size: {width}x{height}")

    # 2. Collect Tesseract bounding boxes (word-level).
    data = pytesseract.image_to_data(deskewed_img, output_type=pytesseract.Output.DICT)
    tess_boxes = []
    for i in range(len(data["level"])):
        text = data["text"][i].strip()
        if text:  # non-empty recognized text
            top = data["top"][i]
            bottom = top + data["height"][i]
            tess_boxes.append((top, bottom))

    tess_boxes.sort(key=lambda x: x[0])
    logging.debug(f"Tesseract bounding boxes found: {len(tess_boxes)}")

    # 3. Fallback to connected components if Tesseract found < 3 lines
    fallback_needed = (len(tess_boxes) < 3)
    if fallback_needed:
        logging.debug("Running connected components fallback.")
        cc_lines = get_connected_component_lines(deskewed_img, line_threshold=10)
        logging.debug(f"Connected components lines found: {len(cc_lines)}")
    else:
        cc_lines = []

    # Combine Tesseract boxes + CC lines
    all_boxes = tess_boxes + cc_lines
    all_boxes.sort(key=lambda x: x[0])

    # Merge boxes that are close
    line_threshold = 10
    merged_boxes = []
    for box in all_boxes:
        if not merged_boxes:
            merged_boxes.append(list(box))
        else:
            if box[0] <= merged_boxes[-1][1] + line_threshold:
                merged_boxes[-1][1] = max(merged_boxes[-1][1], box[1])
            else:
                merged_boxes.append(list(box))

    logging.debug(f"Total merged bounding boxes: {len(merged_boxes)}")

    # 4. Build segments to cover entire image.
    #    We fill from 0->first_box, each bounding box, and gaps between them.
    cover_segments = []
    if not merged_boxes:
        # If no boxes, entire image is one segment
        cover_segments.append((0, height))
    else:
        # Gap before first box
        if merged_boxes[0][0] > 0:
            cover_segments.append((0, merged_boxes[0][0]))
        # Each box plus gap after it
        for i in range(len(merged_boxes)):
            t_i, b_i = merged_boxes[i]
            cover_segments.append((t_i, b_i))
            if i < len(merged_boxes) - 1:
                next_top = merged_boxes[i+1][0]
                if b_i < next_top:
                    cover_segments.append((b_i, next_top))
        # Gap after last box
        if merged_boxes[-1][1] < height:
            cover_segments.append((merged_boxes[-1][1], height))

    logging.debug(f"Cover segments (pre-subdivision): {cover_segments}")

    # 5. Subdivide large segments if they exceed MAX_CHUNK_HEIGHT
    MAX_CHUNK_HEIGHT = 300  # e.g., 300 px max
    subdivided_segments = []
    for seg_top, seg_bottom in cover_segments:
        seg_height = seg_bottom - seg_top
        if seg_height <= MAX_CHUNK_HEIGHT:
            subdivided_segments.append((seg_top, seg_bottom))
        else:
            # break it down in multiples of 300 px
            start = seg_top
            while start < seg_bottom:
                end = min(start + MAX_CHUNK_HEIGHT, seg_bottom)
                subdivided_segments.append((start, end))
                start = end

    logging.debug(f"Subdivided segments: {subdivided_segments}")

    # 6. Crop each segment with a small overlap
    chunk_overlap = 5
    chunks = []
    for seg_top, seg_bottom in subdivided_segments:
        final_bottom = min(seg_bottom + chunk_overlap, height)
        roi = deskewed_img.crop((0, seg_top, width, final_bottom))
        text_in_chunk = pytesseract.image_to_string(roi).strip()
        chunks.append({
            "image": roi,
            "top": seg_top,
            "bottom": final_bottom,
            "text_len": len(text_in_chunk)
        })

    # 7. If no chunks, fallback to entire image
    if not chunks:
        logging.debug("No segments created, fallback to entire image.")
        text_in_image = pytesseract.image_to_string(deskewed_img).strip()
        return [{
            "image": deskewed_img,
            "top": 0,
            "bottom": height,
            "text_len": len(text_in_image)
        }]

    logging.debug(f"Created {len(chunks)} chunk(s) total.")
    return chunks

def process_image(file_path: Path, out_path: Path) -> dict:
    """
    OCR-based line chunking, returning chunked images as outputs.
    Includes debug info in the 'details' about each chunk's bounding box & text length.
    """
    img = Image.open(file_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    chunk_dicts = adaptive_chunk_image(img)

    # Create relative path
    if 'documents' in file_path.parts:
        rel_path = Path(*file_path.parts[file_path.parts.index('documents')+1:])
    else:
        rel_path = file_path.name
    
    # Create chunks folder
    chunks_folder = out_path.parent / f"{out_path.stem}_chunks"
    chunks_folder.mkdir(parents=True, exist_ok=True)
    
    chunk_paths = []
    chunk_debug_info = []
    for i, chunk_data in enumerate(chunk_dicts):
        roi = chunk_data["image"]
        top_bb = chunk_data["top"]
        bottom_bb = chunk_data["bottom"]
        text_len = chunk_data["text_len"]

        chunk_filename = f"{out_path.stem}_chunk_{i}.jpg"
        out_chunk_path = chunks_folder / chunk_filename
        roi.save(out_chunk_path, "JPEG", quality=100)
        
        chunk_rel_path = rel_path.parent / f"{rel_path.stem}_chunks" / chunk_filename
        chunk_paths.append(str(chunk_rel_path))

        chunk_debug_info.append({
            "index": i,
            "file_path": str(chunk_rel_path),
            "bounding_box": [top_bb, bottom_bb],
            "text_len": text_len
        })

    return {
        "outputs": chunk_paths,
        "source": str(rel_path),
        "details": {
            "num_chunks": len(chunk_dicts),
            "chunks": chunk_debug_info
        }
    }

def process_document(file_path: str, output_folder: Path) -> dict:
    """
    Integrate with process_file utility, returning manifest-friendly output.
    """
    file_path = Path(file_path)
    def process_fn(f: str, o: Path) -> dict:
        return process_image(Path(f), o)
    return process_file(
        file_path=str(file_path),
        output_folder=output_folder,
        process_fn=process_fn,
        file_types={
            '.jpg': process_fn,
            '.jpeg': process_fn,
            '.png': process_fn
        }
    )

def chunk(
    source_folder: Path = typer.Argument(..., help="Source folder containing images"),
    source_manifest: Path = typer.Argument(..., help="Manifest file"),
    output_folder: Path = typer.Argument(..., help="Output folder for chunked images")
):
    """
    Batch chunking CLI that processes background-removed PNGs.
    Uses source paths from manifest to locate original files.
    """
    processor = BatchProcessor(
        input_manifest=source_manifest,
        output_folder=output_folder,
        process_name="chunk",
        base_folder=source_folder / "documents",  # Add /documents to base folder path
        processor_fn=lambda f, o: process_document(f, o),
        use_source=False
    )
    processor.process()

if __name__ == "__main__":
    typer.run(chunk)