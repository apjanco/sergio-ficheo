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
from io import BytesIO
import os

from utils.batch import BatchProcessor
from utils.processor import process_file

console = Console()
logging.basicConfig(level=logging.INFO, format='%(message)s')

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
    # Convert PIL to OpenCV, keeping original for final output
    original_img = pil_img.copy()
    cv_img = np.array(pil_img.convert('L'))  # grayscale for processing only
    # Threshold and invert to get text as white on black for better contour detection
    _, thresh = cv2.threshold(cv_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return original_img  # No contours => can't deskew

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
        return original_img  # Return original colored image

    (h, w) = cv_img.shape
    center = (w // 2, h // 2)
    # Compute rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Rotate
    rotated = original_img.rotate(angle, resample=Image.BICUBIC, expand=False)
    return rotated

def get_text_baseline_angle(img: Image.Image) -> float:
    """Calculate text baseline angle using Tesseract word-level bounding boxes."""
    try:
        # Ensure image is in correct mode for Tesseract
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Use BytesIO to pass image to Tesseract
        with BytesIO() as bio:
            # Save as PNG to preserve quality
            img.save(bio, format='PNG')
            bio.seek(0)
            # Convert to numpy array for Tesseract
            img_array = np.array(Image.open(bio))
            data = pytesseract.image_to_data(img_array, output_type=pytesseract.Output.DICT)
        
        points = []
        for i in range(len(data['level'])):
            if data['conf'][i] > 30 and data['text'][i].strip():
                x = data['left'][i] + data['width'][i]/2
                y = data['top'][i] + data['height'][i]
                points.append((x, y))
        
        if len(points) < 2:
            return 0.0  # Return 0 if not enough points
        
        x_coords, y_coords = zip(*points)
        coeffs = np.polyfit(x_coords, y_coords, deg=1)
        angle = np.degrees(np.arctan(coeffs[0]))
        
        # Limit correction to small angles
        return max(min(angle, 2.0), -2.0)
    
    except Exception as e:
        logging.warning(f"Error in baseline angle detection: {e}")
        return 0.0  # Return 0 as safe default

def calculate_average_baseline(chunks):
    """Calculate average baseline angle from chunks with substantial text."""
    angles = []
    for chunk in chunks:
        if chunk["text_len"] > 50:  # Only use chunks with significant text
            angle = get_text_baseline_angle(chunk["image"])
            angles.append(angle)  # Always append since get_text_baseline_angle now returns 0.0 on error
    
    if not angles:
        return 0.0
    
    # Remove outliers
    angles = np.array(angles)
    mean = np.mean(angles)
    std = np.std(angles)
    valid_angles = angles[abs(angles - mean) <= 2 * std]
    
    return np.mean(valid_angles) if len(valid_angles) > 0 else 0.0

def deskew_chunk(chunk: dict) -> dict:
    """
    Deskew a chunk based on its text baseline.
    Rotates opposite to the detected angle to straighten text.
    """
    img = chunk["image"]
    angle = get_text_baseline_angle(img)
    
    if abs(angle) < 0.1:
        return chunk
        
    # Rotate opposite to the detected angle to correct the slope
    center = (img.width/2, img.height/2)
    rotated = img.rotate(-angle, center=center, expand=False, resample=Image.BICUBIC)
    
    chunk["image"] = rotated
    return chunk

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

def merge_thin_empty_chunks(chunks, min_height=50, min_text_ratio=0.1):
    """
    Enhanced merge function that:
    1. Merges empty chunks with neighbors
    2. Joins chunks with very little text
    3. Handles overlaps properly during merging
    """
    if len(chunks) <= 1:
        return chunks

    # First pass: calculate average text density for normalization
    total_height = sum(chunk["bottom"] - chunk["top"] for chunk in chunks)
    avg_text_per_pixel = sum(chunk["text_len"] for chunk in chunks) / total_height if total_height > 0 else 0

    # First merge: Extremely thin chunks (less than min_height/3)
    very_thin_merged = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        height = current["bottom"] - current["top"]
        
        # If chunk is extremely thin, always merge it
        if height < min_height/3:
            # Try to merge with previous chunk first
            if very_thin_merged:
                prev_chunk = very_thin_merged[-1]
                new_height = current["bottom"] - prev_chunk["top"]
                if new_height < min_height * 3:
                    # Create new image and preserve colors
                    new_img = Image.new('RGB', (
                        prev_chunk["image"].width,
                        new_height
                    ), color='white')  # Use white background
                    
                    # Paste using original images
                    new_img.paste(prev_chunk["image"], (0, 0))
                    curr_paste_y = prev_chunk["bottom"] - prev_chunk["top"]
                    new_img.paste(current["image"], (0, curr_paste_y))
                    
                    prev_chunk["image"] = new_img
                    prev_chunk["bottom"] = current["bottom"]
                    prev_chunk["text_len"] += current["text_len"]
                    i += 1
                    continue
            # If couldn't merge with previous, try next chunk
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                new_height = next_chunk["bottom"] - current["top"]
                if new_height < min_height * 3:
                    # Merge with next chunk
                    next_chunk["top"] = current["top"]
                    next_chunk["text_len"] += current["text_len"]
                    next_chunk["image"] = Image.new('RGB', (
                        next_chunk["image"].width,
                        next_chunk["bottom"] - next_chunk["top"]
                    ))
                    i += 1
                    continue
        very_thin_merged.append(current)
        i += 1

    # Second pass: normal thin/empty chunk merging
    merged = []
    i = 0
    while i < len(very_thin_merged):
        current = very_thin_merged[i]
        height = current["bottom"] - current["top"]
        text_density = current["text_len"] / height if height > 0 else 0

        should_merge = (
            height < min_height or  # Too small
            (text_density < avg_text_per_pixel * min_text_ratio and height < min_height * 2)  # Very little text
        )
        
        if should_merge:
            # Try to merge with previous chunk first
            if merged:
                prev_chunk = merged[-1]
                # Calculate overlap
                overlap = max(0, prev_chunk["bottom"] - current["top"])
                new_height = current["bottom"] - prev_chunk["top"]
                
                if new_height < min_height * 3:
                    # Create new image with white background
                    new_img = Image.new('RGB', (
                        prev_chunk["image"].width,
                        new_height
                    ), color='white')
                    
                    # Paste preserving colors
                    new_img.paste(prev_chunk["image"], (0, 0))
                    if overlap > 0:
                        curr_img = current["image"].crop((0, overlap, current["image"].width, current["image"].height))
                        new_img.paste(curr_img, (0, prev_chunk["bottom"] - prev_chunk["top"] - overlap))
                    else:
                        new_img.paste(current["image"], (0, prev_chunk["bottom"] - prev_chunk["top"]))
                    
                    prev_chunk["image"] = new_img
                    prev_chunk["bottom"] = current["bottom"]
                    prev_chunk["text_len"] += current["text_len"]
                    i += 1
                    continue
            
            # If couldn't merge with previous, try next chunk
            if i < len(very_thin_merged) - 1:
                next_chunk = very_thin_merged[i + 1]
                # Calculate overlap
                overlap = max(0, current["bottom"] - next_chunk["top"])
                new_height = next_chunk["bottom"] - current["top"]
                
                if new_height < min_height * 3:
                    new_img = Image.new('RGB', (
                        next_chunk["image"].width,
                        new_height
                    ), color='white')
                    # Paste current chunk
                    new_img.paste(current["image"], (0, 0))
                    # Paste next chunk, skipping the overlapped region
                    next_img = next_chunk["image"].crop((0, overlap, next_chunk["image"].width, next_chunk["image"].height))
                    new_img.paste(next_img, (0, current["bottom"] - current["top"] - overlap))
                    
                    next_chunk["image"] = new_img
                    next_chunk["top"] = current["top"]
                    next_chunk["text_len"] += current["text_len"]
                    i += 1
                    continue
        
        merged.append(current)
        i += 1
    
    return merged

def find_safe_cut_point(img: Image.Image, start: int, end: int, margin: int = 20) -> int:
    """
    Find a safe point to cut the image between text lines.
    Returns the Y coordinate that appears to be between lines.
    """
    height = end - start
    if height <= margin * 2:
        return start + height // 2
    
    # Look at a slice of the image
    slice_img = img.crop((0, start, img.width, end))
    # Convert to grayscale for analysis
    cv_img = np.array(slice_img.convert('L'))
    
    # Get horizontal projection (sum of dark pixels in each row)
    projection = np.sum(cv_img < 128, axis=1)
    
    # Find the row with the fewest dark pixels (likely between lines)
    min_pixels = float('inf')
    best_cut = start + height // 2  # default to middle if no better point found
    
    # Search in the middle region of the slice
    search_start = height // 3
    search_end = 2 * height // 3
    
    for y in range(search_start, search_end):
        # Look at a small window around this point
        window_sum = sum(projection[max(0, y-2):min(len(projection), y+3)])
        if window_sum < min_pixels:
            min_pixels = window_sum
            best_cut = start + y
    
    return best_cut

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
    # Set Tesseract to use in-memory mode if available
    if hasattr(pytesseract, 'set_temp_directory'):
        pytesseract.set_temp_directory(None)  # Use RAM instead of disk

    # Get image dimensions
    width, height = img.size
    
    # Don't chunk if image is relatively small
    if height < 1000:  # Adjust threshold as needed
        text_in_img = pytesseract.image_to_string(img).strip()
        return [{
            "image": img,
            "top": 0,
            "bottom": height,
            "text_len": len(text_in_img)
        }]
        
    # 1. Deskew the image
    deskewed_img = deskew_image(img)
    width, height = deskewed_img.size

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

    # 3. Fallback to connected components if Tesseract found < 3 lines
    fallback_needed = (len(tess_boxes) < 3)
    if fallback_needed:
        cc_lines = get_connected_component_lines(deskewed_img, line_threshold=10)
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

    # 5. Subdivide large segments if they exceed MAX_CHUNK_HEIGHT
    MAX_CHUNK_HEIGHT = 300  # e.g., 300 px max
    subdivided_segments = []
    for seg_top, seg_bottom in cover_segments:
        seg_height = seg_bottom - seg_top
        if seg_height <= MAX_CHUNK_HEIGHT:
            subdivided_segments.append((seg_top, seg_bottom))
        else:
            # break it down, finding safe cut points
            start = seg_top
            while start < seg_bottom:
                if seg_bottom - start <= MAX_CHUNK_HEIGHT:
                    end = seg_bottom
                else:
                    # Find a safe place to cut
                    target_end = start + MAX_CHUNK_HEIGHT
                    end = find_safe_cut_point(deskewed_img, target_end - 40, target_end + 40)
                
                subdivided_segments.append((start, end))
                start = end

    # 6. Crop each segment with a small overlap
    chunk_overlap = 15
    chunks = []
    for i, (seg_top, seg_bottom) in enumerate(subdivided_segments):
        # Only add overlap at the bottom of chunks (except last)
        actual_top = seg_top
        actual_bottom = seg_bottom + (chunk_overlap if i < len(subdivided_segments)-1 else 0)
        
        # Crop from original colored image
        roi = deskewed_img.crop((0, actual_top, width, actual_bottom))
        text_in_chunk = pytesseract.image_to_string(roi.convert('L')).strip()  # Convert to grayscale only for OCR
        chunks.append({
            "image": roi,  # Keep original colors
            "top": actual_top,
            "bottom": actual_bottom,
            "text_len": len(text_in_chunk)
        })
    
    # Calculate average baseline angle from all chunks
    avg_angle = calculate_average_baseline(chunks)
    
    # Apply same rotation to all chunks with text
    deskewed_chunks = []
    for chunk in chunks:
        if chunk["text_len"] > 0:
            center = (chunk["image"].width/2, chunk["image"].height/2)
            chunk["image"] = chunk["image"].rotate(-avg_angle, center=center, 
                                                 expand=False, resample=Image.BICUBIC)
        deskewed_chunks.append(chunk)
    
    # Merge thin empty chunks with neighbors
    chunks = merge_thin_empty_chunks(deskewed_chunks)
    
    return chunks

def process_image(file_path: Path, out_path: Path) -> dict:
    """
    OCR-based line chunking, returning chunked images as outputs.
    Includes debug info in the 'details' about each chunk's bounding box & text length.
    """
    try:
        # More robust image loading
        with open(file_path, 'rb') as f:
            img = Image.open(f)
            # Convert RGBA or palette images to RGB
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            elif img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Load the image fully before closing file
            img.load()
        
        # Simplified console output
        console.print(f"Processing: {file_path.name}", style="dim")
        
        chunk_dicts = adaptive_chunk_image(img)
        
        # Create chunks folder only once
        chunks_folder = out_path.parent / f"{out_path.stem}_chunks"
        chunks_folder.mkdir(parents=True, exist_ok=True)
        
        # Process chunks in memory first
        chunk_paths = []
        chunk_debug_info = []
        
        # Get relative path info and parent image path
        if 'documents' in file_path.parts:
            rel_path = Path(*file_path.parts[file_path.parts.index('documents')+1:])
            parent_image_path = str(file_path)  # Store full path to parent image
        else:
            rel_path = file_path.name
            parent_image_path = str(file_path)
            
        # Batch process chunks to reduce disk operations
        for i, chunk_data in enumerate(chunk_dicts):
            roi = chunk_data["image"]
            chunk_filename = f"{out_path.stem}_chunk_{i}.jpg"
            out_chunk_path = chunks_folder / chunk_filename
            
            # Use high quality JPEG compression to reduce file size while maintaining quality
            roi.save(out_chunk_path, "JPEG", quality=95, optimize=True)
            
            chunk_rel_path = rel_path.parent / f"{rel_path.stem}_chunks" / chunk_filename
            chunk_paths.append(str(chunk_rel_path))
            
            chunk_debug_info.append({
                "index": i,
                "file_path": str(chunk_rel_path),
                "bounding_box": [chunk_data["top"], chunk_data["bottom"]],
                "text_len": chunk_data["text_len"],
                "parent_image": parent_image_path  # Add parent image path to each chunk
            })
            
            # Free up memory
            roi = None
        
        return {
            "outputs": chunk_paths,
            "source": str(rel_path),
            "parent_image": parent_image_path,  # Include in top level
            "details": {
                "num_chunks": len(chunk_dicts),
                "chunks": chunk_debug_info,
                "parent_info": {
                    "path": parent_image_path,
                    "relative_path": str(rel_path)
                }
            }
        }
    except UnidentifiedImageError as e:
        console.print(f"[red]Invalid image format for {file_path}: {e}")
        return {"error": f"Invalid image format: {e}"}
    except Exception as e:
        console.print(f"[red]Error: {file_path.name} - {str(e)}")
        return {"error": str(e)}
    finally:
        # Ensure memory cleanup
        if 'img' in locals():
            img.close()
        img = None
        chunk_dicts = None

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