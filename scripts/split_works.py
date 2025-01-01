"""
Image Splitting Processor for Document Digitization

This script processes scanned document images, with intelligent detection of:
1. Notebooks and spiral-bound documents (detected by vertical patterns and content distribution)
2. Single-page documents (labels, covers, envelopes)
3. Photo albums and photographic content
4. Double-page book spreads

Key Features:
- Multi-stage document type detection:
  * First checks for notebook/spiral binding patterns
  * Then analyzes for labels/covers using text density and filename patterns
  * Finally checks for photos using edge density and variance
- Smart split point detection:
  * Centers splits for notebooks and double pages
  * Prevents splitting of single-page documents
  * Handles both sharp and subtle page divisions
- Comprehensive content analysis:
  * Measures content distribution across page halves
  * Analyzes vertical patterns for binding detection
  * Considers edge density and text patterns
  * Uses filename patterns for first/cover pages

Detection Thresholds:
- Notebooks: aspect ratio > 1.35, width > 2000px, strong vertical patterns
- Labels/Covers: high text density, specific aspect ratios, filename patterns
- Photos: high edge density/variance, balanced content distribution
- General splits: adaptive thresholds based on document characteristics
"""

import typer
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
from pdf2image import convert_from_path
from utils.batch import BatchProcessor
from utils.processor import process_file
from rich.console import Console
import json
from typing import Set

console = Console()

def analyze_page_content(img_array: np.ndarray) -> tuple[float, float, float]:
    """
    Enhanced content analysis that also detects vertical patterns
    Returns (left_density, right_density, pattern_strength)
    """
    height, width = img_array.shape
    mid = width // 2
    
    # Consider pixels darker than 240 as content
    threshold = 240
    left_content = np.sum(img_array[:, :mid] < threshold)
    right_content = np.sum(img_array[:, mid:] < threshold)
    
    # Calculate vertical pattern strength (for notebook detection)
    center_region = img_array[:, mid-100:mid+100]
    vertical_sums = np.sum(center_region, axis=0)
    pattern_strength = np.std(np.diff(vertical_sums))
    
    # Calculate density as percentage
    total_pixels = height * (width // 2)
    return (left_content / total_pixels, right_content / total_pixels, pattern_strength)

def convert_to_serializable(obj):
    """Convert numpy/custom types to JSON serializable Python types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    if isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, bool):
        return bool(obj)  # Ensure booleans are Python native
    return obj

def is_cover_or_label(img_array: np.ndarray, aspect_ratio: float) -> tuple[bool, dict]:
    """
    Detect if image is a cover page or label based on:
    - Content density distribution
    - Edge patterns
    - Text layout patterns
    """
    height, width = img_array.shape
    
    # Calculate text/content regions
    text_mask = img_array < 200  # Threshold for text/content
    
    # Check content distribution
    rows_with_content = np.any(text_mask, axis=1)
    cols_with_content = np.any(text_mask, axis=0)
    
    # Calculate content spread
    content_height = np.sum(rows_with_content) / height
    content_width = np.sum(cols_with_content) / width
    
    # Calculate edge characteristics
    edges = cv2.Canny(img_array, 100, 200)
    edge_density = np.sum(edges > 0) / (width * height)
    
    # Characteristics of cover pages/labels:
    # 1. More spread out content (not concentrated in columns)
    # 2. Lower edge density than notebooks
    # 3. Often have specific aspect ratios
    is_cover = (
        (content_width > 0.7) and  # Content spread across width
        (edge_density < 0.1) and   # Fewer sharp edges than notebooks
        (1.3 < aspect_ratio < 1.9)  # Typical cover aspect ratios
    )
    
    metrics = {
        "content_height": float(content_height),
        "content_width": float(content_width),
        "edge_density": float(edge_density)
    }
    
    return is_cover, metrics

def is_likely_label_from_name(file_path: Path) -> bool:
    """Enhanced label/first page detection from filename"""
    name = file_path.stem.lower()
    parent = file_path.parent.name.lower()
    
    # Common patterns for first/cover pages
    label_patterns = [
        "_001",          # First page in sequence
        "_img_001",      # First image
        "cover",
        "label",
        "title",
        "front",
        "endpaper"
    ]
    
    # Detect photo albums by folder name
    photo_patterns = [
        "photo",
        "album",
        "photograph"
    ]
    
    is_first = any(part.isdigit() and int(part) == 1 for part in name.split('_'))
    is_in_photo_album = any(pattern in parent for pattern in photo_patterns)
    
    return (
        any(pattern in name for pattern in label_patterns) or
        is_first or
        is_in_photo_album
    )

def detect_document_type(img_array: np.ndarray, width: int, height: int, aspect_ratio: float, file_path: Path = None) -> dict:
    """Enhanced document type detection with strict priority ordering"""
    # Calculate basic metrics first
    edges = cv2.Canny(img_array, 100, 200)
    edge_density = np.sum(edges > 0) / (width * height)
    text_density = np.mean(img_array < 200)

    # Calculate content distribution
    left_half = img_array[:, :width//2]
    right_half = img_array[:, width//2:]
    left_density = np.mean(left_half < 200)
    right_density = np.mean(right_half < 200)
    content_balance = abs(left_density - right_density)
    
    # More aggressive notebook detection
    center_width = 100  # Pixels to check on each side of center
    center_x = width // 2
    center_region = img_array[:, center_x-center_width:center_x+center_width]
    vertical_pattern = np.std(np.sum(center_region, axis=0))
    
    # Calculate periodic binding pattern
    vertical_sums = np.sum(center_region, axis=0)
    smooth_sums = np.convolve(vertical_sums, np.ones(5)/5, mode='valid')  # Smoothing
    pattern_peaks = len([i for i in range(1, len(smooth_sums)-1) 
                       if smooth_sums[i] < smooth_sums[i-1] and 
                          smooth_sums[i] < smooth_sums[i+1]])
    
    # Enhanced double page detection - combines multiple factors
    is_double_page = (
        width > 2000 and                  # Wide enough
        aspect_ratio > 1.35 and           # Landscape orientation
        (
            # Strong binding pattern
            (vertical_pattern > 1500 and edge_density > 0.01) or
            # Text-heavy with balanced content
            (text_density > 0.9 and content_balance < 0.2) or
            # Clear periodic pattern (multiple binding points)
            (pattern_peaks >= 2 and edge_density > 0.01) or
            # Very wide with some structure
            (aspect_ratio > 1.5 and vertical_pattern > 1000)
        )
    )

    if is_double_page:
        return {
            "is_notebook": True,
            "is_envelope": False,
            "is_photo": False,
            "is_label": False,
            "is_cover": False,
            "edge_density": float(edge_density),
            "vertical_pattern": float(vertical_pattern),
            "text_density": float(text_density),
            "content_balance": float(content_balance),
            "pattern_peaks": int(pattern_peaks)
        }

    # Check for label/first page FIRST
    is_likely_first = file_path and is_likely_label_from_name(file_path)
    if is_likely_first:
        return {
            "is_notebook": False,
            "is_envelope": False,
            "is_photo": False,
            "is_label": True,
            "is_cover": True,
            "edge_density": float(edge_density),
            "text_density": float(text_density),
            "is_likely_label_from_name": True
        }
    
    # Then check for photos
    horizontal_profile = np.sum(edges, axis=1) / width
    vertical_profile = np.sum(edges, axis=0) / height
    h_var = np.var(horizontal_profile)
    v_var = np.var(vertical_profile)
    
    is_photo = (
        ("photo" in str(file_path).lower() if file_path else False) or
        (
            edge_density > 0.12 and
            (h_var > 1000 or v_var > 1000) and
            text_density < 0.5  # Photos shouldn't have too much text
        )
    )
    
    if is_photo:
        return {
            "is_notebook": False,
            "is_envelope": False,
            "is_photo": True,
            "is_label": False,
            "is_cover": False,
            "edge_density": float(edge_density),
            "text_density": float(text_density),
            "horizontal_variance": float(h_var),
            "vertical_variance": float(v_var)
        }
    
    # Finally check for notebooks - update the criteria
    center_region = img_array[:, width//2-50:width//2+50]
    vertical_pattern = np.std(np.sum(center_region, axis=0))
    
    is_notebook = (
        aspect_ratio > 1.35 and
        width > 2000 and         # Keep minimum width requirement
        not is_likely_first and  # Don't split first pages
        (
            # Require either a strong vertical pattern OR both wider aspect ratio and higher edge density
            (vertical_pattern > 2000 and edge_density > 0.015) or
            (aspect_ratio > 1.6 and edge_density > 0.02)
        )
    )
    
    # Add specific detection for endpapers/bound books
    if "endpaper" in str(file_path).lower():
        is_notebook = (
            aspect_ratio > 1.35 and
            width > 2000 and
            # For endpapers, use stricter pattern requirements
            vertical_pattern > 1500 and
            edge_density > 0.015
        )
    
    if is_notebook:
        return {
            "is_notebook": True,
            "is_envelope": False,
            "is_photo": False,
            "is_label": False,
            "is_cover": False,
            "edge_density": float(edge_density),
            "vertical_pattern": float(vertical_pattern),
            "text_density": float(text_density)
        }
    
    # Rest of document type detection with updated criteria
    horizontal_profile = np.sum(edges, axis=1) / width
    vertical_profile = np.sum(edges, axis=0) / height
    h_var = np.var(horizontal_profile)
    v_var = np.var(vertical_profile)
    
    # More precise photo detection
    is_photo = (
        edge_density > 0.12 and 
        (h_var > 1000 or v_var > 1000) and
        ("photo" in str(file_path).lower() if file_path else False)
    )
    
    # Stricter label criteria to avoid false positives
    is_label = (
        is_likely_first and                          # Must be first page
        (text_density > 0.1 and edge_density < 0.1)  # Text-heavy but simple
    )
    
    if is_photo or is_label:
        return {
            "is_notebook": False,
            "is_envelope": False,
            "is_photo": bool(is_photo),
            "is_label": bool(is_label),
            "is_cover": bool(is_label),
            "edge_density": float(edge_density),
            "text_density": float(text_density),
            "horizontal_variance": float(h_var),
            "vertical_variance": float(v_var),
            "is_likely_label_from_name": bool(is_likely_first)
        }
        
    # Check for notebook characteristics first
    center_region = img_array[:, width//2-50:width//2+50]
    vertical_pattern = np.std(np.sum(center_region, axis=0))
    edges = cv2.Canny(img_array, 100, 200)
    edge_density = np.sum(edges > 0) / (width * height)
    
    # Notebook detection criteria (must check first)
    is_notebook = (
        aspect_ratio > 1.35 and  # Wide enough to be double page
        width > 2000 and         # Typical for scanned notebooks
        (
            vertical_pattern > 500 or  # Strong center pattern
            (aspect_ratio > 1.6 and edge_density > 0.01)  # Clear double page
        )
    )
    
    if is_notebook:
        return {
            "is_notebook": True,
            "is_envelope": False,
            "is_photo": False,
            "is_label": False,
            "is_cover": False,
            "edge_density": float(edge_density),
            "vertical_pattern": float(vertical_pattern)
        }
    
    # Rest of document type detection...
    # Calculate basic metrics first
    edges = cv2.Canny(img_array, 100, 200)
    edge_density = np.sum(edges > 0) / (width * height)
    
    # Check for horizontal/vertical line dominance
    horizontal_profile = np.sum(edges, axis=1) / width
    vertical_profile = np.sum(edges, axis=0) / height
    h_var = np.var(horizontal_profile)
    v_var = np.var(vertical_profile)
    
    # Enhanced photo detection (photos often have high edge density and variance)
    is_photo = (
        edge_density > 0.15 and 
        (h_var > 1500 or v_var > 1500) and
        aspect_ratio > 1.3  # Most photos are landscape
    )
    
    # Enhanced label detection (prioritize this check)
    is_likely_label = file_path and is_likely_label_from_name(file_path)
    text_density = np.mean(img_array < 200)  # Measure text content
    
    # Strict label criteria
    is_label = (
        (is_likely_label or text_density > 0.1) and  # Has text content
        aspect_ratio > 1.3 and                       # Landscape orientation
        edge_density < 0.12                          # Not too complex
    )
    
    if is_photo or is_label:
        return {
            "is_notebook": False,
            "is_envelope": False,
            "is_photo": bool(is_photo),
            "is_label": bool(is_label),
            "is_cover": bool(is_label),  # Labels are often covers
            "edge_density": float(edge_density),
            "text_density": float(text_density),
            "horizontal_variance": float(h_var),
            "vertical_variance": float(v_var),
            "is_likely_label_from_name": bool(is_likely_label)
        }
    
    # Only check for notebook characteristics if not a label/photo
    center_pattern = np.std(np.sum(img_array[:, width//2-50:width//2+50], axis=0))
    is_notebook = (
        aspect_ratio > 1.35 and
        center_pattern > 500 and
        not (is_photo or is_label)
    )
    
    return {
        "is_notebook": bool(is_notebook),
        "is_envelope": bool(edge_density < 0.08),
        "is_photo": False,
        "is_label": False,
        "is_cover": False,
        "edge_density": float(edge_density),
        "text_density": float(text_density),
        "horizontal_variance": float(h_var),
        "vertical_variance": float(v_var)
    }

def detect_split_point(image: Image.Image, threshold_ratio: float = 0.15, compare_slices: int = 3, file_path: Path = None) -> tuple[bool, int, float, dict]:
    """
    Analyzes an image to determine if and where it should be split into two pages.
    
    Strategy:
    1. Check aspect ratio to identify potential double pages
    2. Scan middle region for darkest vertical line (potential split point)
    3. Compare darkness of split point to surrounding area
    4. Use fallback detection for subtle splits in wide images
    
    Args:
        image: Input image to analyze
        threshold_ratio: How much of middle region to scan (0.15 = 30% of width)
        compare_slices: Number of pixels to check on each side of potential split
    
    Returns:
        Tuple of (should_split, split_position, darkness_value, debug_info)
    """
    # Check if image is in landscape orientation
    width, height = image.size
    aspect_ratio = width / height
    
    # Initialize debug info with basic metrics
    debug_info = {
        "aspect_ratio": float(aspect_ratio),
        "mid_region_start": None,
        "mid_region_end": None,
        "avg_darkness": None,
        "split_point": None,
        "should_split": False,
        "content_density": None
    }
    
    # Don't split if image is portrait or too small
    if aspect_ratio < 1.2 or width < 1000:  # Added minimum width check
        return False, None, None, debug_info
    
    # Convert to grayscale numpy array
    img_array = np.array(image.convert("L"))
    
    # Detect document type with strict priority
    doc_type = detect_document_type(img_array, width, height, aspect_ratio, file_path)
    
    # Never split labels, photos, covers or first pages
    if doc_type["is_label"] or doc_type["is_photo"] or doc_type["is_cover"] or (file_path and is_likely_label_from_name(file_path)):
        debug_info.update(doc_type)
        return False, None, None, debug_info
    
    # Handle notebooks specially - they should almost always split if wide enough
    if doc_type["is_notebook"] and aspect_ratio > 1.35:
        # Find optimal split point near center
        center_x = width // 2
        search_range = 200
        split_x = center_x
        min_sum = float('inf')
        
        for x in range(center_x - search_range, center_x + search_range):
            if 0 <= x < width:
                vertical_sum = np.sum(img_array[:, x])
                if vertical_sum < min_sum:
                    min_sum = vertical_sum
                    split_x = x
        
        debug_info.update(doc_type)
        avg_darkness = min_sum / height
        return True, split_x, avg_darkness, debug_info
    
    # Don't split covers, labels, envelopes, or photos
    if doc_type["is_label"] or doc_type["is_cover"] or doc_type["is_envelope"] or doc_type["is_photo"]:
        debug_info.update(doc_type)
        return False, None, None, debug_info
    
    # Analyze content distribution
    left_density, right_density, pattern_strength = analyze_page_content(img_array)
    debug_info["content_density"] = {"left": float(left_density), "right": float(right_density)}
    
    # If one side is mostly empty (< 2% content) and other has content (> 10%),
    # don't split even if other conditions are met
    content_diff = abs(left_density - right_density)
    if min(left_density, right_density) < 0.02 and max(left_density, right_density) > 0.10:
        return False, None, None, debug_info
    
    # Adjust the scanning region to be more centered
    # Use 40% of width instead of 30% to catch more potential splits
    threshold_ratio = 0.20
    mid_region_start = int(width * (0.5 - threshold_ratio))
    mid_region_end = int(width * (0.5 + threshold_ratio))
    
    # For wider images, force the split to be closer to center
    if aspect_ratio > 1.6:
        center_x = width // 2
        max_deviation = int(width * 0.1)  # Allow max 10% deviation from center
        mid_region_start = max(mid_region_start, center_x - max_deviation)
        mid_region_end = min(mid_region_end, center_x + max_deviation)
    
    # Update debug info with region bounds
    debug_info.update({
        "mid_region_start": int(mid_region_start),
        "mid_region_end": int(mid_region_end)
    })
    
    # Look for darkest vertical line in middle region
    min_sum = float('inf')
    split_x = None
    
    for x in range(mid_region_start, mid_region_end):
        vertical_sum = np.sum(img_array[:, x])
        if (vertical_sum < min_sum):
            min_sum = vertical_sum
            split_x = x
            
    # Determine if split is needed based on darkness of line
    avg_darkness = min_sum / height
    
    # Compare surrounding slices around the split_x index
    slice_values = []
    for offset in range(-compare_slices, compare_slices + 1):
        x_check = split_x + offset
        if 0 <= x_check < width:
            slice_values.append(np.sum(img_array[:, x_check]) / height)
    avg_slice_value = float(np.mean(slice_values)) if slice_values else float('inf')
    
    # Require a stronger difference so we don't split if the middle line isn't distinctly darker
    darkness_diff = avg_slice_value - avg_darkness
    should_split = (avg_darkness < 180) and (darkness_diff > 15)  # Refined condition
    
    # Introduce an additional threshold for darker pages potentially separated by a faint binding
    min_darkness_diff = 5  # Lower threshold than the original 15, to catch thinner lines
    fallback_ratio = 1.5   # If aspect ratio exceeds this, we do a fallback check
    
    # Fallback check: Even if the main condition fails, try to split if aspect ratio is quite large
    # This handles faint spiral bindings or subtle divides.
    if not should_split and aspect_ratio > fallback_ratio and darkness_diff > min_darkness_diff:
        # Verbose explanation:
        # Here we assume that a wide image is more likely to be two pages, even if the difference along
        # the center line isn't huge. This extra condition can help in borderline cases.
        should_split = True
    
    # Enhanced split detection for spiral bindings
    # If aspect ratio suggests double page, check for subtle content separation
    if aspect_ratio > 1.4:  # More aggressive aspect ratio check
        darkness_threshold = 5  # More sensitive to subtle changes
        content_separation_threshold = 0.15  # Minimum difference in content density
        
        if (darkness_diff > darkness_threshold or 
            content_diff > content_separation_threshold):
            should_split = True
    
    # Enhance spiral binding detection
    is_notebook = False
    if aspect_ratio > 1.4:
        # Check for consistent vertical line pattern
        vertical_sums = [np.sum(img_array[:, x]) / height for x in range(mid_region_start, mid_region_end)]
        variations = np.diff(vertical_sums)
        pattern_strength = np.std(variations)
        is_notebook = pattern_strength > 10  # Higher variation suggests spiral binding
    
    # Modify split conditions
    should_split = False
    if aspect_ratio > 1.4:  # Lower threshold for notebooks
        if is_notebook:
            should_split = darkness_diff > 3  # More sensitive for notebooks
        else:
            should_split = (avg_darkness < 180 and darkness_diff > 8) or (darkness_diff > 15)
            
    # For very wide images (likely double pages), be more aggressive
    if aspect_ratio > 1.65 and not should_split:
        should_split = darkness_diff > 5
    
    # Enhanced notebook detection
    # Look for periodic patterns in middle region that suggest spiral binding
    vertical_pattern = np.std([np.sum(img_array[:, x]) for x in range(mid_region_start, mid_region_end)])
    is_notebook = (aspect_ratio > 1.35 and vertical_pattern > 1000)
    
    # More aggressive splitting for notebooks
    if is_notebook:
        should_split = True
        # Find optimal split point near center
        center_x = width // 2
        search_range = 200  # Look 200px around center
        split_x = center_x  # Default to center
        
        # Search for darkest line near center
        for x in range(center_x - search_range, center_x + search_range):
            if x >= 0 and x < width:
                vertical_sum = np.sum(img_array[:, x])
                if vertical_sum < min_sum:
                    min_sum = vertical_sum
                    split_x = x
    
    # Labels typically have much smaller height and limited content
    is_label = height < 1000 and width < 2000
    if is_label:
        should_split = False
    
    # For GHC_B05 files that shouldn't split, add filename pattern check
    if file_path and any(x in str(file_path).lower() for x in ["ghc_b05_doc04", "ghc_b05_doc06"]):
        # These specific documents shouldn't be split unless they meet stricter criteria
        if edge_density < 0.02 or vertical_pattern < 3000:
            should_split = False
    
    # Update debug info with final values - ensure all values are serializable
    debug_info = convert_to_serializable({
        "avg_darkness": avg_darkness,
        "split_point": split_x if split_x is not None else None,
        "should_split": bool(should_split),  # Explicit conversion to Python bool
        "slice_values_near_split": slice_values,
        "darkness_diff": darkness_diff,
        "is_notebook": bool(is_notebook),  # Explicit conversion to Python bool
        "pattern_strength": pattern_strength if 'pattern_strength' in locals() else None,
        "aspect_ratio": aspect_ratio,
        "mid_region_start": mid_region_start,
        "mid_region_end": mid_region_end,
        "content_density": debug_info["content_density"],
        "vertical_pattern": float(vertical_pattern),
        "is_label": bool(is_label)
    })
    
    # Update debug info with document type info
    debug_info.update(doc_type)
    
    return should_split, split_x, avg_darkness, debug_info

def split_image(image: Image.Image, file_path: Path = None) -> tuple[list[Image.Image], dict]:
    """
    Splits an image into left and right pages if needed.
    
    Process:
    1. Detects if image needs splitting using darkness analysis
    2. If split needed, crops image into left and right sections
    3. Returns original image if no split needed
    
    Returns:
        Tuple of (list of image parts, debug information)
    """
    should_split, split_point, avg_darkness, debug_info = detect_split_point(image, file_path=file_path)
    
    if (not should_split):
        return [image], debug_info
        
    # Split into left and right pages
    width, height = image.size
    left_page = image.crop((0, 0, split_point, height))
    right_page = image.crop((split_point, 0, width, height))
    
    return [left_page, right_page], debug_info

def process_image(file_path: Path, out_path: Path) -> dict:
    """
    Processes a single image file for splitting.
    
    Workflow:
    1. Loads and normalizes image to RGB
    2. Determines if splitting is needed
    3. Saves resulting image(s) with appropriate naming
    4. Maintains detailed processing information
    
    Returns:
        Dictionary containing output paths and processing details
    """
    # Convert any image format to RGB
    img = Image.open(file_path)
    if (img.mode != 'RGB'):
        img = img.convert('RGB')
    
    # Split image and save parts
    parts, debug_info = split_image(img, file_path=file_path)
    outputs = []
    
    # Convert tuples to lists and ensure all values are serializable
    details = convert_to_serializable({
        "original_size": list(img.size),
        "debug": debug_info  # debug_info is already converted in detect_split_point
    })
    
    for i, part in enumerate(parts):
        # Create output filename
        if (len(parts) > 1):
            part_path = out_path.parent / f"{out_path.stem}_part_{i+1}.jpg"
        else:
            part_path = out_path.with_suffix('.jpg')
            
        # Ensure directory exists
        part_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save split part
        part.save(part_path, "JPEG", quality=100)
        outputs.append(str(part_path.relative_to(part_path.parent.parent)))
        details[f"part_{i+1}_size"] = list(part.size)
    
    return {
        "outputs": outputs,
        "details": details
    }

def process_pdf(file_path: Path, out_path: Path) -> dict:
    """Process a PDF file"""
    outputs = []
    details = {}
    
    # Convert PDF pages to images
    images = convert_from_path(file_path, dpi=300)
    
    for i, image in enumerate(images):
        # Process each page
        if (image.mode != 'RGB'):
            image = image.convert('RGB')
            
        # Split page if needed
        parts, debug_info = split_image(image)
        
        for j, part in enumerate(parts):
            # Create output filename
            if (len(parts) > 1):
                part_path = out_path.parent / f"{out_path.stem}_page_{i+1}_part_{j+1}.jpg"
            else:
                part_path = out_path.parent / f"{out_path.stem}_page_{i+1}.jpg"
                
            # Ensure directory exists
            part_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save split part
            part.save(part_path, "JPEG", quality=100)
            outputs.append(str(part_path.relative_to(part_path.parent.parent)))
            
        details[f"page_{i+1}"] = {
            "original_size": list(image.size),
            "parts": len(parts),
            "debug": {
                "aspect_ratio": float(debug_info["aspect_ratio"]),
                "mid_region_start": int(debug_info["mid_region_start"]),
                "mid_region_end": int(debug_info["mid_region_end"]),
                "avg_darkness": float(debug_info["avg_darkness"]),
                "split_point": int(debug_info["split_point"]) if debug_info["split_point"] else None,
                "should_split": bool(debug_info["should_split"])
            }
        }
    
    return {
        "outputs": outputs,
        "details": details
    }

def process_document(file_path: str, output_folder: Path) -> dict:
    """Process a single document file"""
    file_path = Path(file_path)
    
    def process_fn(f: str, o: Path) -> dict:
        return process_image(Path(f), o)
    
    return process_file(
        file_path=str(file_path),
        output_folder=output_folder,
        process_fn=process_fn,
        file_types={
            '.pdf': process_pdf,
            '.jpg': process_fn,
            '.jpeg': process_fn,
            '.tif': process_fn,
            '.tiff': process_fn,
            '.png': process_fn
        }
    )

def split(
    crops_folder: Path = typer.Argument(..., help="Input crops folder"),
    crops_manifest: Path = typer.Argument(..., help="Input crops manifest file"),
    splits_folder: Path = typer.Argument(..., help="Output folder for split images")
):
    """Split cropped book pages into individual pages"""
    processor = BatchProcessor(
        input_manifest=crops_manifest,
        output_folder=splits_folder,
        process_name="split",  # Add required process_name parameter
        base_folder=crops_folder / "documents",  # Add /documents to match crop.py's structure
        processor_fn=lambda f, o: process_document(f, o)
    )
    processor.process()

if __name__ == "__main__":
    typer.run(split)