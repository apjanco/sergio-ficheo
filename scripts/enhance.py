import typer
from PIL import Image, ImageEnhance
from pathlib import Path
import numpy as np
import cv2
from utils.batch import BatchProcessor
from utils.processor import process_file
from rich.console import Console
from typing import Literal
import pytesseract
from sklearn.cluster import KMeans
from collections import Counter

DocumentType = Literal['handwritten', 'typescript', 'mixed']
PaperType = Literal['lined', 'plain']
ContentType = Literal['text', 'diagram', 'mixed']

class DocumentAnalyzer:
    def analyze_image(self, img_array: np.ndarray) -> dict:
        """
        Simplified document analysis.
        Adds fallback morphological heuristic if OCR confidence is inconclusive.
        """
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Document type detection
        doc_type = self._detect_document_type(gray)
        
        # Background color (yellowing) analysis
        is_yellowed = self._detect_yellowing(img_array)
        
        return {
            "document_type": doc_type,
            "is_yellowed": is_yellowed
        }
    
    def _detect_document_type(self, gray: np.ndarray) -> str:
        """
        Detect document type using:
        1) OCR confidence (primary)
        2) Fallback morphological stroke-density heuristic if OCR is inconclusive
        """
        # Binarize for OCR
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Attempt OCR in a try/except block to handle Tesseract errors
        try:
            ocr_data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT)
            confidences = [conf for conf in ocr_data['conf'] if conf != -1]
        except pytesseract.TesseractError:
            # If OCR fails entirely, default to morphological fallback
            return self._morphological_heuristic(binary)
        
        if not confidences:
            # If no recognized text, use fallback
            return self._morphological_heuristic(binary)
        
        # Use confidence threshold to determine type
        avg_confidence = sum(confidences) / len(confidences)
        
        if avg_confidence > 60:
            return 'typescript'
        else:
            return 'handwritten'
    
    def _morphological_heuristic(self, binary: np.ndarray) -> str:
        """
        Simple morphological stroke-density approach:
        - If there's a large amount of small connected strokes, assume handwriting
        - Otherwise, assume typescript
        """
        kernel = np.ones((3, 3), np.uint8)
        morph_grad = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
        # Density of 'edges' or strokes
        non_zero = cv2.countNonZero(morph_grad)
        density = (non_zero / (morph_grad.shape[0] * morph_grad.shape[1])) * 100
        
        # Heuristic threshold for stroke density
        return 'handwritten' if density > 0.5 else 'typescript'
    
    def _detect_yellowing(self, img: np.ndarray) -> float:
        """
        Detect yellow cast in document (0 = no yellow cast, 1 = strong yellow cast).
        Consider an extended range and clamp final result to [0,1].
        """
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        _, _, b_channel = cv2.split(lab)
        b_mean = np.mean(b_channel)
        
        # Simple linear scale relative to neutral (128)
        # A slight tweak to the divisor to avoid over-reporting yellow
        raw_yellow = (b_mean - 128) / 30.0
        
        return max(0, min(1, raw_yellow))

class DocumentEnhancer:
    def enhance(self, img: np.ndarray, doc_type: str, is_yellowed: float) -> np.ndarray:
        """
        Core document enhancement logic with gentle color correction
        """
        # Convert to LAB for processing
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # STEP 1: Enhance text contrast using CLAHE
        if doc_type == 'handwritten':
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            l = clahe.apply(l)
            l = cv2.convertScaleAbs(l, alpha=1.1, beta=-5)
        else:
            clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(16, 16))
            l = clahe.apply(l)
        
        # STEP 2: Handle color cast carefully
        if is_yellowed > 0.1:
            # Gentler yellow reduction
            yellow_reduction = min(8, int(3 * is_yellowed))
            b = cv2.subtract(b, yellow_reduction)
            
            # Very subtle a-channel adjustment
            a = cv2.convertScaleAbs(a, alpha=0.98, beta=0)
        
        # Merge corrected channels
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # STEP 3: Sharpen
        gaussian_blur = cv2.GaussianBlur(enhanced_rgb, (0, 0), 3)
        sharpened = cv2.addWeighted(enhanced_rgb, 1.5, gaussian_blur, -0.5, 0)
        
        return sharpened

def enhance_image(image: Image.Image) -> tuple[Image.Image, dict]:
    """Simplified enhancement pipeline"""
    img_array = np.array(image)
    
    # Analyze document
    analyzer = DocumentAnalyzer()
    analysis = analyzer.analyze_image(img_array)
    
    # Enhance document
    enhancer = DocumentEnhancer()
    enhanced = enhancer.enhance(
        img_array,
        analysis['document_type'],
        analysis['is_yellowed']
    )
    
    return Image.fromarray(enhanced), {"analysis": analysis}

def process_image(file_path: Path, out_path: Path) -> dict:
    """Process a single image file for enhancement"""
    img = Image.open(file_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Get source folder structure from input path
    source_dir = Path(file_path).parts[-4:-1]
    
    # Enhance image and get parameters
    enhanced, params = enhance_image(img)
    
    # Save enhanced image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enhanced.save(out_path, "JPEG", quality=100)
    
    # Build output path preserving full source hierarchy
    rel_path = Path(*source_dir) / out_path.name
    
    details = {
        "original_size": list(img.size),
        "enhanced_size": list(enhanced.size),
        "enhancement_params": params
    }
    
    return {
        "outputs": [str(rel_path)],
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
            '.jpg': process_fn,
            '.jpeg': process_fn,
            '.tif': process_fn,
            '.tiff': process_fn,
            '.png': process_fn
        }
    )

def enhance(
    rotated_folder: Path = typer.Argument(..., help="Input rotated images folder"),
    rotated_manifest: Path = typer.Argument(..., help="Input rotated manifest file"),
    enhanced_folder: Path = typer.Argument(..., help="Output folder for enhanced images")
):
    """Enhance image quality of rotated document pages"""
    processor = BatchProcessor(
        input_manifest=rotated_manifest,
        output_folder=enhanced_folder,
        process_name="enhance",
        base_folder=rotated_folder / "documents",  # Add /documents to match rotation's structure
        processor_fn=lambda f, o: process_document(f, o)
    )
    processor.process()

if __name__ == "__main__":
    typer.run(enhance)