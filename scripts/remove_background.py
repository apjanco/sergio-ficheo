import typer
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
from typing import Literal

from utils.batch import BatchProcessor
from utils.processor import process_file

class BlackBackgroundRemoverMulti:
    """
    A pipeline that:
    1) Thresholds for black background
    2) Finds all external contours
    3) Keeps a subset of contours based on heuristic (e.g., biggest or center-located)
    4) Morphologically refine, blur => alpha
    5) Crop final image to bounding box of alpha
    """

    def remove_background(self, img_array: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Steps:
        A) Check black coverage to skip if almost no black background.
        B) Threshold for black => doc=white
        C) Find contours, keep the ones we want (heuristics).
        D) Combine kept contours into a mask
        E) Morph open/close, blur => partial transparency
        F) Crop to bounding box of alpha
        G) Return final RGBA + debug params
        """

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        image_area = h * w

        # A) Check black coverage (optional):
        BLACK_THRESH = 80
        black_pixels = np.count_nonzero(gray < BLACK_THRESH)
        black_ratio = black_pixels / float(image_area)
        black_coverage_cutoff = 0.01  # if <1% black, skip removal
        if black_ratio < black_coverage_cutoff:
            # skip => fully opaque
            rgba_skip = cv2.cvtColor(img_array, cv2.COLOR_RGB2RGBA)
            rgba_skip[:, :, 3] = 255
            return rgba_skip, {
                "method": "skipped_almost_no_black",
                "black_ratio": black_ratio,
                "black_coverage_cutoff": black_coverage_cutoff,
                "black_thresh": BLACK_THRESH
            }

        # B) Threshold with a more aggressive black limit
        _, bin_mask = cv2.threshold(gray, BLACK_THRESH, 255, cv2.THRESH_BINARY)
        # bin_mask: 255 => doc/foreground, 0 => black background

        # C) Find external contours
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # fallback => fully opaque
            rgba_fallback = cv2.cvtColor(img_array, cv2.COLOR_RGB2RGBA)
            rgba_fallback[:, :, 3] = 255
            return rgba_fallback, {
                "method": "no_contour_found_fallback",
                "black_thresh": BLACK_THRESH
            }

        # We must decide which contours to keep. Example logic:
        #   1. Compute area of each contour
        #   2. Possibly compute bounding box or distance from center
        #   3. Keep the largest contour, plus any that are "close to center" or have decent area
        # This is entirely up to your data. We'll show a simple approach.

        # compute areas
        contour_areas = [cv2.contourArea(c) for c in contours]
        total_foreground_area = sum(contour_areas)

        # Sort by area descending
        sorted_by_area = sorted(zip(contours, contour_areas), key=lambda x: x[1], reverse=True)

        # Example heuristic:
        #   - Always keep the largest contour if it's at least 20% of total foreground
        #   - Keep any other contour that is at least 20% the size of the largest
        #   - Possibly keep a contour that is near the center if it's not too tiny
        # NOTE: Tweak these numbers for your scenario

        largest_contour_area = sorted_by_area[0][1]
        keep_contours = []

        # define some thresholds
        MIN_FRAC_OF_FOREGROUND = 0.2  # must be >= 20% of the total foreground area
        MIN_FRAC_OF_LARGEST = 0.2     # must be >= 20% of the largest contour's area
        # or consider bounding-box near center

        for contour, area in sorted_by_area:
            if area == 0:
                continue
            # fraction of total
            frac_of_foreground = area / total_foreground_area
            # fraction of largest contour
            frac_of_largest = area / largest_contour_area

            # check center-dist if needed
            # bounding box
            x, y, cw, ch = cv2.boundingRect(contour)
            center_x = x + cw/2
            center_y = y + ch/2
            # maybe check distance from image center
            img_center_x = w / 2
            img_center_y = h / 2
            dist_x = abs(center_x - img_center_x)
            dist_y = abs(center_y - img_center_y)

            # For example: keep if it’s big, or if it's near center
            # (You can refine "near center" with a threshold, or keep only largest + 2nd largest, etc.)
            if (frac_of_foreground > MIN_FRAC_OF_FOREGROUND) or \
               (frac_of_largest > MIN_FRAC_OF_LARGEST) or \
               (dist_x < w*0.1 and dist_y < h*0.1):
                keep_contours.append(contour)

        # If "keep_contours" ends up empty, fallback to just keep the largest
        if not keep_contours:
            keep_contours = [sorted_by_area[0][0]]

        # D) Combine kept contours into a single mask
        doc_mask = np.zeros_like(bin_mask, dtype=np.uint8)
        for c in keep_contours:
            cv2.drawContours(doc_mask, [c], -1, color=255, thickness=-1)

        # E) Morphological open/close, blur => partial transparency
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        doc_mask_opened = cv2.morphologyEx(doc_mask, cv2.MORPH_OPEN, kernel_open)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        doc_mask_closed = cv2.morphologyEx(doc_mask_opened, cv2.MORPH_CLOSE, kernel_close)

        blurred_mask = cv2.GaussianBlur(doc_mask_closed, (21, 21), 0)
        blurred_mask = cv2.normalize(blurred_mask, None, 0, 255, cv2.NORM_MINMAX)

        alpha_flt = blurred_mask.astype(np.float32) / 255.0
        alpha_flt *= 0.95
        final_mask = (alpha_flt * 255).astype(np.uint8)

        # Create RGBA
        rgba = cv2.cvtColor(img_array, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = final_mask

        # F) Crop to bounding box of non-zero alpha (so it's as tight as possible)
        #    Find all coords where alpha > 0
        ys, xs = np.nonzero(final_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            # if everything got removed, fallback
            return rgba, {
                "method": "empty_alpha_fallback",
                "black_thresh": BLACK_THRESH
            }

        minx, maxx = xs.min(), xs.max()
        miny, maxy = ys.min(), ys.max()

        # Crop
        cropped_rgba = rgba[miny:maxy+1, minx:maxx+1, :]

        # debug params
        params = {
            "method": "multi_obj_black_bg_removal",
            "black_thresh": BLACK_THRESH,
            "black_ratio": black_ratio,
            "total_foreground_area": total_foreground_area,
            "largest_contour_area": largest_contour_area,
            "num_contours_found": len(contours),
            "num_contours_kept": len(keep_contours),
            "crop_bbox": [int(minx), int(miny), int(maxx), int(maxy)]
        }
        return cropped_rgba, params


def remove_background_from_image(image: Image.Image) -> tuple[Image.Image, dict]:
    """
    Public pipeline: remove black background, keep multiple objects by heuristic, and crop.
    """
    img_array = np.array(image)
    remover = BlackBackgroundRemoverMulti()
    cropped_rgba, analysis_params = remover.remove_background(img_array)

    # Convert back to PIL
    out_pil = Image.fromarray(cropped_rgba, mode='RGBA')
    return out_pil, {"analysis": analysis_params}


def process_image(file_path: Path, out_path: Path) -> dict:
    """
    Process a single image file with the multi-object black background approach, then crop.
    """
    img = Image.open(file_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    bg_removed, params = remove_background_from_image(img)

    # Get source folder structure from input path
    source_dir = Path(*file_path.parts[file_path.parts.index('documents')+1:])
    
    # Save as PNG and ensure .png extension
    out_path = out_path.with_suffix('.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bg_removed.save(out_path, "PNG")  # Remove quality parameter as it's not used for PNG
    
    # Ensure output path in manifest also has .png extension
    rel_path = source_dir.with_suffix('.png')

    details = {
        "original_size": list(img.size),
        "bg_removed_size": list(bg_removed.size),
        "bg_removal_params": params
    }
    return {
        "outputs": [str(rel_path)],
        "details": details
    }


def process_document(file_path: str, output_folder: Path) -> dict:
    """
    Uses your existing `process_file` from utils.processor.
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
            '.tif': process_fn,
            '.tiff': process_fn,
            '.png': process_fn
        }
    )


def remove_background(
    rotated_folder: Path = typer.Argument(..., help="Folder with input images"),
    rotated_manifest: Path = typer.Argument(..., help="Manifest file"),
    bgremoved_folder: Path = typer.Argument(..., help="Output folder")
):
    """
    CLI for multi-object black/dark background removal with bounding box crop.
    """
    processor = BatchProcessor(
        input_manifest=rotated_manifest,
        output_folder=bgremoved_folder,
        process_name="remove_multi_obj_black_bg",
        base_folder=rotated_folder / "documents",
        processor_fn=lambda f, o: process_document(f, o)
    )
    processor.process()


if __name__ == "__main__":
    typer.run(remove_background)