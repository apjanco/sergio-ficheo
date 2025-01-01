from pathlib import Path
from typing import List, Dict

def batch_check_files(files_to_check: List[Path], batch_size: int = 100) -> Dict[Path, bool]:
    """Check existence of multiple files in batches to reduce disk seeks"""
    results = {}
    for i in range(0, len(files_to_check), batch_size):
        batch = files_to_check[i:i + batch_size]
        for path in batch:
            results[path] = path.exists()
    return results

def ensure_dirs(path: Path):
    """Ensure all parent directories exist"""
    path.parent.mkdir(parents=True, exist_ok=True)

def get_image_files(folder: Path, patterns: List[str] = ["*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.png"]) -> List[Path]:
    """Get all image files in a folder and subfolders"""
    files = []
    for pattern in patterns:
        files.extend(folder.glob(f"**/{pattern}"))
    return files

def get_skip_files() -> List[str]:
    """Get list of files to skip processing"""
    return ['']

def get_relative_output_path(source_path: str) -> str:
    """Convert input manifest source path to output path"""
    # Preserve full path including parent folders
    return source_path

def reconstruct_input_path(base_folder: Path, source_path: str) -> Path:
    """Build full input path from base folder and manifest source path"""
    # base_folder already includes 'documents'
    return base_folder / source_path
