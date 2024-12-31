from pathlib import Path
from datetime import datetime
from typing import Callable, Any
from rich.console import Console

console = Console()

def process_file(
    file_path: str,
    output_folder: Path,
    process_fn: Callable[[Path, Path], Any],
    file_types: dict = None
) -> dict:
    """Generic file processor with robust error handling"""
    file_path = Path(file_path)
    
    # Extract just the relative path without project structure
    parts = file_path.parts
    if 'documents' in parts:
        # Get everything after 'documents'
        rel_path = Path(*parts[parts.index('documents') + 1:])
    else:
        # Fallback to just the filename if no 'documents' in path
        rel_path = file_path.name
    
    # Force .jpg extension for output path
    rel_path = rel_path.with_suffix('.jpg')
    # Create output path preserving structure
    out_path = output_folder / "documents" / rel_path
    
    manifest_entry = {
        "source": str(file_path.relative_to(file_path.anchor)),  # Keep original source path
        "outputs": [str(out_path.relative_to(output_folder))],  # Store relative output path with .jpg
        "processed_at": datetime.now().isoformat(),
        "success": False,
        "details": {}
    }
    
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if file_types and file_path.suffix.lower() not in file_types:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Always return manifest entry if file exists, but mark as skipped
        if out_path.exists():
            manifest_entry.update({
                "success": True,
                "skipped": True
            })
            return manifest_entry
            
        # Process only if file doesn't exist
        result = process_fn(file_path, out_path)
        if isinstance(result, dict):
            manifest_entry.update(result)
        manifest_entry["success"] = True
            
        return manifest_entry
        
    except Exception as e:
        console.print(f"[red]Error processing {file_path}: {str(e)}")
        manifest_entry["error"] = f"{type(e).__name__}: {str(e)}"
        return manifest_entry
