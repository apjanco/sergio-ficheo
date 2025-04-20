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
    file_path = Path(file_path)  # Ensure file_path is a Path object
    
    # Always preserve the input path structure but remove any 'documents' prefix
    parts = file_path.parts
    if 'documents' in parts:
        rel_path = Path(*parts[parts.index('documents') + 1:])
    else:
        rel_path = file_path
    
    # Keep original file extension
    out_path = output_folder / "documents" / rel_path
    
    manifest_entry = {
        "source": str(rel_path),  # Store just the relative path
        "outputs": [str(rel_path)],  # Also just relative path
        "processed_at": datetime.now().isoformat(),
        "success": False,
        "details": {}
    }
    
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Accept common image formats
        if file_types and file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # For skipped files, keep the expected output path
        if out_path.exists():
            manifest_entry.update({
                "success": True,
                "skipped": True
            })
            return manifest_entry
            
        # Process only if file doesn't exist
        result = process_fn(file_path, out_path)
        if isinstance(result, dict):
            # Clean up paths in result to remove documents/ prefix
            if "outputs" in result:
                cleaned_outputs = []
                for output in result["outputs"]:
                    output_path = Path(output)
                    if "documents" in output_path.parts:
                        output_path = Path(*output_path.parts[output_path.parts.index("documents") + 1:])
                    cleaned_outputs.append(str(output_path))
                result["outputs"] = cleaned_outputs
            manifest_entry.update(result)
        manifest_entry["success"] = True
            
        return manifest_entry
        
    except Exception as e:
        console.print(f"[red]Error processing {file_path}: {str(e)}")
        manifest_entry["error"] = f"{type(e).__name__}: {str(e)}"
        return manifest_entry
