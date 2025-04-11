import typer
import srsly
from pathlib import Path
import os
import re
import urllib.parse  # Add this for URL encoding/decoding

def natural_sort_key(s: str):
    """Sort strings alphanumerically like Finder."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def build_documents_manifest(
    documents_dir: Path = typer.Argument(..., help="Directory to scan for files and folders"),
    documents_manifest: Path = typer.Argument(..., help="Output file path (.jsonl)")
):
    """
    Recursively scan the given documents directory and create a JSONL file listing
    all files and subfolders (relative paths only), sorted alphanumerically.
    
    Required, because on a spinning disk, and lots of files, thigns were too slow.
    """
    
    # Safely handle the path with special characters and spaces
    documents_dir = Path(os.path.expanduser(str(documents_dir))).resolve()
    documents_manifest = Path(os.path.expanduser(str(documents_manifest))).resolve()
    
    # Convert to .jsonl extension and ensure it is in the manifests directory
    documents_manifest = documents_manifest.with_suffix('.jsonl')
    
    # Ensure the directory for the manifest file exists
    documents_manifest.parent.mkdir(parents=True, exist_ok=True)
    
    entries = []
    
    # Recursively list files & folders
    for root, dirs, files in os.walk(documents_dir):
        root_path = Path(root)
        for d in dirs:
            rel_path = root_path.joinpath(d).relative_to(documents_dir)
            entries.append({"path": str(rel_path), "type": "directory"})
        for f in files:
            if f.lower().endswith(('.pdf', '.jpg', '.jpeg', '.tif', '.tiff', '.png', '.jxl')):
                file_path = root_path.joinpath(f)
                rel_path = file_path.relative_to(documents_dir)
                entries.append({
                    "path": str(rel_path),
                    "type": "file",
                    "mtime": os.path.getmtime(file_path),
                    "size": os.path.getsize(file_path)
                })
    
    # Sort all entries by alphanumeric order of their 'path'
    entries.sort(key=lambda e: natural_sort_key(e["path"]))
    
    # Write the sorted entries to the JSONL file
    srsly.write_jsonl(documents_manifest, entries)
    print(f"Saved {len(entries)} entries to {documents_manifest}")

if __name__ == "__main__":
    typer.run(build_documents_manifest)