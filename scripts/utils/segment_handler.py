from pathlib import Path
from PIL import Image
from typing import Dict, Union
import shutil
import os
import json
import tempfile
from rich.console import Console  # Add this import

console = Console()

class SegmentHandler:
    """Handles loading, saving, and path management for image segments"""
    
    @staticmethod
    def exists(path: Union[str, Path], base_folder: Path = None) -> bool:
        """Check if segment exists"""
        if base_folder:
            full_path = base_folder / Path(path)
        else:
            full_path = Path(path)
        return full_path.exists()

    @staticmethod
    def check_segment_exists(source_path: Path, segment_index: int) -> bool:
        """Check if a specific segment exists"""
        paths = SegmentHandler.get_segment_paths(source_path)
        segment_name = SegmentHandler.make_segment_name(source_path.stem, segment_index)
        segment_path = paths["segments_folder"] / segment_name
        return segment_path.exists()

    @staticmethod
    def load_segment(segment_path: Union[str, Path], base_folder: Path = None) -> Image.Image:
        """Load a segment image with proper path resolution"""
        try:
            if base_folder:
                full_path = base_folder / Path(segment_path)
            else:
                full_path = Path(segment_path)
                
            if not full_path.exists():
                raise FileNotFoundError(f"Segment not found: {full_path}")
                
            img = Image.open(full_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
            
        except Exception as e:
            raise Exception(f"Error loading segment {segment_path}: {str(e)}")

    @staticmethod
    def save_segment_output(
        output: str,
        out_path: Path,
        extension: str = '.md'
    ) -> Dict:
        """Save segment output and return manifest entry"""
        out_path = out_path.with_suffix(extension)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(output)
            
        # Get relative path from documents/ onwards
        parts = out_path.parts
        if 'documents' in parts:
            rel_path = Path(*parts[parts.index('documents')+1:])
        else:
            rel_path = out_path.name
            
        return {
            "outputs": [str(rel_path)],
            "source": str(rel_path)
        }

    @staticmethod
    def get_segment_paths(source_path: Path) -> Dict:
        """Get segment folder and file paths"""
        if source_path.suffix:  # If it's a file
            segments_folder = source_path.parent / f"{source_path.stem}_segments"
        else:  # If it's a directory
            segments_folder = source_path / f"{source_path.name}_segments"
            
        # Get parent path without documents prefix
        parts = source_path.parts
        if 'documents' in parts:
            parent_path = Path(*parts[parts.index('documents')+1:])
        else:
            parent_path = source_path
            
        return {
            "segments_folder": segments_folder,
            "parent_path": parent_path
        }

    @staticmethod
    def get_relative_path(file_path: Path) -> Path:
        """Get relative path from documents onwards"""
        parts = file_path.parts
        if 'documents' in parts:
            return Path(*parts[parts.index('documents')+1:])
        return file_path

    @staticmethod 
    def make_segment_name(base_name: str, segment_index: int) -> str:
        """Create standardized segment filename"""
        stem = Path(base_name).stem
        return f"{stem}_segment_{segment_index}.jpg"

    @staticmethod
    def is_processing(folder: Path) -> bool:
        """Check if a folder is currently being processed"""
        lock_file = folder / ".processing"
        return lock_file.exists()

    @staticmethod
    def start_processing(folder: Path) -> None:
        """Mark folder as being processed"""
        # Ensure folder exists before creating lock file
        folder.mkdir(parents=True, exist_ok=True)
        lock_file = folder / ".processing"
        lock_file.touch()

    @staticmethod
    def finish_processing(folder: Path) -> None:
        """Mark folder as finished processing"""
        lock_file = folder / ".processing"
        if lock_file.exists():
            lock_file.unlink()

    @staticmethod
    def is_complete(folder: Path) -> bool:
        """Check if a folder was completely processed"""
        done_file = folder / ".done"
        return done_file.exists() and not SegmentHandler.is_processing(folder)

    @staticmethod
    def mark_complete(folder: Path, metadata: dict = None) -> None:
        """Mark folder as completely processed"""
        done_file = folder / ".done"
        if metadata:
            with open(done_file, 'w') as f:
                json.dump(metadata, f)
        else:
            done_file.touch()

    @staticmethod
    def process_safely(folder: Path, process_fn, metadata: dict = None):
        """Process a folder with safety checks"""
        try:
            # Create folder first
            folder.mkdir(parents=True, exist_ok=True)

            # If already complete and not processing, skip
            if SegmentHandler.is_complete(folder):
                console.print(f"[yellow]Skipping completed folder: {folder}")
                return True

            # If interrupted mid-processing, clean up folder contents but keep folder
            if SegmentHandler.is_processing(folder):
                console.print(f"[yellow]Cleaning up interrupted processing: {folder}")
                for item in folder.glob("*"):
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)

            # Start processing
            SegmentHandler.start_processing(folder)
            try:
                result = process_fn()
                SegmentHandler.mark_complete(folder, metadata)
                return result
            finally:
                SegmentHandler.finish_processing(folder)
        except Exception as e:
            console.print(f"[red]Error in process_safely: {e}")
            # Don't delete folder on error, but remove processing flag
            if folder.exists() and SegmentHandler.is_processing(folder):
                SegmentHandler.finish_processing(folder)
            raise
