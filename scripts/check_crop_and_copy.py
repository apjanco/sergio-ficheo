import typer
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from utils.batch import BatchProcessor
from utils.processor import process_file
from rich.console import Console
import srsly
import shutil

console = Console()

def convert_to_jpg(file_path: Path, out_path: Path) -> dict:
    """Convert any supported file to JPG"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_path.suffix.lower() == '.pdf':
        # For PDFs, we need to convert and save all pages
        try:
            images = convert_from_path(str(file_path), dpi=300)
            outputs = []
            details = {}
            
            # Process each page
            for i, image in enumerate(images):
                # Force jpg extension and consistent naming
                page_path = out_path.parent / f"{out_path.stem}_page_{i + 1}.jpg"
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                page_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(page_path, "JPEG", quality=100)
                outputs.append(str(page_path.relative_to(page_path.parent.parent)))
                details[f"page_{i + 1}"] = {"size": image.size}
            
            return {
                "outputs": outputs,
                "details": details
            }
        except Exception as e:
            console.print(f"[red]PDF conversion error for {file_path}: {e}")
            raise
    else:
        # Handle single image files
        img = Image.open(file_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        out_path = out_path.with_suffix('.jpg')
        img.save(out_path, "JPEG", quality=100)
        return {
            "outputs": [str(out_path.relative_to(out_path.parent.parent))],
            "details": {"size": img.size}
        }

def check_crop_and_copy(
    source_folder: Path = typer.Argument(..., help="Source folder with original files"),
    target_folder: Path = typer.Argument(..., help="Target folder to check/copy to")
):
    """Check missing files from manifest and recreate them"""
    manifest_path = target_folder / "crop_manifest.jsonl"
    if not manifest_path.exists():
        raise typer.Exit("Crop manifest not found")
    
    # Create list of files to process from manifest
    temp_manifest = target_folder / 'check_manifest.jsonl'
    with open(temp_manifest, 'w') as f:
        for entry in srsly.read_jsonl(manifest_path):
            if not entry.get("source"):
                continue
                
            source_path = Path(entry["source"])
            original_file = source_folder / source_path
            
            # Check if any output is missing
            outputs = entry.get("outputs", [])
            if not outputs:
                continue
                
            any_missing = False
            for output in outputs:
                if not (target_folder / "documents" / output).exists():
                    any_missing = True
                    break
            
            # If any output is missing and source exists, add to process list
            if any_missing and original_file.exists():
                f.write(srsly.json_dumps({
                    "type": "file",
                    "path": str(source_path)
                }) + '\n')
    
    # Process missing files
    processor = BatchProcessor(
        input_manifest=temp_manifest,
        output_folder=target_folder,
        process_name="check_and_copy",
        processor_fn=process_document,
        base_folder=source_folder
    )
    processor.process()
    temp_manifest.unlink()

def process_document(file_path: str, output_folder: Path) -> dict:
    """Process a single document file"""
    supported_types = {
        '.pdf': convert_to_jpg,
        '.jpg': convert_to_jpg,
        '.jpeg': convert_to_jpg,
        '.tif': convert_to_jpg,
        '.tiff': convert_to_jpg,
        '.png': convert_to_jpg
    }
    
    return process_file(
        file_path=file_path,
        output_folder=output_folder,
        process_fn=convert_to_jpg,
        file_types=supported_types
    )

if __name__ == "__main__":
    typer.run(check_crop_and_copy)
