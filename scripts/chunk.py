import typer
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from rich.progress import track
from rich.console import Console
import re

console = Console()

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

def chunk_image(image: Image.Image, num_chunks: int, overlap_percentage: int) -> list:
    """Chunk the image into vertical sub-images with overlap."""
    width, height = image.size
    chunk_height = height // num_chunks
    overlap = int(chunk_height * (overlap_percentage / 100))
    chunks = []
    for i in range(num_chunks):
        top = i * chunk_height - (i * overlap)
        bottom = top + chunk_height + overlap
        if bottom > height:
            bottom = height
        chunk = image.crop((0, top, width, bottom))
        chunks.append(chunk)
    return chunks

def chunk_images(
    collection_path: Path = typer.Argument(..., help="Path to the adjusted images", exists=True),
    out_dir: Path = typer.Argument(..., help="Output directory to save the chunked images"),
    num_chunks: int = typer.Option(4, help="Number of chunks to split the image into"),
    overlap_percentage: int = typer.Option(10, help="Percentage of overlap between chunks")
):
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for ext in ["*.jpg"]:
        images.extend(collection_path.glob(f"**/{ext}"))
    images = sorted(images, key=natural_sort_key)

    for image in track(images, description="Chunking images..."):
        try:
            relative_path = image.relative_to(collection_path)
            output_path = out_dir / relative_path.parent / relative_path.stem
            output_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory structure exists
            if any((output_path / f"{relative_path.stem}_chunk_{i}.jpg").exists() for i in range(num_chunks)):
                console.print(f"Skipping existing image chunks: {relative_path}")
                continue
            img = Image.open(image)
            chunks = chunk_image(img, num_chunks, overlap_percentage)
            for i, chunk in enumerate(chunks):
                chunk.save(output_path / f"{relative_path.stem}_chunk_{i}.jpg")
        except (UnidentifiedImageError, ValueError) as e:
            console.print(f"Skipping {image.name}: {e}")

if __name__ == "__main__":
    typer.run(chunk_images)