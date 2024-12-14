import typer
from pathlib import Path
from rich.progress import track
from rich.console import Console
from PIL import Image, UnidentifiedImageError
import re  # Add this import statement

console = Console()

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

def check_split_and_copy(
    cropped_dir: Path = typer.Argument(..., help="Directory containing cropped images"),
    split_dir: Path = typer.Argument(..., help="Directory containing split images")
):
    images = list(cropped_dir.glob("**/*.jpg"))
    images = sorted(images, key=natural_sort_key)

    for image in track(images, description="Checking and copying images..."):
        try:
            relative_path = image.relative_to(cropped_dir)
            output_path = split_dir / relative_path.parent / (relative_path.stem + ".jpg")
            left_path = split_dir / relative_path.parent / (relative_path.stem + "_left.jpg")
            right_path = split_dir / relative_path.parent / (relative_path.stem + "_right.jpg")

            if not output_path.exists() and not (left_path.exists() or right_path.exists()):
                console.print(f"Copying original image: {image}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                img = Image.open(image)
                img.save(output_path)
        except (UnidentifiedImageError, ValueError) as e:
            console.print(f"Skipping {image.name}: {e}")

if __name__ == "__main__":
    typer.run(check_split_and_copy)
