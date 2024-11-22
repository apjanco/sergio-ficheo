import typer
import srsly
from pathlib import Path
from rich.progress import track
import re

app = typer.Typer()

def process_json(
    data_path: Path = typer.Argument(..., help="Path to the cleaned text files", exists=True),
    image_path: Path = typer.Argument(..., help="Path to the adjusted images", exists=True),
    out_file: Path = typer.Argument(..., help="Output path and filename to save the data file")
):
    """
    Process images and text. Outcome is a single JSONL file with a dictionary for each page with the transcribed text.
    """
    def extract_sort_key(filename):
        match = re.search(r'C(\d+)_(\d+)', filename)
        return (int(match.group(1)), int(match.group(2))) if match else (float('inf'), float('inf'))

    md_files = sorted(data_path.glob("**/*.md"), key=lambda x: extract_sort_key(x.stem))  # Process all files in numerical order
    image_files = {img.stem: img for img in image_path.glob("**/*.jpg")}
    
    data = []
    for md_file in track(md_files, description="Processing files..."):
        img_data = {}
        img_name = md_file.stem
        img_data["image"] = image_files.get(img_name, None).name if img_name in image_files else None
        text = md_file.read_text()
        img_data["text"] = text
        data.append(img_data)

    # Ensure the output directory exists
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Ensure the output file is created or erased if it already exists
    if out_file.exists():
        out_file.unlink()
    out_file.touch()

    # Save the processed data to a JSONL file
    srsly.write_jsonl(out_file, data)

app.command()(process_json)

if __name__ == "__main__":
    app()