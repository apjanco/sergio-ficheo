import typer
from pathlib import Path
from rich.progress import track

def recombine_chunks(
    transcribed_chunks_folder: Path = typer.Argument(..., help="Path to the transcribed chunks", exists=True),
    recombined_folder: Path = typer.Argument(..., help="Output folder for recombined files")
):
    if not recombined_folder.exists():
        recombined_folder.mkdir(parents=True, exist_ok=True)

    subfolders = [f for f in transcribed_chunks_folder.iterdir() if f.is_dir()]
    
    for subfolder in track(subfolders, description="Recombining chunks..."):
        chunk_files = sorted(subfolder.glob("*.md"))
        combined_text = ""
        for chunk_file in chunk_files:
            combined_text += chunk_file.read_text() + "\n"
        
        output_file = recombined_folder / f"{subfolder.name}.md"
        output_file.write_text(combined_text.strip())

if __name__ == "__main__":
    typer.run(recombine_chunks)