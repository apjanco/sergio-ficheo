import typer
from rich import print
from rich.progress import track

def clean(
    collection_path: Path = typer.Argument(..., help="Path to the collections", exists=True),
    site_dir: Path = typer.Argument(..., help="Output _site directory")
):
    """
    Clean the site directory and remove all files and directories.
    """
    print(f"[green]Cleaning {site_dir}")
    if site_dir.exists():
        for item in track(site_dir.iterdir(), description="Removing files"):
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    print("[green]Done")

if __name__ == "__main__":
    typer.run(clean)