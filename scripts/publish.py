import typer 
import srsly
import markdown
import subprocess
from pathlib import Path
from rich.progress import track
import shutil
from jinja2 import Environment, FileSystemLoader, select_autoescape

def publish(
    collection_path: Path = typer.Argument(..., help="Path to the collections", exists=True),
    data_path: Path = typer.Argument(..., help="Path to the data file", exists=True),
    site_dir: Path = typer.Argument(..., help="Output _site directory")
):
    """
    Load the json data from the process step
    Create a website, single page txt file and PDF file with text and image side by side."""
    # create a website
    
    # if the site directory exists, remove it
    # this avoids bugs where artifacts from old builds cause unexpected results
    if site_dir.exists():
        for item in track(site_dir.iterdir(), description="Removing files"):
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    # passthrough for index.html
    env = Environment(
        loader=FileSystemLoader("_templates"),
        autoescape=select_autoescape()
    )
    template = template = env.get_template("index.html")
    index = template.render()
    (site_dir / "index.html").write_text(index)
    
    template = env.get_template("index.jinja")
    main = template.render()
    (site_dir / "main.html").write_text(main)

    # pass through for _site_assets directory
    shutil.copytree("_site_assets", site_dir / "assets")
    
    # create the assets directory
    assets_dir = site_dir / "assets"
    if not assets_dir.exists():
        assets_dir.mkdir(parents=True, exist_ok=True)
    # create an img directory in assets
    img_dir = assets_dir / "img"
    if not img_dir.exists():
        img_dir.mkdir(parents=True, exist_ok=True)

    data = srsly.read_jsonl(data_path)
    data = list(data)
    # sort by filenames 
    data = sorted(data, key=lambda x: x["image"])
    for i, page in track(enumerate(data)):
        template = env.get_template("page.jinja")
        date = page.get("dates", None) #"dates":[{"text":"February 1914","start":0,"end":13,"label":"DATE"},...
        text = page.get("text", None)
        if text:
            text = markdown.markdown(text)
        image = page.get("image", 'placeholder')
        ents = page.get("ents", None) #"ents":[{"text":"MEMORANDA","start":0,"end":9,"label":"PERSON"}]
        # get next and previous from data 
        previous = data[i-1]["image"] if i > 0 else None
        next = data[i+1]["image"] if i < len(data) - 1 else None
        html = template.render(image=image, text=text, date=date, ents=ents, previous=previous, next=next)
        page_path = site_dir / f"{image}.html"
        page_path.write_text(html)

        # copy the image to the img directory
        image_path = collection_path / page["image"]
        if not image_path.exists():
            print(f"Image {image_path} not found")
        image_name = image_path.name
        img_path = img_dir / image_name
        img_path.write_bytes(image_path.read_bytes())

    # use subprocess to run pagefind: npx pagefind --source _site --glob "**/*.html"
    subprocess.run(["npx", "pagefind", "--site", "_site"])

if __name__ == "__main__":
    typer.run(publish)