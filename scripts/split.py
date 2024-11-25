import typer 
from PIL import Image, ImageDraw
from pathlib import Path
from rich.progress import track

def remove_ruler(img): 
    img1 = ImageDraw.Draw(img)
    w, h = img.size
    shape = [(0, 0), (w, 100)]  # Upper left, lower right
    img1.rectangle(shape, fill="#000000")
    return img

def split(
    collection_path: Path = typer.Argument(..., help="Path to the collections", exists=True),
    out_dir: Path = typer.Argument(..., help="Output directory to save the split images"),
    skip_split: bool = typer.Option(False, help="Flag to skip splitting images")
):
    if not out_dir.exists():
         out_dir.mkdir(parents=True, exist_ok=True)

    images = list(collection_path.glob("**/*.jpg"))
    # do not split the first and last image in a notebook
    skip = ['SM_NPQ_C01_001.jpg',
            'SM_NPQ_C01_104.jpg',
            'SM_NPQ_C02_001.jpg',
            'SM_NPQ_C02_104.jpg',
            'SM_NPQ_C03_001.jpg',
            'SM_NPQ_C03_104.jpg',
            'SM_NPQ_C04_001.jpg',
            'SM_NPQ_C04_104.jpg',
            'SM_NPQ_C05_001.jpg',
            'SM_NPQ_C05_071.jpg'
            ]
    for image in track(images, description="Processing images..."):
        if image.name in skip or skip_split:
            img = Image.open(image)
            img.save(out_dir / f"{image.stem}.jpg")
        else:
            img = Image.open(image)
            width, height = img.size
            center_line = width // 2
            left = img.crop((0, 0, center_line, height))
            right = img.crop((center_line, 0, width, height))
            left.save(out_dir / f"{image.stem}_left.jpg")
            right.save(out_dir / f"{image.stem}_right.jpg")
            # if 'SM_NPQ_C01' in image.stem:
            #     center_line = 2.2
            #     left = img.crop((0, 0, width//center_line, height))
            #     # left = remove_ruler(left)
            #     right = img.crop((width//center_line, 0, width, height))
            #     # right = remove_ruler(right)
            #     left.save(out_dir / f"{image.stem}_right.jpg")
            #     right.save(out_dir / f"{image.stem}_left.jpg")
            # if 'SM_NPQ_C02' in image.stem:
            #     center_line = 2.3
            #     left = img.crop((0, 0, width//center_line, height))
            #     # left = remove_ruler(left)
            #     right = img.crop((width//center_line, 0, width, height))
            #     # right = remove_ruler(right)
            #     left.save(out_dir / f"{image.stem}_right.jpg")
            #     right.save(out_dir / f"{image.stem}_left.jpg")
            # if 'SM_NPQ_C03' in image.stem:
            #     center_line = 2.3
            #     left = img.crop((0, 0, width//center_line, height))
            #     # left = remove_ruler(left)
            #     right = img.crop((width//center_line, 0, width, height))
            #     # right = remove_ruler(right)
            #     left.save(out_dir / f"{image.stem}_right.jpg")
            #     right.save(out_dir / f"{image.stem}_left.jpg")
            # if 'SM_NPQ_C04' in image.stem:
            #     center_line = 2.2
            #     left = img.crop((0, 0, width//center_line, height))
            #     # left = remove_ruler(left)
            #     right = img.crop((width//center_line, 0, width, height))
            #     # right = remove_ruler(right)
            #     left.save(out_dir / f"{image.stem}_right.jpg")
            #     right.save(out_dir / f"{image.stem}_left.jpg")
            # if 'SM_NPQ_C05' in image.stem:
            #     center_line = 2
            #     left = img.crop((0, 0, width//center_line, height))
            #     # left = remove_ruler(left)
            #     right = img.crop((width//center_line, 0, width, height))
            #     # right = remove_ruler(right)
            #     left.save(out_dir / f"{image.stem}_right.jpg")
            #     right.save(out_dir / f"{image.stem}_left.jpg")
            
if __name__ == "__main__":
    typer.run(split)