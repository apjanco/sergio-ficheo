import torch
import typer
from PIL import Image
from pathlib import Path
from rich.progress import track
from spacy.cli._util import Arg

from kraken import blla, binarization, serialization



def segment(
    collection_path: Path = Arg(..., help="Path to the collections", exists=True),
    text_direction: str = Arg("horizontal-lr", help="Text direction of the images"),
):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    images = list(collection_path.glob("**/*.jpg"))
    print(f"Found {len(images)} images")
    for image in track(images, description="Segmenting images..."):
        if not image.with_suffix('.xml').exists():
            img = Image.open(image)
            bw_im = binarization.nlbin(img)
            try:
                baseline_seg = blla.segment(bw_im, device=device, text_direction=text_direction)
                baseline_seg.imagename = image.stem + image.suffix
                # Temporary hack following code update 
                # for line in baseline_seg.lines:
                #     line.cuts = []
                #     line.confidences = []
                #     line.prediction = ''
                alto_xml = serialization.serialize(baseline_seg, image_size=img.size, template='alto')
                image.with_suffix('.xml').write_text(alto_xml)
            except Exception as e:
                print(f"error in {image}.")
                continue
   
if __name__ == "__main__":
    typer.run(segment)
