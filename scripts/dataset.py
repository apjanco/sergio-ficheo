import typer
import srsly
from rich import print
from pathlib import Path
from datasets import load_dataset

def dataset(
        collection_path: Path = typer.Argument(..., help="Path to the collections", exists=True),
        dataset_name: str = typer.Argument(..., help="HF Dataset name"),
):
    image_files = list(collection_path.glob("**/*.jpg"))
    
    # generate metadata.jsonl file
    # see https://huggingface.co/docs/datasets/image_dataset#imagefolder
    metadata = []
    for image in image_files:
          metadata.append({"file_name":image.name,"name":image.name})
    srsly.write_jsonl(f"{str(collection_path)}/metadata.jsonl", metadata)

    # create the dataset using the imagefolder option
    dataset = load_dataset("imagefolder", data_dir=collection_path)
    dataset.push_to_hub(dataset_name)
    print(f"Dataset {dataset_name} created and pushed to the Hub")

if __name__ == "__main__":
    typer.run(dataset)