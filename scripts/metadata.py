import typer
import srsly
import pandas as pd 
from rich import print
from pathlib import Path
from rich.progress import track

def metadata(
    spreadsheet_path: Path = typer.Argument(..., help="Path to the spreadsheet", exists=True),
    output_path: Path = typer.Argument(..., help="Output _site directory")
):
    """
    Generate doc-level metadata for the collections.
    """
    df = pd.read_excel(spreadsheet_path)

    # Replace NaN values with an empty string
    df = df.fillna("").replace({pd.NA: "", pd.NaT: "", None: ""})
    
    # create a list of dicts for each row
    metadata = []
    for index, row in track(df.iterrows(), description="Processing rows..."):
        metadata.append(row.to_dict())
    #TODO match docs and split images using the values in 'Número y tipo de material original' ex. 1v-3v
    for item in metadata:
        if "Nivel" == "Documento":
            pages = item["Número y tipo de material original / \nNúmeros de folio"]
            # recto is the front page, verso is the back page

    # Ensure the output directory exists
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # save metadata.jsonl
    metadata_path = output_path / "metadata.jsonl"
    srsly.write_jsonl(metadata_path, metadata)

if __name__ == "__main__":
    typer.run(metadata)
