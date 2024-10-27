import typer 
import srsly
import json
from pathlib import Path
from rich.progress import track
from transformers import AutoModelForCausalLM, AutoTokenizer
import spacy 

def process(
    data_path: Path = typer.Argument(..., help="Path to the collections", exists=True),
    model_name: str = typer.Argument(..., help="name of spaCy model"),
    out_file: Path = typer.Argument(..., help="Output path and filename to save the data file")
):
    """
    Process images and text. Outcome is a single a json file with a dictionary for each page with the transcribed text, recognized 
    people, locations, and dates as well as the summary text.         
    """
    nlp = spacy.load(model_name)
    
    txt_files = list(data_path.glob("**/*.txt"))
    
    data = []
    for txt_file in track(txt_files):
        img_data = {}
        img_data["image"] = txt_file.with_suffix('.jpg').name
        text = txt_file.read_text()
        doc = nlp(text)
        img_data["text"] = text
        img_data["ents"] = [{"text":ent.text, "start":ent.start_char, "end":ent.end_char, "label":ent.label_} for ent in doc.ents]
        data.append(img_data)
        
    srsly.write_jsonl(out_file, data)


if __name__ == "__main__":
    typer.run(process)