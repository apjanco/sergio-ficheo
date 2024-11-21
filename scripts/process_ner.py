import typer
import spacy
import srsly
from pathlib import Path
from rich.progress import track
from typing_extensions import Annotated

app = typer.Typer()

def process_ner(
    json_file: Annotated[Path, typer.Argument(help="Path to the JSONL file", exists=True)],
    spacy_model: Annotated[str, typer.Argument(help="spaCy model name")] = "es_core_news_lg",
):
    """
    Perform named entity recognition (NER) on the text in the JSONL file.
    """
    # Load or download spaCy model
    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        from spacy.cli.download import download
        download(spacy_model)
        nlp = spacy.load(spacy_model)

    data = list(srsly.read_jsonl(json_file))  # Convert generator to list and process all entries
    
    for item in track(data, description="Performing NER..."):
        text = item["text"]
        doc = nlp(text)
        # Extract named entities
        ents = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        # Calculate the frequency for each entity
        entities = {}
        for ent in ents:
            if ent["text"] in entities:
                entities[ent["text"]] += 1
            else:
                entities[ent["text"]] = 1
        # Sort the entities by frequency
        entities = dict(sorted(entities.items(), key=lambda item: item[1], reverse=True))
        # Add the frequencies to ents
        for ent in ents:
            ent["frequency"] = entities[ent["text"]]
        item["entities"] = ents

    # Save the processed data with NER to a new JSONL file
    out_file = json_file.with_stem(json_file.stem + "_ner")
    srsly.write_jsonl(out_file, data)

if __name__ == "__main__":
    typer.run(process_ner)