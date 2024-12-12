import typer
import srsly
from pathlib import Path
from rich.progress import track
from typing_extensions import Annotated
from fuzzywuzzy import fuzz, process
import logging

app = typer.Typer()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fuzzy_clean_ner(json_file: Annotated[Path, typer.Argument(help="Path to the JSONL file", exists=True)]):
    """
    Perform fuzzy matching to clean NER data.
    """
    logging.info("Starting fuzzy matching for NER data")

    # Load the entire JSONL file into a list of data items
    data = list(srsly.read_jsonl(json_file))
    logging.info(f"Loaded {len(data)} records from {json_file}")

    # Separate entities by type
    entities_by_type = {'PER': [], 'LOC': [], 'ORG': [], 'MISC': []}
    for item in data:
        for ent in item["entities"]:
            if ent["label"] in entities_by_type:
                entities_by_type[ent["label"]].append(ent["text"])

    # Perform fuzzy matching to group similar entity names within each type
    matched_entities = {}
    for entity_type, entities in entities_by_type.items():
        unique_entities = list(set(entities))
        matched_entities[entity_type] = {}
        for entity in unique_entities:
            matches = process.extract(entity, unique_entities, scorer=fuzz.token_sort_ratio)
            for match, score in matches:
                if score > 80 and match != entity:
                    if entity not in matched_entities[entity_type]:
                        matched_entities[entity_type][entity] = []
                    matched_entities[entity_type][entity].append(match)
            logging.info(f"Entity: {entity}, Matches: {matches}")

    logging.info("Completed fuzzy matching of entity names")

    # Replace similar entity names with the most frequent one within each type
    for item in data:
        for ent in item["entities"]:
            entity_type = ent["label"]
            for key, values in matched_entities[entity_type].items():
                if ent["text"] in values:
                    ent["text"] = key
                    break

    logging.info("Replaced similar entity names with the most frequent one")

    # Save the cleaned data with NER to a new JSONL file
    out_file = json_file.with_name("data_ner_fuzzy_cleaned.jsonl")
    srsly.write_jsonl(out_file, data)
    logging.info(f"Saved fuzzy cleaned NER data to {out_file}")

@app.command()
def main(
    json_file: Path = typer.Argument(..., help="Path to the JSONL file")
):
    fuzzy_clean_ner(json_file)

if __name__ == "__main__":
    typer.run(main)
