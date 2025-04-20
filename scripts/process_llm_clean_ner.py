import typer
import srsly
from pathlib import Path
from rich.progress import track
from typing_extensions import Annotated
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import yaml
import pprint
import logging

app = typer.Typer()

# Configure logging to write to a file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename='process_llm_clean_ner.log', filemode='w')

def process_llm_clean_ner(
    json_file: Annotated[Path, typer.Argument(help="Path to the JSONL file", exists=True)],
    llm_model: Annotated[str, typer.Argument(help="LLM model name")]
):
    data = list(srsly.read_jsonl(json_file))  # Read the JSONL file into a list of data items
    
    llm = ChatOllama(model=llm_model, format="json", num_ctx=4000, temperature=0)  # Reduce context size
    prompt_template = PromptTemplate(
        template="Review and correct the following named entities (PER, LOC, ORG) in the context of the original text. Return the corrected entities in JSON format:\n\nText: {text}\n\nEntities:\n{entities}",
        input_variables=["text", "entities"]
    )

    for idx, item in enumerate(track(data, description=f"Performing LLM NER cleaning...")):
        if "cleaned_ner" in item:  # Skip re-cleaning if already cleaned
            print(f"Skipping already cleaned item {idx}")
            continue

        text = item["text"]
        entities = yaml.dump(item["entities"])
        full_prompt = prompt_template.template.replace("{text}", text).replace("{entities}", entities)
        logging.info(f"Processing image: {item.get('image', 'No image field')}")
        print(f"Processing item {idx}")

        prompt = prompt_template | llm | StrOutputParser()
        result = prompt.invoke({"text": text, "entities": entities})

        try:
            processed_data_item = json.loads(result)  # Parse the LLM result
            old_entities = item["entities"]
            new_entities = processed_data_item.get("entities", processed_data_item)
            if isinstance(new_entities, list):
                new_entities = {"Entities": new_entities}
            item["entities"] = new_entities  # Update the item with cleaned entities
            logging.info(f"Old Entities: {old_entities}")
            logging.info(f"New Entities: {new_entities}")
            print(f"Updated entities for item {idx}")
        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON for item: {item.get('image', 'No image field')}")
            print(f"Failed to parse JSON for item {idx}")
            continue

        item["cleaned_ner"] = True  # Set the cleaned flag
        pprint.pprint(item)

        # Save the updated data after processing each item
        srsly.write_jsonl(json_file, data)
        print(f"Saved updated data for item {idx}")

@app.command()
def main(
    json_file: Path = typer.Argument(..., help="Path to the JSONL file"),
    llm_model: str = typer.Argument(..., help="LLM model name")
):
    process_llm_clean_ner(json_file, llm_model)

if __name__ == "__main__":
    typer.run(main)