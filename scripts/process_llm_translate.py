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

app = typer.Typer()

def process_llm_translate(
    json_file: Annotated[Path, typer.Argument(help="Path to the JSONL file", exists=True)],
    llm_model: Annotated[str, typer.Argument(help="LLM model name")],
    output_field: Annotated[str, typer.Argument(help="Field to store the LLM result")],
):
    data = list(srsly.read_jsonl(json_file))
    out_file = json_file.with_stem(json_file.stem + f"_{output_field}")
    
    if out_file.exists():
        print(f"Output file {out_file} already exists. Skipping processing.")
        return

    processed_data = []

    llm = ChatOllama(model=llm_model, format="json", num_ctx=8000, temperature=0)
    prompt_template = PromptTemplate(template="Translate the following text to English:\n\n{text}", input_variables=["text"])

    for idx, item in enumerate(track(data, description=f"Performing LLM translation...")):
        if output_field in item:
            processed_data.append(item)
            continue

        text = item["text"]
        full_prompt = prompt_template.template.replace("{text}", text)
        print(f"Processing image: {item.get('image', 'No image field')}")

        prompt = prompt_template | llm | StrOutputParser()
        result = prompt.invoke({"text": text})

        try:
            processed_data_item = json.loads(result)
        except json.JSONDecodeError:
            processed_data_item = clean_invalid_json(result)

        item[output_field] = processed_data_item.get("text", processed_data_item)
        if isinstance(processed_data_item, dict) and "summary" in processed_data_item:
            item[output_field] = processed_data_item["summary"]

        processed_data.append(item)
        srsly.write_jsonl(json_file.with_stem(json_file.stem + f"_{output_field}_progress"), processed_data)

    srsly.write_jsonl(out_file, processed_data)

def clean_invalid_json(text):
    llm = ChatOllama(model="llama3.1:8b", format="json", num_ctx=8000, temperature=0)
    prompt_template = PromptTemplate(template=Path("/Users/dtubb/code/sergio-ficheo/prompts/cleaner.txt").read_text(), input_variables=["text"])
    prompt = prompt_template | llm | StrOutputParser()
    result = prompt.invoke({"text": text})
    
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {}

if __name__ == "__main__":
    typer.run(process_llm_translate)
