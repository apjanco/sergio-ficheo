import typer
import srsly
from pathlib import Path
from rich.progress import track
from rich import print
from typing_extensions import Annotated
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
import yaml
import os
import signal
import sys
from langchain.schema import HumanMessage

# Load configuration
with open("/Users/dtubb/code/sergio-ficheo/project.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

app = typer.Typer()

def count_tokens(text: str) -> int:
    return len(text.split())

def handle_sigint(signal, frame):
    print("\n[red]Process interrupted. Exiting...[/red]")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

def chunk_text(text: str, max_tokens: int) -> list:
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def combine_summaries(summaries: list, llm: ChatOllama) -> str:
    combined_text = " ".join(summaries)
    prompt = f"Combine the following summaries into a single coherent summary in Spanish:\n\n{combined_text}"
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        if response:
            result = response.content.strip()
            return result
        else:
            print("No valid response from LLM for combining summaries.")
            return combined_text
    except Exception as e:
        print(f"Error invoking LLM for combining summaries: {e}")
        return combined_text

def process_llm_summarize(
    json_file: Annotated[Path, typer.Argument(help="Path to the JSONL file", exists=True)],
    llm_model: Annotated[str, typer.Argument(help="LLM model name")],
    output_field: Annotated[str, typer.Argument(help="Field to store the LLM result")],
):
    data = list(srsly.read_jsonl(json_file))
    progress_file = json_file.with_stem(json_file.stem + f"_{output_field}_progress")

    if progress_file.exists():
        processed_data = list(srsly.read_jsonl(progress_file))
        processed_ids = {item['image'] for item in processed_data}
    else:
        processed_data = []
        processed_ids = set()

    if llm_model == "chatgpt-4.0-mini":
        openai.api_key = os.getenv("OPENAI_API_KEY")
        prompt_template = """Describe and summarize TEXT in Spanish. RETURN AS VALID JSON DICTIONARY WITH ONE KEY `summary`. SUMMARY MUST BE SPANISH. \nTEXT: {text}"""
    else:
        llm = ChatOllama(model=llm_model, format="json", num_ctx=1000, temperature=0)
        prompt_template = PromptTemplate(template="Describe and summarize TEXT in Spanish. RETURN AS VALID JSON DICTIONARY WITH ONE KEY `summary`. SUMMARY MUST BE SPANISH. \nTEXT: {text}", input_variables=["text"])

    for idx, item in enumerate(track(data, description=f"Performing LLM summarization...")):
        if output_field in item:
            print(f"[yellow]Skipping already processed item: {item['image']}[/yellow]")
            processed_data.append(item) 
            continue

        text = item.get("cleaned_text")
        if not text:
            print(f"[yellow]Skipping item with no cleaned text: {item['image']}[/yellow]")
            processed_data.append(item)
            continue

        full_prompt = prompt_template.replace("{text}", text) if llm_model == "chatgpt-4.0-mini" else prompt_template.template.replace("{text}", text)
        print(f"[green]Processing item: {item['image']}[/green]")
        print(f"[blue]Full prompt:\n{full_prompt}[/blue]")

        if llm_model == "chatgpt-4.0-mini":
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": full_prompt}
                    ],
                    max_tokens=8192,
                    temperature=0
                )
                result = response.choices[0].message['content'].strip()
                summary = result.get("summary", "")
            except Exception as e:
                print(f"Error invoking ChatGPT: {e}")
                summary = ""
        else:
            try:
                response = llm.invoke([HumanMessage(content=full_prompt)])
                if response:
                    result = response.content.strip()
                    print(f"\n[blue]LLM output:\n{result}[/blue]")
                    summary = result.get("summary", "")
                else:
                    print("No valid response from LLM.")
                    summary = ""
            except Exception as e:
                print(f"Error invoking LLM: {e}")
                summary = ""

        item[output_field] = summary.strip()  # Ensure only the summary is saved as plain text

        # Update the original file with the processed item
        data[idx] = item
        srsly.write_jsonl(json_file, data)

        # Update the progress file
        processed_data.append(item)
        srsly.write_jsonl(progress_file, processed_data)

if __name__ == "__main__":
    typer.run(process_llm_summarize)
