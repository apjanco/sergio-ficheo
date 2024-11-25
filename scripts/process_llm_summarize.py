import typer
import srsly
from pathlib import Path
from rich.progress import track
from rich import print
from typing_extensions import Annotated
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import StreamingStrOutputParser
import yaml
import os
import openai
import signal
import sys

# Load configuration
with open("/Users/dtubb/code/sergio-ficheo/project.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

app = typer.Typer()

def count_tokens(text: str) -> int:
    return len(text.split())

def split_text(text, max_tokens):
    words = text.split()
    chunks = []
    while words:
        chunk = words[:max_tokens]
        chunks.append(" ".join(chunk))
        words = words[max_tokens:]
    return chunks

def handle_sigint(signal, frame):
    print("\n[red]Process interrupted. Exiting...[/red]")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

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
        prompt_template = "Describe and summarize TEXT in Spanish. IGNORE INDECIPHERABLE NUMBERS AND TEXT. :\n\n{text}"
    else:
        llm = ChatOllama(model=llm_model, format="json", num_ctx=8000, temperature=0)
        prompt_template = PromptTemplate(template="Describe and summarize TEXT in Spanish. IGNORE INDECIPHERABLE NUMBERS AND TEXT. :\n\n{text}", input_variables=["text"])

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
            input_tokens = count_tokens(full_prompt)
            max_output_tokens = 8192 - input_tokens - 100  # Reserve some tokens for prompt and response structure
            text_chunks = split_text(text, max_output_tokens)
            result = ""
            for chunk in text_chunks:
                chunk_prompt = prompt_template.replace("{text}", chunk)
                print(f"[blue]Sending chunk to OpenAI:\n{chunk_prompt}[/blue]")  # Debug print
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": chunk_prompt}
                    ],
                    max_tokens=max_output_tokens,
                    temperature=0
                )
                result += response.choices[0].message['content'].strip() + "\n"
        else:
            input_tokens = count_tokens(full_prompt)
            max_output_tokens = min(8000 - input_tokens - 100, len(text.split()))  # Adjust based on the local LLM's capacity and text length
            text_chunks = split_text(text, max_output_tokens)
            result = ""
            for chunk in text_chunks:
                chunk_prompt = prompt_template.template.replace("{text}", chunk)
                print(f"[blue]Sending chunk to local LLM:\n{chunk_prompt}[/blue]")  # Debug print
                prompt = prompt_template | llm | StreamingStrOutputParser()
                chunk_result = ""
                for char in prompt.invoke({"text": chunk}):
                    print(char, end='', flush=True)  # Print each character as it comes in
                    chunk_result += char
                result += chunk_result.strip().strip('"') + "\n"

        print(f"[blue]LLM output:\n{result}[/blue]")

        item[output_field] = result.strip()  # Ensure the full result is saved

        # Update the original file with the processed item
        data[idx] = item
        srsly.write_jsonl(json_file, data)

        # Update the progress file
        processed_data.append(item)
        srsly.write_jsonl(progress_file, processed_data)

if __name__ == "__main__":
    typer.run(process_llm_summarize)
