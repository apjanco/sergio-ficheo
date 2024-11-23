import typer
import srsly
from pathlib import Path
from rich.progress import track
from rich import print
from typing_extensions import Annotated
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import openai
import os
import tiktoken

app = typer.Typer()

def count_tokens(text: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def split_text(text, max_tokens, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    chunks = []
    while tokens:
        chunk = tokens[:max_tokens]
        chunks.append(encoding.decode(chunk))
        tokens = tokens[max_tokens:]
    return chunks

def process_llm_clean(
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
        prompt_template = """Imagine yourself as an AI archivist and quality assurance expert. I will provide you with text enclosed in double quotation marks. Without making any substantive changes, your task is to:

- Revise punctuation, line spacing, and paragraph breaks for clarity
- Fix any spelling, grammar, and obvious punctuation errors based on context.
- Break the text into paragraphs at logical places.
- If edits like adding a word or adjusting spelling based on context, put the changes in [square brackets].
- Add any bold/italic formatting needed.
- For all dates writted out, add formated dates in brackets based . e.g quince de marzo de mil ochocientos och = [1808-03-15]
- Remove any LLM repeats, echoed text, random characters or numbers, or Pipes (e.g. "|":). 

"{text}" """
    else:
        llm = ChatOllama(model=llm_model, format="json", num_ctx=8000, temperature=0)
        prompt_template = PromptTemplate(template="""Imagine yourself as an AI archivist and quality assurance expert. I will provide you with text enclosed in double quotation marks. Without making any substantive changes, your task is to:

- Revise punctuation, line spacing, and paragraph breaks for clarity
- Fix any spelling, grammar, and obvious punctuation errors based on context.
- Break the text into paragraphs at logical places.
- If edits like adding a word or adjusting spelling based on context, put the changes in [square brackets].
- Add any bold/italic formatting needed.
- For all dates writted out, add formated dates in brackets based . e.g quince de marzo de mil ochocientos och = [1808-03-15]
- Remove any LLM repeats, echoed text, random characters or numbers, or Pipes (e.g. "|":). 

"{text}" """, input_variables=["text"])

    for idx, item in enumerate(track(data, description=f"Performing LLM text cleaning...")):
        if output_field in item:
            print(f"[yellow]Skipping already processed item: {item['image']}[/yellow]")
            continue

        text = item["text"]
        full_prompt = prompt_template.replace("{text}", text) if llm_model == "chatgpt-4.0-mini" else prompt_template.template.replace("{text}", text)
        print(f"[green]Processing item: {item['image']}[/green]")
        print(f"[blue]Full prompt:\n{full_prompt}[/blue]")

        if llm_model == "chatgpt-4.0-mini":
            input_tokens = count_tokens(full_prompt, model="gpt-4")
            max_output_tokens = 8192 - input_tokens - 100  # Reserve some tokens for prompt and response structure
            text_chunks = split_text(text, max_output_tokens)
            result = ""
            for chunk in text_chunks:
                chunk_prompt = prompt_template.replace("{text}", chunk)
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
            prompt = prompt_template | llm | StrOutputParser()
            result = prompt.invoke({"text": text})

        print(f"[blue]LLM output:\n{result}[/blue]")

        # Strip enclosing quotes
        result = result.strip().strip('"')

        item[output_field] = result  # Ensure the full result is saved

        # Update the original file with the processed item
        data[idx] = item
        srsly.write_jsonl(json_file, data)

        # Update the progress file
        processed_data.append(item)
        srsly.write_jsonl(progress_file, processed_data)

if __name__ == "__main__":
    typer.run(process_llm_clean)