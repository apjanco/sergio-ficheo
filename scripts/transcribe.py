import typer
from rich import print
from rich.progress import track
from typing_extensions import Annotated
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from pathlib import Path 
from PIL import Image
import torch
import yaml
import re

def transcribe(
    collection_path: Annotated[Path, typer.Argument(help="Path to the collections", exists=True)],
    model_name: Annotated[str, typer.Argument(help="HF model name")],
    output_folder: Annotated[Path, typer.Argument(help="Output folder for transcriptions")],  # Remove exists=True
    prompt: Annotated[str, typer.Argument(help="Prompt for the transcription")]
):
    print(f"[green]Transcribing images in {collection_path}")
    print(f"[cyan]Using model {model_name}")
    print(f"[yellow]Using prompt: {prompt}")
    
    if not output_folder.exists():
         output_folder.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    
    images = sorted(collection_path.glob("**/*.jpg"), key=lambda x: x.stem)
    for image_path in track(images, total=len(images)):
        image = Image.open(image_path).convert("RGB")
        
        # Resize based on the size of the chunk up to a maximum of 1000
        max_size = 1000
        width, height = image.size
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 200:
            print(f"[red]Skipping {image_path} due to unacceptable aspect ratio: {aspect_ratio}")
            continue
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int((max_size / width) * height)
            else:
                new_height = max_size
                new_width = int((max_size / height) * width)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Check for image size before sending
        if image.size[0] > max_size or image.size[1] > max_size:
            print(f"[red]Skipping {image_path} due to size exceeding {max_size}px")
            continue
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        output_file = (output_folder / image_path.relative_to(collection_path)).with_suffix(".md")
        if output_file.exists():
            continue
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=800)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_text = output_text[0].strip()
        
        # Check if the output is coordinates or blank
        if re.match(r"^\(\d+,\d+\),\(\d+,\d+\)$", output_text) or output_text.lower() == "blank" or not output_text or output_text in ["The text is not visible in the image.", "The text on the image is not clear and appears to be a mix of different colors and patterns. It is difficult to extract any meaningful information from it."]:
            output_text = ""

        output_file.write_text(output_text)

if __name__ == "__main__":
    typer.run(transcribe)