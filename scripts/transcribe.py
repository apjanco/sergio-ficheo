import re
import typer
from pathlib import Path
import atexit
from PIL import Image
import numpy as np
import torch
import multiprocessing
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from rich.console import Console
from utils.batch import BatchProcessor

console = Console()

DEFAULT_PROMPT = "Extract all text line by line. Do not number lines. RETURN ONLY PLAIN TEXT. SAY NOTHING ELSE"

def cleanup_resources():
    try:
        for tracker in multiprocessing.resource_tracker._resource_tracker._handlers.values():
            tracker.join()
    except Exception:
        pass
atexit.register(cleanup_resources)

class TranscriptionProcessor:
    _instance = None
    _model = None
    _processor = None

    def __new__(cls, model_name: str = None, prompt: str = DEFAULT_PROMPT):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = None, prompt: str = DEFAULT_PROMPT):
        if not hasattr(self, 'initialized'):
            self.model_name = model_name
            self.prompt = prompt
            self.initialized = True
            self._load_model()

    def _load_model(self):
        if self._model is None and self.model_name:
            try:
                console.print(f"[yellow]Loading model {self.model_name}...")
                self._processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    device_map="auto"
                )
                console.print("[green]Model loaded successfully")
            except Exception as e:
                console.print(f"[red]Error loading model: {e}")
                self._model = None
                self._processor = None

    @property
    def model(self):
        return self._model

    @property
    def processor(self):
        return self._processor

    @property
    def tokenizer(self):
        return self._processor.tokenizer if self._processor else None

    def estimate_text_density(self, image: Image.Image) -> int:
        try:
            img_array = np.array(image.convert('L'))
            height, width = img_array.shape
            mean = np.mean(img_array)
            std_dev = np.std(img_array)
            
            text_mask = img_array < (mean - 0.5 * std_dev)
            text_pixel_count = np.sum(text_mask)
            
            pixels_per_word = 8000
            estimated_words = int(text_pixel_count / pixels_per_word)
            
            if width * height < 500000:
                estimated_words = max(8, estimated_words)
            else:
                estimated_words = max(15, estimated_words)
            
            return min(estimated_words, 40)
        except Exception:
            return 15

    def count_tokens(self, text: str) -> int:
        if not self.tokenizer:
            return len(text.split())
        return len(self.tokenizer.encode(text))

    def process_image(self, image: Image.Image) -> str:
        if not self.model or not self.processor:
            return "Model not available - Mock transcription"

        try:
            max_size = 1000
            width, height = image.size
            aspect_ratio = max(width, height) / float(min(width, height))
            if aspect_ratio > 200:
                return ""

            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int((max_size / width) * height)
                else:
                    new_height = max_size
                    new_width = int((max_size / height) * width)
                image = image.resize((new_width, new_height), Image.LANCZOS)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]
            # Convert messages to a single text prompt
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Let the processor handle both text and images
            inputs = self.processor(
                text=prompt_text,
                images=image,
                return_tensors="pt",
                max_length=2048,
                truncation=True
            )

            device = next(self.model.parameters()).device
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=800,
                    min_new_tokens=10,
                    num_beams=1,
                    do_sample=False,         # or True if you want sampling
                    repetition_penalty=1.1,
                    length_penalty=1.0,
                    temperature=0.01,        # remove or set do_sample=True
                    top_p=0.001,             # remove or set do_sample=True
                    top_k=1,                 # remove or set do_sample=True
                )

            input_len = inputs["input_ids"].shape[1]
            output_text = self.tokenizer.decode(
                outputs[0][input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()

            # Filter out non-useful outputs
            if not output_text or output_text.lower() == "blank":
                return ""
            if re.match(r"^\(\d+,\d+\),\(\d+,\d+\)$", output_text):
                return ""
            if output_text in [
                "The text is not visible in the image.",
                "The text on the image is not clear and appears to be a mix of different colors and patterns."
            ]:
                return ""
            return output_text

        except Exception as e:
            console.print(f"[red]Error in vision-language processing: {e}")
            return f"Error in processing: {str(e)}"

def process_chunk(img_file: Path, output_folder: Path, transcriber: TranscriptionProcessor) -> dict:
    try:
        img_path = img_file.resolve()
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
        image = Image.open(img_path).convert("RGB")
        chunk_match = re.search(r'_chunk_(\d+)', img_path.name)
        chunk_num = int(chunk_match.group(1)) if chunk_match else 0
        
        parent_folder = img_path.parent
        while parent_folder.name and not parent_folder.name.endswith('_chunks'):
            parent_folder = parent_folder.parent
        
        if parent_folder.name.endswith('_chunks'):
            parent_image_name = parent_folder.name[:-7]
            parent_image_path = parent_folder.parent / f"{parent_image_name}.png"
        else:
            parent_image_path = img_path.parent / f"{img_path.stem.rsplit('_chunk_', 1)[0]}.png"

        if '_chunks' in str(img_path):
            rel_path = img_path.relative_to(img_path.parent.parent)
        else:
            rel_path = img_path.name

        output_path = output_folder / rel_path.with_suffix('.md')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        estimated_words = transcriber.estimate_text_density(image)
        text = transcriber.process_image(image)
        token_count = transcriber.count_tokens(text)
        
        output_path.write_text(text)
        
        return {
            "chunk_id": chunk_num,
            "source": str(img_file.relative_to(img_file.parent.parent)),
            "parent_image": str(parent_image_path),
            "outputs": [str(output_path.relative_to(output_folder))],
            "details": {
                "estimated_words": estimated_words,
                "token_count": token_count,
                "has_content": bool(text.strip()),
                "parent_info": {
                    "folder": str(img_file.parent),
                    "original_image": str(parent_image_path)
                }
            }
        }
    except Exception as e:
        console.print(f"[red]Error processing chunk {img_file}: {e}")
        return {"error": str(e)}

def process_document(file_path: str, output_folder: Path, model_name: str = None, prompt: str = DEFAULT_PROMPT) -> dict:
    path = Path(file_path)
    transcriber = TranscriptionProcessor(model_name, prompt)
    
    if path.is_dir():
        chunks = []
        for img_file in sorted(
            path.glob("*.jpg"),
            key=lambda x: int(re.search(r'_chunk_(\d+)', x.name).group(1)) 
                         if re.search(r'_chunk_(\d+)', x.name) else 0
        ):
            chunk_info = process_chunk(img_file, output_folder, transcriber)
            chunks.append(chunk_info)
        return {
            "source": str(path),
            "chunks": chunks,
            "success": True if not any(c.get("error") for c in chunks) else False
        }
    else:
        chunk_info = process_chunk(path, output_folder, transcriber)
        return {
            "source": str(path),
            "chunks": [chunk_info],
            "success": not bool(chunk_info.get("error"))
        }

def transcribe(
    chunk_folder: Path = typer.Argument(..., help="Input chunks folder"),
    chunk_manifest: Path = typer.Argument(..., help="Input chunks manifest"),
    transcribed_folder: Path = typer.Argument(..., help="Output folder for transcriptions"),
    model_name: str = typer.Option(
        "Qwen/Qwen2-VL-2B-Instruct",
        "--model", "-m",
        help="Model name to use"
    ),
    prompt: str = typer.Option(
        DEFAULT_PROMPT,
        "--prompt", "-p",
        help="Prompt for transcription"
    )
):
    console.print(f"Using model: {model_name}")
    console.print(f"Using prompt: {prompt}")
    
    processor = BatchProcessor(
        input_manifest=chunk_manifest,
        output_folder=transcribed_folder,
        process_name="transcribe",
        processor_fn=lambda f, o: process_document(f, o, model_name, prompt),
        base_folder=chunk_folder
    )
    
    return processor.process()

if __name__ == "__main__":
    typer.run(transcribe)