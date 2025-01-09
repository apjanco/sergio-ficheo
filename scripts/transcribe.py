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

# Add cleanup handler
def cleanup_resources():
    for tracker in multiprocessing.resource_tracker._resource_tracker._handlers.values():
        tracker.join()
atexit.register(cleanup_resources)

class TranscriptionProcessor:
    _instance = None
    _model = None
    _processor = None

    def __new__(cls, model_name: str = None, prompt: str = DEFAULT_PROMPT):
        if cls._instance is None:
            cls._instance = super(TranscriptionProcessor, cls).__new__(cls)
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
                self._processor = AutoProcessor.from_pretrained(self.model_name)
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,  # Changed from float16
                    device_map="cuda" if torch.cuda.is_available() else "cpu"
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
        """Alias for processor to maintain compatibility"""
        return self._processor

    def estimate_text_density(self, image: Image.Image) -> int:
        """Estimate number of words in image based on text density analysis"""
        try:
            # Convert to grayscale numpy array
            img_array = np.array(image.convert('L'))
            height, width = img_array.shape
            
            # Calculate basic image statistics
            mean = np.mean(img_array)
            std_dev = np.std(img_array)
            
            # More lenient threshold to catch lighter handwriting
            text_mask = img_array < (mean - 0.5 * std_dev)  # Reduced from 0.8 to 0.5
            text_pixel_count = np.sum(text_mask)
            
            # More generous area per word estimate
            pixels_per_word = 8000  # Reduced from 10000 to catch more potential text
            
            # Calculate base estimate with higher tolerance
            estimated_words = int(text_pixel_count / pixels_per_word)
            
            # Adjust scaling to be more generous
            if width * height < 500000:  # Small image
                estimated_words = max(8, estimated_words)  # Increased minimum
            else:  # Larger image
                estimated_words = max(15, estimated_words)  # Increased minimum
            
            # Higher cap for maximum words to avoid missing text
            return min(estimated_words, 40)  # Increased from 25 to 40
            
        except Exception as e:
            return 15  # More generous fallback (up from 10)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's processor"""
        if not self._processor:
            return len(text.split())  # Fallback to word count
        return len(self._processor.tokenizer.encode(text))

    def process_image(self, image: Image.Image) -> str:
        """Process image with Qwen VL model"""
        if not self.model or not self.processor:
            return "Model not available - Mock transcription for testing"
            
        try:
            # Normalize image size
            max_size = 800  # Reduced from 1000
            width, height = image.size
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int((max_size / width) * height)
                else:
                    new_height = max_size
                    new_width = int((max_size / height) * width)
                image = image.resize((new_width, new_height), Image.LANCZOS)

            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]

            # Process inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=800)
                output_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

            return output_text.strip()

        except Exception as e:
            console.print(f"[red]Error in vision-language processing: {e}")
            return f"Error in processing: {str(e)}"

def process_chunk(img_file: Path, output_folder: Path, transcriber: TranscriptionProcessor) -> dict:
    try:
        # Convert to absolute path and verify existence
        img_path = Path(img_file).resolve()
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
        image = Image.open(img_path).convert("RGB")
        chunk_match = re.search(r'_chunk_(\d+)', img_path.name)
        chunk_num = int(chunk_match.group(1)) if chunk_match else 0
        
        # Improve parent path handling
        parent_folder = img_path.parent
        while parent_folder.name and not parent_folder.name.endswith('_chunks'):
            parent_folder = parent_folder.parent
            
        if parent_folder.name.endswith('_chunks'):
            parent_image_name = parent_folder.name[:-7]
            parent_image_path = parent_folder.parent / f"{parent_image_name}.png"
        else:
            parent_image_path = img_path.parent / f"{img_path.stem.rsplit('_chunk_', 1)[0]}.png"
        
        # Use Path operations for output path
        rel_path = img_path.relative_to(img_path.parent.parent) if '_chunks' in str(img_path) else img_path.name
        output_path = output_folder / rel_path.with_suffix('.md')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        estimated_words = transcriber.estimate_text_density(image)
        text = transcriber.process_image(image)
        token_count = transcriber.count_tokens(text)
        output_path.write_text(text)
        
        return {
            "chunk_id": chunk_num,
            "source": str(img_file.relative_to(img_file.parent.parent)),
            "parent_image": str(parent_image_path),  # Use full path to parent image
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
        for img_file in sorted(path.glob("*.jpg"), 
                             key=lambda x: int(re.search(r'_chunk_(\d+)', x.name).group(1)) 
                             if re.search(r'_chunk_(\d+)', x.name) else 0):
            chunk = process_chunk(img_file, output_folder, transcriber)
            chunks.append(chunk)
        
        return {
            "source": str(path),
            "chunks": chunks,
            "success": True if not any(c.get("error") for c in chunks) else False
        }
    else:
        chunk = process_chunk(path, output_folder, transcriber)
        return {
            "source": str(path),
            "chunks": [chunk],
            "success": not bool(chunk.get("error"))
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
    """Transcribe chunked document images to markdown files"""
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