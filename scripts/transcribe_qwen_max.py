import os
import typer
from rich import print
from rich.progress import track
from typing_extensions import Annotated
from pathlib import Path 
from PIL import Image
from dotenv import load_dotenv 
from io import BytesIO
import base64
from openai import OpenAI
from utils.batch import BatchProcessor
from utils.processor import process_file
from utils.segment_handler import SegmentHandler

# Base 64 encoding format
def encode_image(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def process_image(file_path: Path, out_path: Path) -> dict:
    """Process a single image file, returning manifest-compatible output"""
    try:
        # Ensure output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert output path to .txt extension
        out_path = out_path.parent / (out_path.stem + '.txt')
        out_path.touch()
        
        try:
            print(f"[cyan]Processing image: {file_path}")
            
            # Load and process image
            image = Image.open(file_path).convert("RGB")
            
            # Encode image for API
            base64_image = encode_image(image)
            
            # Initialize OpenAI client with DashScope endpoint
            client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            )
            
            print(f"[cyan]Sending to Qwen API...")
            
            # Get transcription using OpenAI-compatible method
            completion = client.chat.completions.create(
                model="qwen-vl-max",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text line by line. Do not number lines. RETURN ONLY PLAIN TEXT. SAY NOTHING ELSE"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }]
            )
            
            print(f"[green]Received response from Qwen API")
            
            # Extract transcription from response
            transcription = completion.choices[0].message.content
            
            # Save transcription
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(transcription)
            
            # Create manifest entry
            result = {
                "outputs": [str(SegmentHandler.get_relative_path(out_path))],
                "source": str(SegmentHandler.get_relative_path(file_path)),
                "details": {
                    "has_content": bool(transcription.strip())
                }
            }
            
            # Add parent image info
            rel_path = SegmentHandler.get_relative_path(file_path)
            result["parent_image"] = str(rel_path)
                
            return result
            
        except Exception as e:
            print(f"[red]Error processing image {file_path}: {str(e)}")
            # Return error but keep empty file
            return {
                "error": str(e),
                "outputs": [str(SegmentHandler.get_relative_path(out_path))],
                "source": str(SegmentHandler.get_relative_path(file_path))
            }

    except Exception as e:
        print(f"[red]Error processing {file_path}: {e}")
        return {"error": str(e)}

def process_document(file_path: str, output_folder: Path) -> dict:
    """Process a document using the process_file utility"""
    file_path = Path(file_path)
    
    def process_fn(f: str, o: Path) -> dict:
        return process_image(Path(f), o)
    
    return process_file(
        file_path=str(file_path),
        output_folder=output_folder,
        process_fn=process_fn,
        file_types={
            '.png': process_fn
        }
    )

def transcribe(
    background_removed_folder: Path = typer.Argument(..., help="Input background removed images folder"),
    background_removed_manifest: Path = typer.Argument(..., help="Input background removed manifest"),
    transcribed_folder: Path = typer.Argument(..., help="Output folder for transcriptions"),
    testing: bool = typer.Option(False, help="Run on a small subset of data"),
):
    """Batch transcription CLI using Qwen VL Max model"""
    print(f"[green]Transcribing images in {background_removed_folder}")
    print(f"[cyan]Using model qvq-max")
    
    load_dotenv()
    
    if not os.getenv('DASHSCOPE_API_KEY'):
        print("[red]Error: DASHSCOPE_API_KEY environment variable not set")
        return

    processor = BatchProcessor(
        input_manifest=background_removed_manifest,
        output_folder=transcribed_folder,
        process_name="transcribe_qwen_max",
        processor_fn=lambda f, o: process_document(f, o),
        base_folder=background_removed_folder / "documents"
    )
    
    return processor.process()

if __name__ == "__main__":
    typer.run(transcribe) 