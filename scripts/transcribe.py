import typer
from rich import print
from rich.progress import track
from typing_extensions import Annotated
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from pathlib import Path 
from PIL import Image

def transcribe(
    collection_path: Annotated[Path, typer.Argument(help="Path to the collections", exists=True)],
    model_name: Annotated[str, typer.Argument(help="HF model name")],
    testing: bool = typer.Option(False, help="Run on a small subset of data"),
):
    prompt = """extract all text. keep formatting as markdown"""
    print(f"[green]Transcribing images in {collection_path}")
    print(f"[cyan]Using model {model_name}")
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    images = list(collection_path.glob("**/*.jpg"))
    if testing:
        images = images[:10]
    for image_path in track(images, total=len(images)):
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((1000, 1000))
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

            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=800)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if testing:
                print(f"Image: {image_path}")
                print(f"Output: {output_text[0]}")
                (data_dir / image_path.name).with_suffix(".testing.md").write_text(output_text[0])
            else:
                (data_dir / image_path.name).with_suffix(".md").write_text(output_text[0])


if __name__ == "__main__":
    typer.run(transcribe)