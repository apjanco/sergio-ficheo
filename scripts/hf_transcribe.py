import typer
from rich import print
from rich.progress import track
from typing_extensions import Annotated
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from pathlib import Path 
from PIL import Image
from datasets import load_dataset

def transcribe(
    dataset: Annotated[str, typer.Argument(help="HF Dataset name")],
    model_name: Annotated[str, typer.Argument(help="HF model name")],
):
    prompt = """extract text. identify relevant entities. return result as json. Example: {"date": "10/28/43", "text":"I went to the store.","ents":["store","cheese"]}"""
    print(f"[green]Transcribing images in {dataset}")
    print(f"[cyan]Using model {model_name}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    ds = load_dataset(dataset)
    images = ds['train']
    # create a data folder 
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    for image_row in track(images, total=images.shape[0]):
            if (data_dir / image_row['name']).with_suffix(".md").exists():
                continue
            image = image_row['image']
            image.thumbnail((1000,1000))
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
            generated_ids = model.generate(**inputs, max_new_tokens=1000)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            (data_dir / image_row['name']).with_suffix(".md").write_text(output_text[0])


if __name__ == "__main__":
    typer.run(transcribe)