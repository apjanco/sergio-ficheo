import srsly
from pathlib import Path
import csv
import typer

app = typer.Typer()

def strip_quotes(text):
    return text.strip().strip('"')

def clean_entities(entities, label):
    return '; '.join(ent['text'].replace('\n', ' ') for ent in entities if ent['label'] == label)

def export_to_csv(json_file: Path, output_folder: Path):
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    data = list(srsly.read_jsonl(json_file))
    csv_file = output_folder / "exported_data.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        headers = ["image", "original_text", "cleaned_text", "english_translation", "summary", "persons", "locations", "organizations"]
        writer.writerow(headers)
        for item in data:
            row = {
                "image": item.get('image', 'No image available'),
                "original_text": strip_quotes(item.get('text', 'No original text available')),
                "cleaned_text": strip_quotes(item.get('cleaned_text', 'No cleaned text available')),
                "english_translation": strip_quotes(item.get('english_translation', 'No translation available')),
                "summary": strip_quotes(item.get('summary', 'No summary available')),
                "persons": clean_entities(item.get('entities', []), 'PER'),
                "locations": clean_entities(item.get('entities', []), 'LOC'),
                "organizations": clean_entities(item.get('entities', []), 'ORG')
            }
            writer.writerow(row.values())
    print(f"CSV file saved as {csv_file}")

@app.command()
def main(
    json_file: Path = typer.Argument(..., help="Path to the JSONL file"),
    output_folder: Path = typer.Argument(..., help="Path to the output folder for CSV file")
):
    export_to_csv(json_file, output_folder)

if __name__ == "__main__":
    typer.run(main)