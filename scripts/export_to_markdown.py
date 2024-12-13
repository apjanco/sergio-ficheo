import srsly
from pathlib import Path
import typer
import re
import shutil

app = typer.Typer()

def strip_quotes(text):
    return text.strip().strip('"')

label_map = {
    'PER': 'person',
    'LOC': 'location',
    'ORG': 'organization'
}

def clean_entities(entities, label, doc_name):
    if label not in label_map:
        return ''  # Skip unknown labels
    if isinstance(entities, dict):
        entities = entities.get('Entities', [])
    return '\n'.join(f"[[/{label_map[label]}/{sanitize_filename(ent['text'])}]]" for ent in entities if 'label' in ent and ent['label'] == label)

def sanitize_filename(name):
    # Remove invalid characters and limit length
    name = re.sub(r'[^\w\s-]', '_', name)
    name = name[:100]
    return name  # Do not URL-encode the filename

def export_to_markdown(json_file: Path, output_folder: Path, image_folder: Path, adjusted_image_folder: Path):
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    data = list(srsly.read_jsonl(json_file))
    entity_mentions = {label: {} for label in label_map.values()}

    for i, item in enumerate(data):
        file_name = sanitize_filename(item.get('image', 'No_image_available').replace('/', '_').replace(' ', '_')) + '.md'
        file_path = output_folder / file_name
        image_path = adjusted_image_folder / Path(item.get('image', 'No_image_available')).name
        image_dest_path = image_folder / image_path.name
        if not image_dest_path.exists():
            shutil.copy(image_path, image_dest_path)
        relative_image_path = f"../images/{image_path.name}"
        prev_page = f"[Previous]({sanitize_filename(data[i-1].get('image', 'No_image_available').replace('/', '_').replace(' ', '_'))}.md)" if i > 0 else ""
        next_page = f"[Next]({sanitize_filename(data[i+1].get('image', 'No_image_available').replace('/', '_').replace(' ', '_'))}.md)" if i < len(data) - 1 else ""
        with open(file_path, mode='w', encoding='utf-8') as file:
            file.write(f"{prev_page} {next_page}\n\n")
            file.write(f"# [{item.get('image', 'No image available')}]({relative_image_path})\n\n")
            file.write(f"## Original Text\n{strip_quotes(item.get('text', 'No original text available'))}\n\n")
            file.write(f"## Cleaned Text\n{strip_quotes(item.get('cleaned_text', 'No cleaned text available'))}\n\n")
            file.write(f"## English Translation\n{strip_quotes(item.get('english_translation', 'No translation available'))}\n\n")
            file.write(f"## Summary\n{strip_quotes(item.get('summary', 'No summary available'))}\n\n")
            file.write(f"## Persons\n{clean_entities(item.get('entities', []), 'PER', file_name)}\n\n")
            file.write(f"## Locations\n{clean_entities(item.get('entities', []), 'LOC', file_name)}\n\n")
            file.write(f"## Organizations\n{clean_entities(item.get('entities', []), 'ORG', file_name)}\n\n")
        print(f"Markdown file saved as {file_path}")

        # Collect entity mentions
        entities = item.get('entities', [])
        if isinstance(entities, dict):
            entities = entities.get('Entities', [])
        for ent in entities:
            if 'label' not in ent or ent['label'] not in label_map:
                continue  # Skip unknown labels
            entity_label = label_map[ent['label']]
            entity_text = sanitize_filename(ent['text'].replace(chr(10), ' '))
            if entity_text not in entity_mentions[entity_label]:
                entity_mentions[entity_label][entity_text] = []
            entity_mentions[entity_label][entity_text].append(file_name)

    # Create/update entity files
    for label, entities in entity_mentions.items():
        entity_folder = output_folder / label
        if not entity_folder.exists():
            entity_folder.mkdir(parents=True, exist_ok=True)
        for entity_text, doc_names in entities.items():
            entity_file_path = entity_folder / f"{entity_text}.md"
            with open(entity_file_path, mode='w', encoding='utf-8') as entity_file:
                entity_file.write(f"# {entity_text}\n\n")
                entity_file.write("## Mentioned in:\n")
                for doc_name in doc_names:
                    entity_file.write(f"- [[../{doc_name}]]\n")
            print(f"Entity file saved as {entity_file_path}")

@app.command()
def main(
    json_file: Path = typer.Argument(..., help="Path to the JSONL file"),
    output_folder: Path = typer.Argument(..., help="Path to the output folder for Markdown files"),
    image_folder: Path = typer.Argument(..., help="Path to the folder for images"),
    adjusted_image_folder: Path = typer.Argument(..., help="Path to the folder for adjusted images")
):
    export_to_markdown(json_file, output_folder, image_folder, adjusted_image_folder)

if __name__ == "__main__":
    typer.run(main)