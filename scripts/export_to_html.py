import srsly
from pathlib import Path
import typer
import re
import shutil

app = typer.Typer()

def strip_quotes(text):
    return text.strip().strip('"')

def convert_newlines_to_paragraphs(text):
    paragraphs = text.split('\n')
    return ''.join(f'<p>{para}</p>' for para in paragraphs if para.strip())

label_map = {
    'PER': 'person',
    'LOC': 'location',
    'ORG': 'organization'
}

def clean_entities(entities, label, doc_name):
    if label not in label_map:
        return ''  # Skip unknown labels
    return '\n'.join(f'<a href="./{label_map[label]}/{sanitize_filename(ent["text"])}.html">{ent["text"]}</a>' for ent in entities if ent['label'] == label)

def sanitize_filename(name):
    # Remove invalid characters and limit length
    name = re.sub(r'[^\w\s-]', '_', name)
    name = name[:100]
    return name  # Do not URL-encode the filename

def export_to_html(json_file: Path, output_folder: Path, image_folder: Path, adjusted_image_folder: Path):
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    # Copy the CSS and JS files to the output folder
    shutil.copy('static/styles.css', output_folder / 'styles.css')
    shutil.copy('static/scripts.js', output_folder / 'scripts.js')

    data = list(srsly.read_jsonl(json_file))
    entity_mentions = {label: {} for label in label_map.values()}

    for i, item in enumerate(data):
        file_name = sanitize_filename(item.get('image', 'No_image_available').replace('/', '_').replace(' ', '_')) + '.html'
        file_path = output_folder / file_name
        image_path = adjusted_image_folder / Path(item.get('image', 'No_image_available')).name
        image_dest_path = image_folder / image_path.name
        if not image_dest_path.exists():
            shutil.copy(image_path, image_dest_path)
        relative_image_path = f"images/{image_path.name}"
        prev_page = f'<a href="{sanitize_filename(data[i-1].get("image", "No_image_available").replace("/", "_").replace(" ", "_"))}.html">Previous</a>' if i > 0 else ""
        next_page = f'<a href="{sanitize_filename(data[i+1].get("image", "No_image_available").replace("/", "_").replace(" ", "_"))}.html">Next</a>' if i < len(data) - 1 else ""
        html_content = f"""
            <html>
            <head>
                <meta charset="UTF-8">
                <title>{item.get('image', 'No image available')}</title>
                <link rel="stylesheet" type="text/css" href="styles.css">
                <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/5.0.0/openseadragon.min.js" integrity="sha512-mdfzGJn9wUFg72mwblmP0uA6j3uB3uEKOQB1gmCCsnsKQNQRys+mITew+5lPFIo0C4NU1Bi/O+Eaw2GMsjC9IA==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
                <script src="scripts.js"></script>
            </head>
            <body>
                <div class="container">
                    <div id="fmb-image" class="openseadragon"></div>
                    <div class="content">
                        <div>
                            {prev_page} {next_page}
                        </div>
                        <h1>{item.get('image', 'No image available')}</h1>
                        <h2>Original Text</h2>{convert_newlines_to_paragraphs(strip_quotes(item.get('text', 'No original text available')))}
                        <h2>Cleaned Text</h2>{convert_newlines_to_paragraphs(strip_quotes(item.get('cleaned_text', 'No cleaned text available')))}
                        <h2>English Translation</h2>{convert_newlines_to_paragraphs(strip_quotes(item.get('english_translation', 'No translation available')))}
                        <h2>Summary</h2>{convert_newlines_to_paragraphs(strip_quotes(item.get('summary', 'No summary available')))}
                        <h2>Persons</h2><p>{clean_entities(item.get('entities', []), 'PER', file_name)}</p>
                        <h2>Locations</h2><p>{clean_entities(item.get('entities', []), 'LOC', file_name)}</p>
                        <h2>Organizations</h2><p>{clean_entities(item.get('entities', []), 'ORG', file_name)}</p>
                    </div>
                </div>
                <script type="text/javascript">
                    OpenSeadragon({{
                        id: "fmb-image",
                        prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/5.0.0/images/",
                        tileSources: {{
                            type: 'image',
                            url: '{relative_image_path}'
                        }},
                        showNavigator: true,
                        controlsFadeDelay: 0,
                        controlsFadeLength: 0,
                        showRotationControl: true,
                        showFullPageControl: true,
                        showHomeControl: true,
                        showZoomControl: true,
                        showSequenceControl: true,
                        maxZoomLevel: 10 // Allow zooming up to 1000%
                    }});
                </script>
            </body>
            </html>
            """
        with open(file_path, mode='w', encoding='utf-8') as file:
            file.write(html_content)
        print(f"HTML file saved as {file_path}")

        # Collect entity mentions
        for ent in item.get('entities', []):
            if ent['label'] not in label_map:
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
            entity_file_path = entity_folder / f"{entity_text}.html"
            entity_html_content = f"""
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>{entity_text}</title>
                    <link rel="stylesheet" type="text/css" href="../styles.css">
                    <script src="../scripts.js"></script>
                </head>
                <body>
                    <h1>{entity_text}</h1>
                    <h2>Mentioned in:</h2>
                    <ul>
            """
            for doc_name in doc_names:
                entity_html_content += f"<li><a href='../{doc_name}'>{doc_name}</a></li>"
            entity_html_content += """
                    </ul>
                </body>
                </html>
            """
            with open(entity_file_path, mode='w', encoding='utf-8') as entity_file:
                entity_file.write(entity_html_content)
            print(f"Entity file saved as {entity_file_path}")

@app.command()
def main(
    json_file: Path = typer.Argument(..., help="Path to the JSONL file"),
    output_folder: Path = typer.Argument(..., help="Path to the output folder for HTML files"),
    image_folder: Path = typer.Argument(..., help="Path to the folder for images"),
    adjusted_image_folder: Path = typer.Argument(..., help="Path to the folder for adjusted images")
):
    export_to_html(json_file, output_folder, image_folder, adjusted_image_folder)

if __name__ == "__main__":
    typer.run(main)