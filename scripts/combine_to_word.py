import srsly
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.section import WD_ORIENT, WD_SECTION
from PIL import Image
import typer
import docx.oxml.shared
from docx.enum.text import WD_ALIGN_PARAGRAPH

app = typer.Typer()

def set_page_size(doc, width, height, left_margin, right_margin, top_margin, bottom_margin):
    section = doc.sections[-1]
    section.page_width = Inches(width)
    section.page_height = Inches(height)
    section.orientation = WD_ORIENT.PORTRAIT
    section.left_margin = Inches(left_margin)
    section.right_margin = Inches(right_margin)
    section.top_margin = Inches(top_margin)
    section.bottom_margin = Inches(bottom_margin)

def combine_to_word(json_file: Path, image_folder: Path, output_folder: Path):
    data = list(srsly.read_jsonl(json_file))  # Process all items
    doc_dict = {
        "SM_NPQ_C01_": Document(),
        "SM_NPQ_C02_": Document(),
        "SM_NPQ_C03_": Document(),
        "SM_NPQ_C04_": Document(),
        "SM_NPQ_C05_": Document()
    }

    for doc in doc_dict.values():
        set_page_size(doc, 10, 11, 0.125, 0, 0, 0)

    for item in data:
        image_path = image_folder / item.get('image', '')
        if not image_path.exists():
            print(f"Image {image_path} not found. Skipping.")
            continue

        stem_parts = image_path.stem.split('_')
        stem = f"{stem_parts[0]}_{stem_parts[1]}_{stem_parts[2]}_"
        if stem not in doc_dict:
            print(f"Image {image_path} does not match any known stem. Skipping.")
            continue

        doc = doc_dict[stem]

        # Add image to the left-hand page
        image = Image.open(image_path)
        width, height = image.size
        aspect_ratio = width / height
        max_width = Inches(10) - Inches(0.125)  # Max width considering left margin
        max_height = Inches(11)
        if (aspect_ratio > max_width / max_height):
            doc.add_picture(str(image_path), width=max_width)
        else:
            doc.add_picture(str(image_path), height=max_height)
        
        # Add a new section for the right-hand page
        new_section = doc.add_section(WD_SECTION.NEW_PAGE)
        set_page_size(doc, 10, 11, 0.5, 0.5, 0.5, 0.5)

        # Add text boxes to the right-hand page
        table = doc.add_table(rows=4, cols=1)
        table.style = 'Table Grid'

        # Remove table borders
        for row in table.rows:
            for cell in row.cells:
                for border in cell._element.xpath('.//w:tcBorders/*'):
                    border.attrib.clear()

        # Center the table
        for row in table.rows:
            for cell in row.cells:
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Add text
        text_cell = table.cell(0, 0)
        text_cell.text = item.get('text', 'No text available')

        # Add translation
        translation_cell = table.cell(1, 0)
        translation_cell.text = item.get('translation', 'No translation available')

        # Add summary and NER data
        summary_ner_cell = table.cell(2, 0)
        summary = item.get('summary', 'No summary available')
        ner_data = item.get('ner', 'No NER data available')
        summary_ner_cell.text = f"Summary: {summary}\nNER Data: {ner_data}"

        # Add file name
        file_name_cell = table.cell(3, 0)
        file_name_cell.text = f"File Name: {image_path.name}"

        doc.add_page_break()

    for stem, doc in doc_dict.items():
        output_file = output_folder / f"{stem[:-1]}.docx"
        doc.save(output_file)
        print(f"Document saved as {output_file}")

@app.command()
def main(
    json_file: Path = typer.Argument(..., help="Path to the JSONL file"),
    image_folder: Path = typer.Argument(..., help="Path to the folder containing images"),
    output_folder: Path = typer.Argument(..., help="Path to the output folder for Word files")
):
    combine_to_word(json_file, image_folder, output_folder)

if __name__ == "__main__":
    typer.run(main)