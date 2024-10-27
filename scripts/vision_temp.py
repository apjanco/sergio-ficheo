"""OCR with PDF/TIFF as source files on GCS"""

# USAGE: python text_detect.py SOURCE_FILE OUTPUT_FILE
#   Note that both SOURCE_FILE and OUTPUT_FILE must be
#   in the Google Cloud bucket. For example:
#
#   python text_detect.py gs://project-name/file.pdf gs://project-name/read
#
# The API will gather the responses for each page into
#   a JSON file on the Google Cloud bucket, e.g.
#   OUTPUT_FILE-output-1-to-1.json.
#
# This script will then take the recognized text from
#   these JSON files and assemble them into a text document
#   with the name OUTPUT_FILE.txt in the same directory where
#   the script is run.

# Note that you must pass the application credentials
#   so that Google Cloud Vision knows which project
#   to use:
#

# This script is based on what Google suggests at
#   https://cloud.google.com/vision/docs/samples/vision-batch-annotate-files

import sys
import io
import os
from pathlib import Path
from google.cloud import vision
from tqdm import tqdm

mime_type = "application/pdf"

pdf_files = Path().glob("*.pdf")
for pdf_file in tqdm(pdf_files):

    client = vision.ImageAnnotatorClient()
    content = pdf_file.read_bytes()
    input_config = {"mime_type": mime_type, "content": content}
    features = [{"type_": vision.Feature.Type.DOCUMENT_TEXT_DETECTION}]

    requests = [{"input_config": input_config, "features": features}]

    response = client.batch_annotate_files(requests=requests)

    for i, image_response in enumerate(response.responses[0].responses):
        print(f"{image_response.full_text_annotation.text}")
        pdf_file.with_suffix(f".{i}.txt").write_text(
            image_response.full_text_annotation.text
        )

print("DONE!")
