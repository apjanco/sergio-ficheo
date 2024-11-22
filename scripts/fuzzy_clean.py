import typer
from pathlib import Path
from rich.progress import track
import re
# from fuzzywuzzy import fuzz

def calculate_max_phrase_length(text, percentage=0.5):
    """Calculate the maximum phrase length based on a percentage of the total word count."""
    word_count = len(text.split())
    return max(1, int(word_count * percentage))

def clean_repeated_phrases(text):
    """Remove repeated phrases from the text."""
    max_phrase_length = calculate_max_phrase_length(text, percentage=0.05)
    lines = text.splitlines()
    clean_lines = []

    for line in lines:
        words = line.split()
        clean_line = []
        i = 0

        while i < len(words):
            phrase = " ".join(words[i:i + max_phrase_length])
            next_phrase = " ".join(words[i + 1:i + 1 + max_phrase_length])
            # Skip short phrases (e.g., two words that are each two letters long)
            if len(phrase.split()) == 2 and all(len(word) <= 2 for word in phrase.split()):
                clean_line.append(words[i])
            elif phrase != next_phrase:
                clean_line.append(words[i])
            i += 1

        clean_lines.append(" ".join(clean_line))

    return "\n".join(clean_lines)

def remove_repeated_phrases(text, min_phrase_length=5):
    """Remove long repeated phrases from the text."""
    lines = text.splitlines()
    clean_lines = []
    previous_phrases = set()

    for line in lines:
        words = line.split()
        clean_line = []
        i = 0

        while i < len(words):
            phrase = " ".join(words[i:i + min_phrase_length])
            if phrase not in previous_phrases:
                clean_line.append(" ".join(words[i:i + min_phrase_length]))
                previous_phrases.add(phrase)
            i += min_phrase_length

        clean_lines.append(" ".join(clean_line))

    return "\n".join(clean_lines)

def remove_repeated_words(text):
    """Remove repeated words from the text."""
    lines = text.splitlines()
    clean_lines = []

    for line in lines:
        words = line.split()
        clean_line = []
        previous_word = ""

        for word in words:
            if word.lower() != previous_word.lower():
                clean_line.append(word)
            previous_word = word

        clean_lines.append(" ".join(clean_line))

    return "\n".join(clean_lines)

def remove_repeated_phrases_between_chunks(text):
    """Remove repeated phrases that might appear between chunks."""
    lines = text.splitlines()
    clean_lines = []
    previous_line = ""

    for line in lines:
        # if fuzz.ratio(previous_line, line) < fuzziness_threshold:
        if previous_line != line:
            clean_lines.append(line)
        previous_line = line

    return "\n".join(clean_lines)

def remove_repeated_phrases_regex(text):
    """Remove repeated phrases and numbers at the beginning of the line using regex."""
    # Pattern to match repeated phrases
    pattern = re.compile(r"(\b\w+\b(?:\s+\b\w+\b){2,})(?=.*\1)")
    text = re.sub(pattern, "", text)
    
    # Pattern to match numbers at the beginning of the line
    text = re.sub(r"^\d+\s+", "", text, flags=re.MULTILINE)
    
    return text

def combine_single_word_paragraphs(text):
    """Combine single-word paragraphs into a single line."""
    lines = text.splitlines()
    combined_lines = []
    current_line = []

    for line in lines:
        if len(line.split()) == 1:
            current_line.append(line)
        else:
            if current_line:
                combined_lines.append(" ".join(current_line))
                current_line = []
            combined_lines.append(line)

    if current_line:
        combined_lines.append(" ".join(current_line))

    return "\n".join(combined_lines)

def fuzzy_clean(
    transcribed_folder: Path = typer.Argument(..., help="Path to the transcribed files", exists=True),
    cleaned_folder: Path = typer.Argument(..., help="Output folder for cleaned files")
):
    if not cleaned_folder.exists():
        cleaned_folder.mkdir(parents=True, exist_ok=True)

    transcribed_files = list(transcribed_folder.glob("**/*.md"))
    
    for transcribed_file in track(transcribed_files, description="Cleaning transcriptions..."):
        text = transcribed_file.read_text()
        
        # Apply cleaning rules
        text = re.sub(r"\(\d+,\d+\),\(\d+,\d+\)", "", text)  # Remove coordinates
        text = re.sub(r"\b(blank|The text is not visible in the image\.|The text on the image is not clear and appears to be a mix of different colors and patterns\. It is difficult to extract any meaningful information from it\.)\b", "", text, flags=re.IGNORECASE)
        
        # Combine single-word paragraphs
        text = combine_single_word_paragraphs(text)
        
        # Clean repeated phrases within the same chunk
        text = clean_repeated_phrases(text)
        
        # Clean long repeated phrases
        text = remove_repeated_phrases(text)
        
        # Clean repeated words
        text = remove_repeated_words(text)
        
        # Clean repeated phrases between chunks
        text = remove_repeated_phrases_between_chunks(text)
        
        # Apply regex-based cleaning to catch repeated phrases and numbers at the beginning of the line
        text = remove_repeated_phrases_regex(text)
        
        # Apply cleaning functions again to catch section-level repetition
        text = clean_repeated_phrases(text)
        text = remove_repeated_phrases(text)
        text = remove_repeated_words(text)
        text = remove_repeated_phrases_between_chunks(text)
        
        cleaned_file = cleaned_folder / transcribed_file.relative_to(transcribed_folder)
        cleaned_file.parent.mkdir(parents=True, exist_ok=True)
        cleaned_file.write_text(text.strip())

if __name__ == "__main__":
    typer.run(fuzzy_clean)