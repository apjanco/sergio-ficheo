import typer
from pathlib import Path
from rich.console import Console
import re
from utils.batch import BatchProcessor
from utils.processor import process_file

console = Console()

class TextCleaner:
    @staticmethod
    def calculate_max_phrase_length(text, percentage=0.5):
        """Calculate the maximum phrase length based on percentage of total word count."""
        word_count = len(text.split())
        return max(1, int(word_count * percentage))

    @staticmethod
    def clean_repeated_phrases(text):
        """Remove repeated phrases from the text."""
        max_phrase_length = TextCleaner.calculate_max_phrase_length(text, percentage=0.05)
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def remove_repeated_phrases_regex(text):
        """Remove repeated phrases and numbers at the beginning of the line using regex."""
        # Pattern to match repeated phrases
        pattern = re.compile(r"(\b\w+\b(?:\s+\b\w+\b){2,})(?=.*\1)")
        text = re.sub(pattern, "", text)
        
        # Pattern to match numbers at the beginning of the line
        text = re.sub(r"^\d+\s+", "", text, flags=re.MULTILINE)
        
        return text

    @staticmethod
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

    @staticmethod
    def remove_specific_phrases(text: str) -> str:
        """Remove specific unwanted phrases from the text."""
        phrases_to_remove = [
            "The text on the document is:",
            "The text on the document reads:",
            "The document reads:",
            "The text reads:",
            "Here is the text extracted from the image:",
            "The text on the image is as follows:",
            "Please note that this is an incomplete sentence, as the rest of the text has been cut off.",
            "```",
            "```plaintext"
        ]
        
        for phrase in phrases_to_remove:
            # Handle variations in whitespace and line breaks
            pattern = phrase.replace(" ", r'\s+')
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
        return text.strip()

    @staticmethod
    def calculate_average_line_length(text: str) -> int:
        """Calculate average line length for paragraph-like lines."""
        lines = text.splitlines()
        # Only consider lines that look like paragraphs (more than 30 chars)
        paragraph_lines = [line for line in lines if len(line) > 30]
        if not paragraph_lines:
            return 72  # Default fallback
        return int(sum(len(line) for line in paragraph_lines) / len(paragraph_lines))

    @staticmethod
    def split_long_lines(text: str, max_length: int = 72) -> str:
        """Simple line wrapper at max_length characters."""
        lines = text.splitlines()
        wrapped_lines = []
        
        for line in lines:
            if len(line) <= max_length:
                wrapped_lines.append(line)
                continue
                
            # Split long line into chunks
            current_line = ""
            words = line.split()
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                if len(test_line) <= max_length:
                    current_line = test_line
                else:
                    if current_line:
                        wrapped_lines.append(current_line)
                    current_line = word
            
            if current_line:
                wrapped_lines.append(current_line)
                
        return "\n".join(wrapped_lines)  # Fixed the string literal here

    @staticmethod
    def remove_boundary_quotes(text: str) -> str:
        """Remove quotes at the beginning and end of the document."""
        text = text.strip()
        if text.startswith('"'):
            text = text[1:]
        if text.endswith('"'):
            text = text[:-1]
        return text.strip()

    @staticmethod
    def clean_text(text: str) -> str:
        """Apply all cleaning steps to the text"""
        # First remove all specific phrases and boundary characters
        text = TextCleaner.remove_specific_phrases(text)
        text = TextCleaner.remove_boundary_quotes(text)
        
        # Then remove coordinates and other patterns
        coordinate_patterns = [
            r"\(\d+,\d+\),\(\d+,\d+\)",  # (123,456),(789,012)
            r"\(\d+,\d+\), \(\d+,\d+\)",  # (123,456), (789,012)
            r"\(\d+, \d+\),\(\d+, \d+\)",  # (123, 456),(789, 012)
            r"\(\d+, \d+\), \(\d+, \d+\)",  # (123, 456), (789, 012)
        ]
        for pattern in coordinate_patterns:
            text = re.sub(pattern, "", text)
            
        text = re.sub(r"\b(blank|The text is not visible in the image\.|The text on the image is not clear and appears to be a mix of different colors and patterns\. It is difficult to extract any meaningful information from it\.)\b", "", text, flags=re.IGNORECASE)
        
        # Remove specific phrases first
        text = TextCleaner.remove_specific_phrases(text)
        
        # Remove quotes at document boundaries
        text = TextCleaner.remove_boundary_quotes(text)
        
        # Apply cleaning steps
        text = TextCleaner.combine_single_word_paragraphs(text)
        text = TextCleaner.clean_repeated_phrases(text)
        text = TextCleaner.remove_repeated_phrases(text)
        text = TextCleaner.remove_repeated_words(text)
        text = TextCleaner.remove_repeated_phrases_between_chunks(text)
        text = TextCleaner.remove_repeated_phrases_regex(text)
        
        # Calculate average line length and split long lines
        avg_length = TextCleaner.calculate_average_line_length(text)
        max_length = min(avg_length * 1.5, 72)  # Use shorter of avg length or 72
        text = TextCleaner.split_long_lines(text, int(max_length))
        
        # Second pass to catch section-level repetition
        text = TextCleaner.clean_repeated_phrases(text)
        text = TextCleaner.remove_repeated_phrases(text)
        text = TextCleaner.remove_repeated_words(text)
        text = TextCleaner.remove_repeated_phrases_between_chunks(text)
        
        return text.strip()

def process_document(file_path: str, output_folder: Path) -> dict:
    """Process a single document file"""
    try:
        # Convert to Path and normalize
        source_path = Path(file_path)
        
        # Get relative path from documents/
        rel_path = source_path
        if 'documents' in source_path.parts:
            rel_path = Path(*source_path.parts[source_path.parts.index('documents')+1:])
        
        # Look for input .md file
        input_path = Path(file_path).with_suffix('.md')
        if not input_path.exists():
            # Try with documents prefix if not found
            input_path = output_folder.parent / "recombined/documents" / rel_path.with_suffix('.md')
            
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        # Read input text
        try:
            text = input_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            text = input_path.read_text()
        
        if not text.strip():
            return {
                "source": str(input_path),
                "error": "Empty file",
                "success": False
            }
        
        # Clean the text
        cleaned_text = TextCleaner.clean_text(text)
        
        # Use relative path for output
        out_path = output_folder / "documents" / rel_path.with_suffix('.md')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write cleaned text
        out_path.write_text(cleaned_text)
        
        # Return success manifest entry with proper relative paths
        return {
            "source": str(rel_path.with_suffix('.md')),  # Relative from documents/
            "outputs": [str(rel_path.with_suffix('.md'))],  # Relative from documents/
            "success": True,
            "details": {
                "original_length": len(text),
                "cleaned_length": len(cleaned_text),
                "reduction_percent": round((1 - len(cleaned_text)/len(text)) * 100, 2)
            }
        }
        
    except Exception as e:
        console.print(f"[red]Error processing {file_path}: {e}")
        return {
            "source": str(file_path),
            "error": str(e)
        }

def fuzzy_clean(
    recombined_folder: Path = typer.Argument(..., help="Path to the recombined files"),
    recombined_manifest: Path = typer.Argument(..., help="Path to the recombined manifest file"),
    cleaned_folder: Path = typer.Argument(..., help="Output folder for cleaned files")
):
    """Clean up text from recombined transcriptions"""
    
    # Validate inputs
    if not recombined_folder.exists():
        raise typer.BadParameter(f"Recombined folder not found: {recombined_folder}")
    if not recombined_manifest.exists():
        raise typer.BadParameter(f"Recombined manifest not found: {recombined_manifest}")
        
    processor = BatchProcessor(
        input_manifest=recombined_manifest,
        output_folder=cleaned_folder,
        process_name="fuzzy_clean",
        processor_fn=lambda f, o: process_document(f, o),
        base_folder=recombined_folder,
        use_source=True  # Use source path from manifest since we're processing MD files
    )
    
    return processor.process()

if __name__ == "__main__":
    typer.run(fuzzy_clean)