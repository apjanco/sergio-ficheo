from pathlib import Path
from typing import Callable, Dict, List, Optional
from rich.console import Console
from .manifest import ManifestProcessor
from .progress import ProgressTracker
import sys

console = Console()

class BatchProcessor:
    """Handles batch processing of files with progress tracking and manifest management"""
    
    def __init__(
        self,
        input_manifest: Path,
        output_folder: Path,
        process_name: str,
        processor_fn: Callable,
        batch_size: int = 100,
        base_folder: Path = None
    ):
        self.input_manifest = Path(input_manifest)
        self.output_folder = Path(output_folder)
        self.base_folder = Path(base_folder) if base_folder else None
        self.process_name = process_name
        self.processor_fn = processor_fn
        self.batch_size = batch_size
        
        # Setup folders and files
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.output_folder / f"{process_name}_manifest.jsonl"
        self.progress_file = self.output_folder / f"{process_name}_progress.jsonl"
        
        # Initialize manifest processors
        self.input_proc = ManifestProcessor(manifest_path=self.input_manifest, progress_file=None)
        self.output_proc = ManifestProcessor(manifest_path=self.manifest_file, progress_file=self.progress_file)
        
    def process(self) -> Dict:
        """Run the batch processing"""
        documents = []
        skipped_count = 0
        
        console.print("Checking files to process...")
        console.print(f"Base folder: {self.base_folder}")
        console.print(f"Input manifest: {self.input_manifest}")
        console.print(f"Output folder: {self.output_folder}")
        
        for doc in self.input_proc.stream_entries():
            # For initial document processing, look for type: file entries
            if doc.get("type") == "file":
                source_path = doc.get("path", "")
                if source_path:
                    # Check if already processed using clean path
                    clean_path = str(Path(source_path))
                    if clean_path in self.output_proc.entries:
                        skipped_count += 1
                        continue
                    documents.append({"path": source_path})
            # For subsequent processing (crops -> splits), look for outputs
            elif "outputs" in doc:
                for output_path in doc["outputs"]:
                    if output_path in self.output_proc.entries:
                        skipped_count += 1
                        continue
                    documents.append({"path": output_path})

        total_files = len(documents)
        stats = {
            "total": total_files + skipped_count,
            "skipped": skipped_count,
            "processed": 0,
            "failed": 0
        }

        console.print(f"\nTotal files: {stats['total']}")
        console.print(f"Already processed: {stats['skipped']}")
        console.print(f"To process: {total_files}\n")

        if total_files == 0:
            return stats

        # Setup progress tracking
        tracker = ProgressTracker(
            total=total_files,
            task_name=f"{self.process_name.title()} files",
            progress_fields=stats
        )

        try:
            with tracker.progress as progress:
                current_batch = []
                
                for doc in documents:
                    current_batch.append(doc)
                    
                    if len(current_batch) >= self.batch_size:
                        self._process_batch(current_batch, stats, progress, tracker.task)
                        current_batch = []
                        self.output_proc.write_progress(stats)

                # Process remaining files
                if current_batch:
                    self._process_batch(current_batch, stats, progress, tracker.task)

        except KeyboardInterrupt:
            console.print("\n[yellow]Processing interrupted by user. Saving progress...")
            self.output_proc._write_manifest(self.manifest_file)  # Save manifest
            self.output_proc.write_progress(stats)  # Save progress
            sys.exit(1)
        except Exception as e:
            console.print(f"\n[red]Error occurred: {e}")
            self.output_proc.write_progress(stats)
            raise

        self.output_proc.write_progress(stats)
        self._print_stats(stats)
        return stats

    def _process_batch(self, batch: List[dict], stats: dict, progress, task):
        """Process a batch of files"""
        for doc in batch:
            try:
                path = Path(doc["path"])
                # Remove any 'documents' prefix for consistency
                if 'documents' in path.parts:
                    path = Path(*path.parts[path.parts.index('documents') + 1:])
                
                # Use base_folder/documents for actual file operations
                full_path = self.base_folder / path if self.base_folder else path
                result = self.processor_fn(str(full_path), self.output_folder)
                
                # Store just the clean relative path
                result["source"] = str(path)
                self.output_proc.save_entry(result)
                
                if result.get("skipped"):
                    stats["skipped"] += 1
                elif result.get("error"):
                    stats["failed"] += 1
                else:
                    stats["processed"] += 1
                    
                progress.update(task, advance=1, **stats)
                
            except Exception as e:
                console.print(f"[red]Error processing {doc['path']}: {e}")
                stats["failed"] += 1
                progress.update(task, advance=1, **stats)

    def _print_stats(self, stats: dict):
        """Print final statistics"""
        console.print(f"\n[green]Processing completed. Final statistics:")
        console.print(f"Processed: {stats['processed']}")
        console.print(f"Skipped: {stats['skipped']}")
        console.print(f"Failed: {stats['failed']}")
