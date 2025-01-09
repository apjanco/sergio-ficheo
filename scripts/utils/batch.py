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
        base_folder: Path = None,
        use_source: bool = False
    ):
        self.input_manifest = Path(input_manifest)
        self.output_folder = Path(output_folder)
        self.base_folder = Path(base_folder) if base_folder else None
        self.process_name = process_name
        self.processor_fn = processor_fn
        self.batch_size = batch_size
        self.use_source = use_source
        
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
            if doc.get("type") == "directory":
                continue

            # Handle both single files and chunks
            paths_to_process = []
            if self.use_source and "source" in doc:
                paths_to_process.append(doc["source"])
            elif "outputs" in doc and doc["outputs"]:
                for out_path in doc["outputs"]:
                    if isinstance(out_path, str):
                        paths_to_process.append(out_path)
                    elif isinstance(out_path, dict) and "path" in out_path:
                        paths_to_process.append(out_path["path"])

            # Add valid paths that haven't been processed
            for path in paths_to_process:
                if path not in self.output_proc.entries:
                    documents.append({"path": path})
                else:
                    skipped_count += 1

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
                # Process in batches
                for i in range(0, len(documents), self.batch_size):
                    batch = documents[i:i + self.batch_size]
                    self._process_batch(batch, stats, progress, tracker.task)
                    # Save progress periodically
                    if i % (self.batch_size * 5) == 0:
                        self.output_proc._write_manifest(self.manifest_file)
                        self.output_proc.write_progress(stats)

            # Final save
            self.output_proc._write_manifest(self.manifest_file)
            self.output_proc.write_progress(stats)
            self._print_stats(stats)
            return stats

        except KeyboardInterrupt:
            console.print("\n[yellow]Processing interrupted by user. Saving progress...")
            self.output_proc._write_manifest(self.manifest_file)
            self.output_proc.write_progress(stats)
            sys.exit(1)
        except Exception as e:
            console.print(f"\n[red]Error occurred: {e}")
            self.output_proc.write_progress(stats)
            raise

    def _process_batch(self, batch: List[dict], stats: dict, progress, task):
        """Process a batch of files"""
        for doc in batch:
            try:
                path = Path(doc["path"])
                
                # Handle directory structure
                if 'documents' not in str(self.base_folder):
                    full_path = self.base_folder / 'documents' / path
                else:
                    full_path = self.base_folder / path
                
                result = self.processor_fn(str(full_path), self.output_folder)
                
                # Handle multi-file results (chunks)
                if isinstance(result, dict):
                    if result.get("chunks"):
                        chunk_results = result["chunks"]
                        for chunk in chunk_results:
                            if not chunk.get("source"):
                                chunk["source"] = str(path)
                            self.output_proc.save_entry(chunk)
                        
                        if result.get("error"):
                            stats["failed"] += 1
                        else:
                            stats["processed"] += len(chunk_results)
                    else:
                        # Handle single file result
                        if not result.get("source"):
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