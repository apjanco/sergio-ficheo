import srsly
from pathlib import Path
from datetime import datetime
import os
from rich.console import Console
import tempfile
import shutil

console = Console()

class ManifestProcessor:
    def __init__(self, manifest_path: Path, progress_file: Path = None):
        self.manifest_path = Path(manifest_path)
        self.progress_file = progress_file
        self.total_files = self.count_lines()
        self.processed = 0 if not progress_file else self.get_last_progress()
        self.entries = {}
        self._load_existing_entries()

    def count_lines(self) -> int:
        """Fast line count without loading content"""
        if not self.manifest_path.exists():
            return 0
        with open(self.manifest_path, 'rb') as f:
            return sum(1 for _ in f)

    def get_last_progress(self) -> int:
        """Get last processed count from progress file"""
        if not self.progress_file or not os.path.exists(self.progress_file):
            return 0
        try:
            with open(self.progress_file, 'rb') as f:
                f.seek(-min(os.path.getsize(self.progress_file), 4096), os.SEEK_END)
                last_lines = f.read().decode().strip().split('\n')
                if last_lines:
                    last_entry = srsly.json_loads(last_lines[-1])
                    return last_entry.get("processed_count", 0)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read progress file: {e}")
        return 0

    def write_progress(self, stats: dict):
        """Write progress information"""
        if not self.progress_file:
            return
        progress_entry = {
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "files_processed": len(self.entries)
        }
        with open(self.progress_file, 'w') as f:
            f.write(srsly.json_dumps(progress_entry) + '\n')

    def stream_entries(self):
        """Stream JSONL entries"""
        if not self.manifest_path.exists():
            return
        for entry in srsly.read_jsonl(self.manifest_path):
            yield entry

    def save_entry(self, entry: dict, manifest_path: Path = None):
        """Update or append entry to manifest"""
        if "source" not in entry:
            return
            
        source = entry["source"]
        if source in self.entries:
            # Only update if entry has changed
            if self.entries[source] != entry:
                self.entries[source] = entry
        else:
            self.entries[source] = entry

        # Write entire manifest atomically if needed
        if len(self.entries) % 100 == 0:
            self._write_manifest(manifest_path or self.manifest_path)

    def _load_existing_entries(self):
        """Load existing entries into memory for deduplication"""
        self.entries = {}
        if self.manifest_path.exists():
            for entry in srsly.read_jsonl(self.manifest_path):
                if "source" in entry:
                    # Store using source path as key without project prefix
                    path = Path(entry["source"])
                    if "documents" in path.parts:
                        # Get path after 'documents'
                        key = str(Path(*path.parts[path.parts.index("documents")+1:]))
                    else:
                        key = str(path)
                    self.entries[key] = entry

    def _write_manifest(self, manifest_path: Path):
        """Write all entries atomically"""
        temp_path = manifest_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            for entry in self.entries.values():
                f.write(srsly.json_dumps(entry) + '\n')
        temp_path.replace(manifest_path)

    def print_status(self):
        """Print initial status"""
        console.print(f"\n[blue]Total files to process: {self.total_files}")
        if self.processed > 0:
            pct = (self.processed / self.total_files) * 100
            console.print(f"[yellow]Resuming from file {self.processed} ({pct:.1f}% complete)")
        console.print("")
