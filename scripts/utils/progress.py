from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn
from rich.console import Console
from pathlib import Path
from datetime import datetime
import srsly
import os

console = Console()

class ProcessingProgress:
    """Handle progress file tracking and stats"""
    def __init__(self, progress_file: Path):
        self.progress_file = progress_file
        self.stats = self.load_progress()

    def load_progress(self) -> dict:
        """Load last progress stats"""
        if not self.progress_file.exists():
            return {"processed": 0, "skipped": 0, "failed": 0, "total": 0}
            
        try:
            with open(self.progress_file, 'rb') as f:
                f.seek(-min(os.path.getsize(self.progress_file), 4096), os.SEEK_END)
                last_line = f.read().decode().strip().split('\n')[-1]
                entry = srsly.json_loads(last_line)
                return entry.get("stats", {})
        except Exception:
            return {"processed": 0, "skipped": 0, "failed": 0, "total": 0}

    def save_progress(self, stats: dict, current_idx: int):
        """Save current progress"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "processed_count": current_idx
        }
        with open(self.progress_file, "a") as f:
            f.write(srsly.json_dumps(entry) + "\n")

    @property
    def processed_count(self) -> int:
        """Get number of processed files"""
        return self.stats.get("processed", 0)

class ProgressTracker:
    def __init__(self, total: int, task_name: str, progress_fields: dict):
        # Remove total from progress_fields since it's passed separately
        fields = progress_fields.copy()
        if 'total' in fields:
            del fields['total']
            
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn(self._build_stats_display(fields))
        )
        self.task = self.progress.add_task(  # Store task ID as self.task
            f"[green]{task_name}",
            total=total,
            **fields
        )
        self.stats = fields

    def _build_stats_display(self, fields: dict) -> str:
        """Build progress bar stats display from fields"""
        return " | ".join(f"{k.title()}: {{task.fields[{k}]}}" for k in fields.keys())

    def update(self, advance: int = 1, **fields):
        """Update progress and stats"""
        self.stats.update(fields)
        self.progress.update(self.task, advance=advance, **fields)

    def __enter__(self):
        return self.progress.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.progress.__exit__(exc_type, exc_val, exc_tb)
