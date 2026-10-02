"""Application state managed by the TUI."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AppState:
    sources_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    dataset_path: Optional[Path] = None
    status_message: str = ""
    current_screen: str = "main"
    selected_menu: int = 0
