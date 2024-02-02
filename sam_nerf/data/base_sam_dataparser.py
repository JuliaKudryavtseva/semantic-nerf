"""Dataclass for clip"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class SAM_features:
    filenames_array: List[Path]
    """filenames to load sam features tensor"""