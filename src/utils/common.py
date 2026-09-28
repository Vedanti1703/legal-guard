"""
src/utils/common.py

Shared utility functions for:
- Path resolution across datasets and project directories
- Robust JSON and JSONL reading/writing
- Legal text normalization (cleaning whitespace, unicode artifacts)
- Logging setup
"""

import os
import sys
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_project_root() -> Path:
    """Returns the absolute Path of the project root directory."""
    # Assuming this file is at src/utils/common.py -> parents[2] is root
    return Path(__file__).resolve().parents[2]


def get_cuad_path() -> Path:
    """Finds and returns the CUAD dataset directory."""
    candidates = [
        Path(r"c:\Users\91993\Downloads\cuad-main"),
        get_project_root() / "data" / "raw" / "cuad",
        get_project_root().parent / "cuad-main",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("CUAD directory not found in candidate paths.")


def get_acord_path() -> Path:
    """Finds and returns the ACORD dataset directory."""
    candidates = [
        Path(r"c:\Users\91993\Downloads\acord-main"),
        get_project_root() / "data" / "raw" / "acord",
        get_project_root().parent / "acord-main",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("ACORD directory not found in candidate paths.")


def get_indian_path() -> Path:
    """Finds and returns the Indian dataset directory."""
    candidates = [
        Path(r"c:\Users\91993\Downloads\indian dataset"),
        get_project_root() / "data" / "raw" / "indian",
        get_project_root().parent / "indian dataset",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Indian dataset directory not found in candidate paths.")


def setup_logger(name: str = "nlp_pipeline", log_file: str = None) -> logging.Logger:
    """Configures and returns a standard logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger


def clean_legal_text(text: str) -> str:
    """
    Cleans raw legal text while strictly preserving legal clause markers
    and section numbering (e.g. 1.1, 2(a), (i), Section 27).
    """
    if not text or not isinstance(text, str):
        return ""

    # Normalize unicode quotation marks and hyphens
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '--')
    text = text.replace('\u00a0', ' ')  # non-breaking space

    # Remove non-printable control characters except newline and tab
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Normalize multiple whitespace characters inside lines while preserving single breaks
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.splitlines()]
    # Remove multiple empty blank lines
    cleaned_lines = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                cleaned_lines.append("")
                prev_blank = True
        else:
            cleaned_lines.append(line)
            prev_blank = False

    return "\n".join(cleaned_lines).strip()


def load_json(filepath: Path | str) -> Any:
    """Reads a JSON file with utf-8 encoding."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, filepath: Path | str, indent: int = 2) -> None:
    """Writes data to a JSON file with utf-8 encoding."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_jsonl(filepath: Path | str) -> Generator[Dict[str, Any], None, None]:
    """Yields parsed JSON objects from a JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logging.warning(f"Error parsing line {line_num} in {filepath}: {e}")


def save_jsonl(records: List[Dict[str, Any]], filepath: Path | str) -> None:
    """Writes a list of dictionaries as JSONL."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
