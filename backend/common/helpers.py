"""
Common helper utilities used across the Drishti project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any
import hashlib
import os

import pandas as pd

from backend.common.constants import DIRECTORIES, SUPPORTED_FILE_TYPES, DISTRICT_MAPPING
from backend.common.exceptions import (
    DatasetNotFoundError,
    EmptyDatasetError,
    UnsupportedFileTypeError,
)


def ensure_directories() -> None:
    """Create all required project directories."""
    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


def file_exists(path: str | Path) -> bool:
    """Return True if file exists."""
    return Path(path).is_file()


def directory_exists(path: str | Path) -> bool:
    """Return True if directory exists."""
    return Path(path).is_dir()


def validate_file_type(path: str | Path) -> None:
    """Validate supported file extension."""
    suffix = Path(path).suffix.lower()

    if suffix not in SUPPORTED_FILE_TYPES:
        raise UnsupportedFileTypeError(
            f"Unsupported file type: {suffix}"
        )


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a CSV safely.
    """
    path = Path(path)

    if not path.exists():
        raise DatasetNotFoundError(str(path))

    validate_file_type(path)

    df = pd.read_csv(path)

    if df.empty:
        raise EmptyDatasetError(
            f"{path.name} contains no records."
        )

    return df


def save_csv(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save dataframe as CSV.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize dataframe column names.
    """

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(r"[()]", "", regex=True)
    )

    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows.
    """
    return df.drop_duplicates(ignore_index=True)


def standardize_district_name(district: Any) -> str:
    """
    Standardize raw district name using project DISTRICT_MAPPING lookup.

    Parameters
    ----------
    district : Any
        Raw district string.

    Returns
    -------
    str
        Normalized standard district name.
    """
    if not isinstance(district, str) or not district.strip():
        return "Unknown"
    
    cleaned = district.strip().upper()
    return DISTRICT_MAPPING.get(cleaned, cleaned.title())


def dataframe_summary(df: pd.DataFrame) -> dict[str, int]:
    """
    Return useful dataframe statistics.
    """

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


def get_file_hash(path: str | Path) -> str:
    """
    Generate SHA256 hash.
    """

    sha = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(8192)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


def list_csv_files(directory: str | Path) -> list[Path]:
    """
    Return all CSV files inside a directory.
    """

    directory = Path(directory)

    if not directory.exists():
        return []

    return sorted(directory.glob("*.csv"))


def is_dataframe_valid(df: pd.DataFrame) -> bool:
    """
    Basic dataframe validation.
    """

    return (
        df is not None
        and not df.empty
        and len(df.columns) > 0
    )


def print_header(title: str) -> None:
    """
    Pretty console header.
    """

    line = "=" * 70

    print(f"\n{line}")
    print(title)
    print(line)


def bytes_to_mb(size: int) -> float:
    """
    Convert bytes to MB.
    """
    return round(size / (1024 * 1024), 2)


def folder_size(directory: str | Path) -> float:
    """
    Calculate folder size in MB.
    """

    total = 0

    for root, _, files in os.walk(directory):
        for file in files:
            total += os.path.getsize(
                os.path.join(root, file)
            )

    return bytes_to_mb(total)


def flatten(iterable: Iterable[Iterable]) -> list:
    """
    Flatten nested iterables.
    """

    return [item for group in iterable for item in group]