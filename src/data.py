from pathlib import Path

import numpy as np
import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw dataset directory
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_dataset(data_dir=DATA_DIR):
    """
    Load all Parquet files in the raw dataset directory.
    """

    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {data_dir}. "
            "Place your dataset files in data/raw/."
        )

    frames = []

    for file in files:
        print(f"Loading: {file.name}")

        frame = pd.read_parquet(file)
        frames.append(frame)

    data = pd.concat(
        frames,
        ignore_index=True
    )

    # Standardise column names
    data.columns = [
        str(column).strip()
        for column in data.columns
    ]

    return data


def clean_dataset(data):
    """
    Perform basic cleaning without changing the original dataset.
    """

    data = data.copy()

    # Remove duplicate column names, keeping the first occurrence
    data = data.loc[:, ~data.columns.duplicated()]

    # Remove rows with missing target labels
    label_column = find_label_column(data)

    if label_column is None:
        raise ValueError(
            "Could not identify the label column. "
            f"Available columns: {list(data.columns)}"
        )

    data = data.dropna(subset=[label_column]).copy()

    # Replace positive and negative infinity with missing values
    data = data.replace([np.inf, -np.inf], np.nan)

    # Remove exact duplicate rows
    data = data.drop_duplicates()

    return data


def find_label_column(data):
    """
    Find a likely target column.
    """

    candidates = [
        "Label",
        "label",
        " Label",
        " Label "
    ]

    for column in candidates:
        if column in data.columns:
            return column

    return None


def inspect_dataset(data):
    """
    Print useful information about the dataset.
    """

    label_column = find_label_column(data)

    print("\nDataset shape:")
    print(data.shape)

    print("\nColumn names:")
    print(data.columns.tolist())

    print("\nMissing values:")
    print(data.isna().sum().sort_values(ascending=False).head(20))

    if label_column:
        print("\nLabel distribution:")
        print(data[label_column].value_counts(dropna=False))

    print("\nFirst five rows:")
    print(data.head())
