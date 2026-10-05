"""Dataset loaders for supplementary Kaggle & Mendeley datasets.

Owner: P1 (Data & Config Lead)
Consumers: P2 (supplementary training data)
Reference: MASTER_PROMPT §6.1 Datasets 2-4
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DATA_DIR = Path("data/raw")


# ---------------------------------------------------------------------------
# Kaggle: Synthetic Patient Wait Time Records
# ---------------------------------------------------------------------------


def load_kaggle_wait_time(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the Kaggle synthetic patient wait-time dataset.

    Dataset: https://www.kaggle.com/datasets/thedevastator/synthetic-patient-wait-time-records-for-hospital

    Args:
        data_dir: Directory containing the Kaggle CSV. Defaults to data/raw/kaggle_patient_wait/.

    Returns:
        DataFrame with standardized column names.

    Raises:
        FileNotFoundError: If no CSV files are found in the directory.
    """
    data_dir = data_dir or RAW_DATA_DIR / "kaggle_patient_wait"
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            "Download from https://www.kaggle.com/datasets/thedevastator/"
            "synthetic-patient-wait-time-records-for-hospital "
            f"and place CSV files in {data_dir}"
        )

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    logger.info(f"Loaded Kaggle wait-time dataset: {len(df)} records from {len(csv_files)} file(s)")

    # Standardize column names to snake_case
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


# ---------------------------------------------------------------------------
# Mendeley: Radiology Workflow Event Logs
# ---------------------------------------------------------------------------


def load_mendeley_radiology(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load the Mendeley radiology workflow event log dataset.

    Dataset: https://data.mendeley.com/datasets/m975p5d63z/1

    Args:
        data_dir: Directory containing the Mendeley CSV files.
                  Defaults to data/raw/mendeley_radiology/.

    Returns:
        Dictionary mapping filename stems to DataFrames.

    Raises:
        FileNotFoundError: If no CSV files are found.
    """
    data_dir = data_dir or RAW_DATA_DIR / "mendeley_radiology"
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            "Download from https://data.mendeley.com/datasets/m975p5d63z/1 "
            f"and place CSV files in {data_dir}"
        )

    datasets: dict[str, pd.DataFrame] = {}
    for f in csv_files:
        df = pd.read_csv(f)
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        datasets[f.stem] = df
        logger.info(f"Loaded Mendeley '{f.stem}': {len(df)} records")

    return datasets


# ---------------------------------------------------------------------------
# Kaggle: Healthcare No-Show and Wait Time Analysis
# ---------------------------------------------------------------------------


def load_kaggle_noshow(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the Kaggle healthcare no-show dataset.

    Dataset: https://www.kaggle.com/datasets/thedevastator/healthcare-no-show-and-wait-time-analysis

    Args:
        data_dir: Directory containing the CSV. Defaults to data/raw/kaggle_noshow/.

    Returns:
        DataFrame with standardized column names.

    Raises:
        FileNotFoundError: If no CSV files are found.
    """
    data_dir = data_dir or RAW_DATA_DIR / "kaggle_noshow"
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            "Download from https://www.kaggle.com/datasets/thedevastator/"
            "healthcare-no-show-and-wait-time-analysis "
            f"and place CSV files in {data_dir}"
        )

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    logger.info(f"Loaded Kaggle no-show dataset: {len(df)} records")

    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


# ---------------------------------------------------------------------------
# Generated Data Loader
# ---------------------------------------------------------------------------


def load_generated_dataset(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the SimPy-generated synthetic dataset.

    Args:
        data_dir: Directory containing patients.csv. Defaults to data/generated/.

    Returns:
        DataFrame with patient records.

    Raises:
        FileNotFoundError: If patients.csv is not found.
    """
    data_dir = data_dir or Path("data/generated")
    patients_path = data_dir / "patients.csv"

    if not patients_path.exists():
        raise FileNotFoundError(
            f"Generated dataset not found at {patients_path}. "
            "Run the data generator first: python -m src.data.generator"
        )

    df = pd.read_csv(patients_path, parse_dates=[
        "registration_time", "queue_entry_time", "prep_start_time",
        "imaging_start_time", "imaging_end_time",
        "reporting_start_time", "reporting_end_time", "departure_time",
    ])
    logger.info(f"Loaded generated dataset: {len(df)} records")
    return df


# ---------------------------------------------------------------------------
# Processed Data Loader
# ---------------------------------------------------------------------------


def load_processed_splits(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-processed train/val/test parquet splits.

    Args:
        data_dir: Directory containing parquet files. Defaults to data/processed/.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        FileNotFoundError: If any parquet file is missing.
    """
    data_dir = data_dir or Path("data/processed")

    splits: list[pd.DataFrame] = []
    for split_name in ("train", "val", "test"):
        path = data_dir / f"{split_name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Processed {split_name} split not found at {path}. "
                "Run the preprocessor first: python -m src.data.preprocessor"
            )
        splits.append(pd.read_parquet(path))
        logger.info(f"Loaded {split_name} split: {len(splits[-1])} records")

    return splits[0], splits[1], splits[2]
