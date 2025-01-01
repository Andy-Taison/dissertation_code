"""
Functions to load and split data
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from ..config import PROCESSED_DIR, RANDOM_STATE


def combine_csv_files(directory: str) -> pd.DataFrame:
    """
    Loads all csv files present in directory as combined pandas dataframe
    Adds filename as new column (2nd column).
    Creates new ID column (1st column) to avoid duplicate ID's.
    Original columns are all maintained.

    :param directory: Directory containing csv files
    :return: Combined dataframe
    """
    # Create path object
    dir_path = Path(directory)

    if not dir_path.is_dir():
        raise ValueError(f"{directory} is not a valid directory.")

    # Get all csv files in directory
    csv_files = list(dir_path.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in directory {directory}")

    # Load all CSV files into a single DataFrame
    print("Combining CSV files (no headers) into a single DataFrame...")
    combined_data = []
    for file in csv_files:
        print(f"Loading file: {file}")
        try:
            # Load each CSV with no header
            df = pd.read_csv(file, header=None)

            # Insert filename as new column without extension
            df.insert(0, "file", file.stem)

            combined_data.append(df)
        except Exception as e:
            print(f"\nError loading file {file}: {e}")

    if not combined_data:
        raise ValueError("\nNo CSV files could be loaded.")

    # Concatenate all loaded DataFrames
    combined_df = pd.concat(combined_data, ignore_index=True)

    # Index added as column for new unique ID's, avoiding duplicate robot ID's
    combined_df = combined_df.reset_index()

    # Summary of combined DataFrame
    print(f"\nCombined {len(combined_data)} files into a single DataFrame.")
    print(f"Shape of combined DataFrame: {combined_df.shape}\n")

    return combined_df


def split_and_save_data(dataframe: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validate, and test sets, saves as CSV files.

    :param dataframe: Data
    :param val_size: Proportion of entire dataset for the validation set
    :param test_size: Proportion of entire dataset for the test set
    :return: train_data, val_data, test_data
    """
    processed_data_dir = Path(PROCESSED_DIR)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_data_dir / "train.csv"
    val_path = processed_data_dir / "val.csv"
    test_path = processed_data_dir / "test.csv"

    print("Splitting datasets...")

    # Split data into train/test
    train_data, test_data = train_test_split(dataframe, test_size=test_size, random_state=RANDOM_STATE)

    # Split train data into train/validate
    train_data, val_data = train_test_split(
        train_data, test_size=val_size / (1 - test_size), random_state=RANDOM_STATE
    )

    # Save split datasets
    train_data.to_csv(train_path, index=False, header=None)
    val_data.to_csv(val_path, index=False, header=None)
    test_data.to_csv(test_path, index=False, header=None)

    print(f"Data split into train ({len(train_data)}), val ({len(val_data)}), and test ({len(test_data)}) sets.")
    print(f"Datasets saved as:\n\t{train_path}\n\t{val_path}\n\t{test_path}\n")

    return train_data, val_data, test_data
