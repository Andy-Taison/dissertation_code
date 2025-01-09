"""
Functions to load and split data
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from ..config import RANDOM_STATE


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


def clean_data(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Function to clean the dataset.
    Removes duplicate rows.
    Removes rows with only zero values.

    :param dataset: Dataset to clean
    :return: Cleaned dataset
    """
    print("Cleaning dataset...")
    # Obtains 3D grid data (last 1331 columns, 3rd column to end if original csv or 5th if combined csv)
    grids = dataset.iloc[:, -1331:]

    # Remove duplicate rows
    dups_removed = grids.drop_duplicates()

    # Keep only rows that have at least 1 non-zero value
    cleaned_grids = dups_removed[(dups_removed != 0).any(axis=1)]

    # Reattach preceding columns
    preceding_cols = dataset.iloc[:, :-1331]
    cleaned_df = pd.concat([preceding_cols.loc[cleaned_grids.index], cleaned_grids], axis=1)

    print("Dataset cleaned.\n")

    return cleaned_df


def split_data(dataframe: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validate, and test sets.

    :param dataframe: Data
    :param val_size: Proportion of entire dataset for the validation set
    :param test_size: Proportion of entire dataset for the test set
    :return: train_data, val_data, test_data
    """
    print("Splitting datasets...")

    # Split data into train/test
    train_data, test_data = train_test_split(dataframe, test_size=test_size, random_state=RANDOM_STATE)

    # Split train data into train/validate
    train_data, val_data = train_test_split(
        train_data, test_size=val_size / (1 - test_size), random_state=RANDOM_STATE
    )

    print(f"Data split into train ({len(train_data)}), val ({len(val_data)}), and test ({len(test_data)}) sets.\n")

    return train_data, val_data, test_data


def save_split_datasets(processed_directory: str | Path, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame):
    """
    Saves datasets as 'train.csv', 'val.csv', and 'test.csv' in 'processed_directory'.

    :param processed_directory: Directory to store dataset CSV files
    :param train_data: Dataset to save in 'train.csv'
    :param val_data: Dataset to save in 'val.csv'
    :param test_data: Dataset to save in 'test.csv'
    """
    processed_data_dir = Path(processed_directory)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_data_dir / "train.csv"
    val_path = processed_data_dir / "val.csv"
    test_path = processed_data_dir / "test.csv"

    # Save split datasets
    print(f"Saving split datasets...")
    train_data.to_csv(train_path, index=False, header=None)  # type: ignore
    val_data.to_csv(val_path, index=False, header=None)  # type: ignore
    test_data.to_csv(test_path, index=False, header=None)  # type: ignore
    print(f"Datasets saved as:\n\t{train_path}\n\t{val_path}\n\t{test_path}\n")


def summarise_dataset(dataset: pd.DataFrame):
    """
    Prints a summary of the dataset passed

    :param dataset: Dataset to summarise
    """
    # Obtains 3D grid data (last 1331 columns, 3rd column to end if original csv or 5th if combined csv)
    grids = dataset.iloc[:, -1331:]

    # Counts for each descriptor value
    total_counts = grids.stack().value_counts()

    # Proportion of each class
    proportions = (total_counts / grids.size * 100).round(2)

    # Sparsity
    zeros_per_row = grids.eq(0).sum(axis=1)
    avg_zeros_per_row = zeros_per_row.mean()
    avg_class_per_row = grids[grids > 0].count(axis=1).mean()

    # Non-zero
    non_zero_counts_per_row = grids.gt(0).sum(axis=1)
    rows_with_only_zero = (non_zero_counts_per_row == 0).sum()
    rows_with_one_non_zero = (non_zero_counts_per_row == 1).sum()
    rows_with_multiple_non_zero = (non_zero_counts_per_row > 1).sum()

    # Unique rows
    unique_rows = len(grids.drop_duplicates())

    # Dataset dimensions
    num_rows, num_columns = grids.shape

    # Print summary
    print("Dataset Summary:")
    print("-" * 50)
    print(f"Dimensions: {num_rows} rows x {num_columns} columns")
    print(f"\nOverall Class Counts:\n{total_counts.to_string()}")
    print(f"\nProportion of Each Class (%):\n{proportions.to_string()}")
    print(f"\nAverage Zeros Per Row: {avg_zeros_per_row:.2f}")
    print(f"Average Non-Zero Classes Per Row: {avg_class_per_row:.2f}")
    print(f"Rows with Only Zero Values: {rows_with_only_zero} ({rows_with_only_zero / num_rows * 100:.2f}%)")
    print(f"Rows with Exactly 1 Non-Zero Value: {rows_with_one_non_zero} ({rows_with_one_non_zero / num_rows * 100:.2f}%)")
    print(f"Rows with Multiple Non-Zero Values: {rows_with_multiple_non_zero} ({rows_with_multiple_non_zero / num_rows * 100:.2f}%)")
    print(f"Unique Rows: {unique_rows} ({unique_rows / num_rows * 100:.2f})%\n")
