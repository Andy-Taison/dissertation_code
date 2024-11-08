import pandas as pd
from sklearn.model_selection import train_test_split
from ..config import *


def combine_csv_files(directory: str) -> pd.DataFrame:
    """
    Loads all csv files present in directory as combined pandas dataframe

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
    print("Combining csv files into single DataFrame.")
    combined_data = pd.concat(
        (pd.read_csv(file) for file in csv_files), ignore_index=True
    )
    print("Files combined.\n")
    return combined_data


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
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"Data split into train ({len(train_data)}), val ({len(val_data)}), and test ({len(test_data)}) sets.")
    print(f"Datasets saved as:\n\t{train_path}\n\t{val_path}\n\t{test_path}\n")

    return train_data, val_data, test_data
