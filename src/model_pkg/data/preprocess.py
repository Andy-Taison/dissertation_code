"""
Functions to load and split data
"""

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from pathlib import Path
from ..config import RANDOM_STATE, EXPANDED_GRID_SIZE, COORDINATE_DIMENSIONS


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


def save_datasets(processed_directory: str | Path, data: list[pd.DataFrame], filenames: list[str]):
    """
    Saves datasets in 'processed_directory' using corresponding indexed filenames and appending '.csv'.

    :param processed_directory: Directory to store dataset CSV files
    :param data: List of DataFrames to save
    :param filenames: List of corresponding filenames to save data as, provide without extension
    """
    processed_data_dir = Path(processed_directory)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    if len(data) != len(filenames):
        raise ValueError("Data length does not match filenames length.")

    # Save datasets
    print(f"Saving datasets...")
    for d, f in zip(data, filenames):
        path = processed_data_dir / f"{f}.csv"
        d.to_csv(path, index=False, header=None)  # type: ignore
        print(f"Dataset saved as: '{path.name}'")


def load_processed_datasets(processed_directory: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train, val and test datasets from CSV files named as such in the 'processed_directory'.

    :param processed_directory: Directory to obtain 'train.csv', 'val.csv', and 'test.csv' files
    :return: train_data, val_data, test_data
    """
    # Load processed data
    processed_data_dir = Path(processed_directory)
    print(f"Loading train, validation and test set from: '{processed_directory.name}'...")
    train_data = pd.read_csv(processed_data_dir / "train.csv", header=None)
    val_data = pd.read_csv(processed_data_dir / "val.csv", header=None)
    test_data = pd.read_csv(processed_data_dir / "test.csv", header=None)
    print("Datasets loaded.")

    return train_data, val_data, test_data


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
    max_non_zero_in_row = non_zero_counts_per_row.max()
    min_non_zero_in_row = non_zero_counts_per_row.min()

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
    print(f"Maximum Non-zero Values in a Row: {max_non_zero_in_row}")
    print(f"Minimum Non-zero values in a Row: {min_non_zero_in_row}")
    print(f"Average Non-Zero Classes Per Row: {avg_class_per_row:.2f}")
    print(f"Rows with Only Zero Values: {rows_with_only_zero} ({rows_with_only_zero / num_rows * 100:.2f}%)")
    print(f"Rows with Exactly 1 Non-Zero Value: {rows_with_one_non_zero} ({rows_with_one_non_zero / num_rows * 100:.2f}%)")
    print(f"Rows with Multiple Non-Zero Values: {rows_with_multiple_non_zero} ({rows_with_multiple_non_zero / num_rows * 100:.2f}%)")
    print(f"Unique Rows: {unique_rows} ({unique_rows / num_rows * 100:.2f})%\n")


def split_diverse_sets(df: pd.DataFrame, compact_threshold: float = 0.8, dispersed_threshold: float = 2.5) -> list:
    """
    Subsets dataframe into 3x component based dataframes, and 3x spatial based dataframes.
    Samples in single-type dominant dataset can contain more than 1 component if one is heavily dominant.

    Spatial score is calculated based on the scaled bounding box volume + the scaled (mean nearest neighbour distance / number voxels).
    When spatial score is between thresholds, it is placed in the moderate spatial dataframe.

    :param df: Dataframe to subset
    :param compact_threshold: Spatial score < threshold, sample placed in compact dataframe
    :param dispersed_threshold: Spatial score > threshold, sample place in dispersed dataframe
    :return: List of dataframes [single-type dominant, moderate component diverse, high component diverse, compact, moderate spatial diverse, dispersed]
    """
    grids = torch.tensor(df.iloc[:, -(EXPANDED_GRID_SIZE ** COORDINATE_DIMENSIONS):].values, dtype=torch.float32)
    # Verify grid data has 1331 columns (11x11x11)
    assert grids.shape[1] == EXPANDED_GRID_SIZE * EXPANDED_GRID_SIZE * EXPANDED_GRID_SIZE, "Grid data does not have the correct number of elements (1331)."

    # Reshape grids into [batch_size, 11, 11, 11]
    grid_data = grids.view(-1, EXPANDED_GRID_SIZE, EXPANDED_GRID_SIZE, EXPANDED_GRID_SIZE)

    # Component based sets
    single_type_dominant = []
    moderate_comp_diverse = []
    high_comp_diverse = []

    # Spatial based sets
    compact = []
    moderate_spatial_diverse = []
    dispersed = []

    for i, sample in enumerate(grid_data):
        idxs = torch.nonzero(sample)
        x_idxs = idxs[:, 0]
        y_idxs = idxs[:, 1]
        z_idxs = idxs[:, 2]
        descriptors = sample[x_idxs, y_idxs, z_idxs].int()

        # Component based categories -----------------------------------------------------------------
        counts = descriptors.bincount()
        probs = counts / counts.sum()

        if (probs > 0.7).any():
            # Single type dominance (can contain more than a single component if one heavily dominates)
            single_type_dominant.append(i)
        elif (probs >= 0.5).any():
            # 50-70% dominance (captures 2 component type samples)
            moderate_comp_diverse.append(i)
        elif (probs < 0.5).all():
            # None dominating 50% or more
            high_comp_diverse.append(i)
        else:
            # Informs about cases not considered
            print(f"Does not fall into any component based category. Descriptor values: {descriptors}")

        # Spatial based categories --------------------------------------------------------------------
        num_voxels = len(descriptors)

        x_len = x_idxs.max() - x_idxs.min() + 1
        y_len = y_idxs.max() - y_idxs.min() + 1
        z_len = z_idxs.max() - z_idxs.min() + 1
        box_vol = x_len * y_len * z_len

        # Compute mean pairwise distance
        if num_voxels >= 2:
            dist_matrix = torch.cdist(idxs.float(), idxs.float())
            dist_matrix.fill_diagonal_(float('inf'))  # Don't include diagonal in calculation
            nearest_neighbour_dist = torch.min(dist_matrix, dim=-1).values
            mean_nn_dist = torch.mean(nearest_neighbour_dist)
        else:
            mean_nn_dist = torch.tensor(0.0)  # Single voxel

        scaled_box = 0.005 * box_vol
        scaled_dist = 2 * (mean_nn_dist / num_voxels * 0.5)  # Reduces the impact of small voxel numbers
        spatial_score = scaled_box + scaled_dist

        if spatial_score < compact_threshold:
            # Clustered and compact
            compact.append(i)
        elif spatial_score <= dispersed_threshold:
            # Moderate distance and compactness
            moderate_spatial_diverse.append(i)
        else:
            # Dispersed and spread out
            dispersed.append(i)

    subset_dfs = []  # 3x component based, 3x spatial based
    for ds in [single_type_dominant, moderate_comp_diverse, high_comp_diverse, compact, moderate_spatial_diverse, dispersed]:
        df_subset = df.iloc[ds].reset_index(drop=True)
        subset_dfs.append(df_subset)

    return subset_dfs
