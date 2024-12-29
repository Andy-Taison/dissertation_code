"""
Launcher script

Run using command:
python src/main.py
"""

from model_pkg import *
from pathlib import Path
import pandas as pd


def run():
    print("Starting VAE pipeline...\n")

    # Preprocess data
    # combined_data = combine_csv_files(config.DATA_DIR)
    # train_data, val_data, test_data = split_and_save_data(combined_data)

    # Load processed datasets
    processed_data_dir = Path(config.PROCESSED_DIR)
    train_data = pd.read_csv(processed_data_dir / "train.csv", header=None)
    val_data = pd.read_csv(processed_data_dir / "val.csv", header=None)
    test_data = pd.read_csv(processed_data_dir / "test.csv", header=None)

    # Create dataloaders
    train_ds, train_loader = create_dataset_and_loader(train_data, True)
    val_ds, val_loader = create_dataset_and_loader(val_data)
    test_ds, test_loader = create_dataset_and_loader(test_data)
    print(f"Preprocessed datasets loaded: train ({len(train_ds)}), val ({len(val_ds)}), and test ({len(test_ds)}) sets.\n")

    robot_ids, grid_data = next(iter(train_loader))
    print(f"robot_ids batch shape: {robot_ids.shape}, sample ID: {robot_ids[0]}")
    print(f"grid_data batch shape: {grid_data.shape}, grid data sample shape: {grid_data[0].shape}\nsample grid_data: \n{grid_data[0]}\n")

    # visualise_robot(grid_data[0], "Test title")

    # Train VAE
    # train()

    print("Pipeline complete.\n")


if __name__ == "__main__":
    run()
