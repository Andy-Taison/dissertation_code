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
    train_data = pd.read_csv(processed_data_dir / "train.csv")
    val_data = pd.read_csv(processed_data_dir / "val.csv")
    test_data = pd.read_csv(processed_data_dir / "test.csv")
    print(f"Preprocessed datasets loaded: train ({len(train_data)}), val ({len(val_data)}), and test ({len(test_data)}) sets\n.")

    # Create dataloaders
    train_loader = create_dataloader(train_data, True)
    val_loader = create_dataloader(val_data)
    test_loader = create_dataloader(test_data)

    # Train VAE
    # train()

    print("Pipeline complete.\n")


if __name__ == "__main__":
    run()
