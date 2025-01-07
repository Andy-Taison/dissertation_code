"""
Director script

Run using command:
python src/main.py
"""

from model_pkg import *
from pathlib import Path
import pandas as pd
from torchinfo import summary
import torch.optim as optim


def run():
    print("Starting VAE pipeline...\n")

    # Preprocess data and save
    # combined_data = combine_csv_files(config.DATA_DIR)
    # train_data, val_data, test_data = split_and_save_data(combined_data, config.PROCESSED_DIR)

    # Load processed data
    processed_data_dir = Path(config.PROCESSED_DIR)
    train_data = pd.read_csv(processed_data_dir / "train.csv", header=None)
    val_data = pd.read_csv(processed_data_dir / "val.csv", header=None)
    test_data = pd.read_csv(processed_data_dir / "test.csv", header=None)

    # Create datasets and dataloaders
    train_ds, train_loader = create_dataset_and_loader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_ds, val_loader = create_dataset_and_loader(val_data, batch_size=config.BATCH_SIZE)
    test_ds, test_loader = create_dataset_and_loader(test_data, batch_size=config.BATCH_SIZE)
    print(f"Preprocessed datasets loaded: train ({len(train_ds)}), val ({len(val_ds)}), and test ({len(test_ds)}) sets.\n")

    # Sample
    robot_ids, grid_data = next(iter(train_loader))
    print(f"robot_ids batch shape: {robot_ids.shape}, sample ID: {robot_ids[0]}")
    print(f"grid_data batch shape: {grid_data.shape}, grid data sample shape: {grid_data[0].shape}\n")

    # visualise_robot(grid_data[0], "Test title")

    # Define model
    vae = VAE(config.INPUT_DIM, config.LATENT_DIM, "test").to(config.DEVICE)

    # Inspect
    # print("Model summary:")
    # summary(vae, config.INPUT_DIM)
    # print("\nModel parameters:")
    # for name, param in vae.named_parameters():
    #     print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
    # print()

    # Initialise training components
    criterion = VaeLoss("mse")
    optimizer = optim.Adam(vae.parameters(), lr=config.LEARNING_RATE)

    # for testing
    from torch.utils.data import DataLoader, Subset
    subset_indices = list(range(128))  # Indices for the first 128 samples
    subset_train_ds = Subset(train_ds, subset_indices)
    subset_val_ds = Subset(val_ds, subset_indices)
    subset_train_loader = DataLoader(subset_train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    subset_val_loader = DataLoader(subset_val_ds, batch_size=config.BATCH_SIZE)

    # Train VAE
    history = train_val(vae, subset_train_loader, subset_val_loader, criterion, optimizer, config.EPOCHS)  # History will be to the latest model, which most likely will not be the best model
    print(history)

    # history = TrainingHistory.load_history("test_history.pth")
    # print(history)
    # history.rollback("last_improved_model")  # Rollback does not save history
    # history.save_history()  # Saving rolled back history will overwrite old history (models unaffected)

    # model, optimizer, scheduler, epoch = load_model_checkpoint(Path(config.MODEL_CHECKPOINT_DIR / "test" / "best_f1_avg_epoch_7.pth"))
    # print(model)
    # print(optimizer)
    # print(scheduler)
    # print(epoch)

    # history = train_val(vae, train_loader, val_loader, criterion, optimizer, config.EPOCHS)

    print("Pipeline complete.")


if __name__ == "__main__":
    run()
