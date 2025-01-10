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
    # print("Combined full dataset")
    # summarise_dataset(combined_data)
    # cleaned_df = clean_data(combined_data)
    # train_data, val_data, test_data = split_data(cleaned_df)
    # save_split_datasets(config.PROCESSED_DIR, train_data, val_data, test_data)
    # print("Training dataset")
    # summarise_dataset(train_data)
    # print("Validation dataset")
    # summarise_dataset(val_data)
    # print("Test dataset")
    # summarise_dataset(test_data)

    # Load processed data
    processed_data_dir = Path(config.PROCESSED_DIR)
    train_data = pd.read_csv(processed_data_dir / "train.csv", header=None)
    val_data = pd.read_csv(processed_data_dir / "val.csv", header=None)
    test_data = pd.read_csv(processed_data_dir / "test.csv", header=None)

    # Create datasets and dataloaders
    print("Training dataset")
    summarise_dataset(train_data)
    train_ds, train_loader = create_dataset_and_loader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    print("Validation dataset")
    summarise_dataset(val_data)
    val_ds, val_loader = create_dataset_and_loader(val_data, batch_size=config.BATCH_SIZE)
    print("Test dataset")
    summarise_dataset(test_data)
    test_ds, test_loader = create_dataset_and_loader(test_data, batch_size=config.BATCH_SIZE)
    print(f"Preprocessed datasets loaded: train ({len(train_ds)}), val ({len(val_ds)}), and test ({len(test_ds)}) sets.\n")

    # Sample
    robot_ids, grid_data = next(iter(train_loader))
    print(f"robot_ids batch shape: {robot_ids.shape}, sample ID: {robot_ids[0]}")
    print(f"grid_data batch shape: {grid_data.shape}, grid data sample shape: {grid_data[0].shape}\n")

    # visualise_robot(grid_data[0], "Test title")

    """
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
    """
    """
    # For testing
    from torch.utils.data import DataLoader, Subset
    subset_indices = list(range(128))  # Indices for the first 128 samples
    subset_train_ds = Subset(train_ds, subset_indices)
    subset_val_ds = Subset(val_ds, subset_indices)
    # subset_train_loader = DataLoader(subset_train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    # subset_val_loader = DataLoader(subset_val_ds, batch_size=config.BATCH_SIZE)

    # Train VAE
    # history = train_val(vae, subset_train_loader, subset_val_loader, criterion, optimizer, config.EPOCHS)  # History will be to the latest model, which most likely will not be the best model
    # print(history)

    # history = TrainingHistory.load_history("test_history.pth")
    # print(history)
    # history.rollback("last_improved_model")  # Rollback does not save history
    # history.save_history()  # Saving rolled back history will overwrite old history (models unaffected)

    # model, optimizer, scheduler, epoch = load_model_checkpoint(Path(config.MODEL_CHECKPOINT_DIR / "test" / "best_f1_avg_epoch_7.pth"))

    train_grid_search(subset_train_ds, subset_val_ds, "test")
    best_history, best_score, best_epoch = search_grid_history()
    print(f"Best tradeoff score: {best_score}")
    print(f"Best tradeoff epoch: {best_epoch}")
    """

    # Grid search
    train_grid_search(train_ds, val_ds, "base")
    best_history, best_score, best_epoch = search_grid_history()

    # Rollback to best performing history and model to load checkpoint
    if best_epoch < best_history.epochs_run:
        best_history.rollback(best_epoch)  # Rollback does not save history
    best_history.save_history(f"best_performing_{best_history.model_name}")

    print(best_history)

    # Get best performing model, optimizer and scheduler
    model, optimizer, scheduler, epoch = load_model_checkpoint(best_history)
    print("Loaded model:")
    print(model)
    print("Loaded optimizer:")
    print(optimizer)
    print("Loaded scheduler:")
    print(scheduler)
    print("Loaded epoch: " + str(epoch))

    print("\nPipeline complete!")


if __name__ == "__main__":
    run()
