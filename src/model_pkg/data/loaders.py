"""
Loads data and creates dataloaders
"""

import pandas as pd
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader

from ..config import *


def create_dataset_and_loader(dataframe: pd.DataFrame, shuffle: bool = False, batch_size: int = BATCH_SIZE) -> tuple[TensorDataset, DataLoader]:
    """
    Extracts the robot_id and 3D grid data, wraps in TensorDatasets and creates DataLoaders.
    Reshapes grid_data to form 11x11x11 matrix.

    Dataloader objects will return (robot_ids, grid_data) when accessing.

    :param shuffle: Boolean to shuffle dataloader, default False
    :param dataframe: DataFrame to create dataloader from
    :param batch_size: Batch size to use, defaults to config setting
    :return: TensorDataset, DataLoader objects
    """
    print("Creating DataLoader object...")

    # Separate robot_id (first column) and 3D grid data (last 1331 columns, 3rd column to end if original csv or 5th if combined csv)
    robot_ids = torch.tensor(dataframe.iloc[:, 0].values, dtype=torch.long)
    grids = torch.tensor(dataframe.iloc[:, -1331:].values, dtype=torch.float32)

    # Verify grid data has 1331 columns (11x11x11)
    assert grids.shape[1] == 11 * 11 * 11, "Grid data does not have the correct number of elements (1331)."

    # Reshape grids into [batch_size, 11, 11, 11]
    grids = grids.view(-1, 11, 11, 11)

    # Create TensorDataset
    dataset = TensorDataset(robot_ids, grids)

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    print("DataLoader created.\n")

    return dataset, loader
