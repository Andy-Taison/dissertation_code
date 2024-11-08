"""
Loads data and creates dataloaders
"""

import pandas as pd
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader

from ..config import *


def create_dataloader(dataframe: pd.DataFrame, shuffle: bool = False, batch_size: int = BATCH_SIZE) -> DataLoader:
    """
    Extracts the robot_id and 3D grid data, wraps in TensorDatasets and creates DataLoaders.

    Dataloader objects will return (robot_ids, grid_data) when accessing.

    :param shuffle: Boolean to shuffle dataloader, default False
    :param dataframe: DataFrame to create dataloader from
    :param batch_size: Batch size to use, defaults to config setting
    :return: DataLoader object
    """
    print("Creating DataLoader object...")

    # Separate robot_id (first column) and 3D grid data (columns 3 to end of line)
    robot_ids = torch.tensor(dataframe.iloc[:, 0].values, dtype=torch.long)
    grids = torch.tensor(dataframe.iloc[:, 2:].values, dtype=torch.float32)

    # Create TensorDataset
    dataset = TensorDataset(robot_ids, grids)

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    print("DataLoader created.\n")

    return loader
