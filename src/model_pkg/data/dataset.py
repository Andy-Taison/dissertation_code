"""
Creates datasets and dataloaders from dataframe
"""

import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import Dataset


class VoxelDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, max_voxels=8, shuffle_voxels=True):
        """
        Dataset for robot voxel data and ID.

        :param dataframe: Dataframe representing 3D grid data (last 1331 columns), Robot ID should be in the first column
        """
        print("Initialising Dataset...")
        grids = torch.tensor(dataframe.iloc[:, -1331:].values, dtype=torch.float32)
        # Verify grid data has 1331 columns (11x11x11)
        assert grids.shape[1] == 11 * 11 * 11, "Grid data does not have the correct number of elements (1331)."

        # Reshape grids into [batch_size, 11, 11, 11]
        self.grid_data = grids.view(-1, 11, 11, 11)
        self.robot_ids = torch.tensor(dataframe.iloc[:, 0].values, dtype=torch.long)
        self.max_voxels = max_voxels
        self.shuffle_voxels = shuffle_voxels

        print("Dataset created.\n")

    def __len__(self) -> int:
        """
        :return: Length of dataset
        """
        return len(self.robot_ids)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves robot ID and voxel data as tensors by dataset index.

        :param idx: Dataset index
        :return: Robot ID, Voxel data
        """
        # Get ID and single grid data (11x11x11 matrix)
        robot_id = self.robot_ids[idx]
        grid = self.grid_data[idx]

        # Non-zero coordinates and values
        non_zero_indices = grid.nonzero(as_tuple=True)  # (x, y, z)
        non_zero_values = grid[non_zero_indices]

        num_voxels = non_zero_values.shape[0]

        # Stack indices and values (x, y, z, value) to form a single tensor
        sparse_data = torch.cat([torch.stack(non_zero_indices, dim=1), non_zero_values.unsqueeze(1)], dim=1)  # (N, 4)

        # Pad to fixed shape (max_voxels, 4)
        if num_voxels < self.max_voxels:
            padding = torch.zeros(self.max_voxels - num_voxels, 4, dtype=torch.int64)
            sparse_data = torch.cat([sparse_data, padding], dim=0)

        # Shuffle order to avoid learning position bias
        if self.shuffle_voxels:
            sparse_data = sparse_data[torch.randperm(sparse_data.shape[0])]

        return robot_id, sparse_data
