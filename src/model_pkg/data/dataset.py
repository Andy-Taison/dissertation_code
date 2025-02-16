"""
Creates datasets and dataloaders from dataframe
"""

import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import Dataset
import torch.nn.functional as F
from ..config import NUM_CLASSES, COORDINATE_DIMENSIONS, DEVICE, EXPANDED_GRID_SIZE


class VoxelDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, max_voxels=8, shuffle_voxels=True):
        """
        Dataset for robot voxel data and ID.

        :param dataframe: Dataframe representing 3D grid data (last 1331 columns), Robot ID should be in the first column
        :param max_voxels: Maximum number of voxels in entire dataset
        """
        print("Initialising Dataset...")
        grids = torch.tensor(dataframe.iloc[:, -(EXPANDED_GRID_SIZE**COORDINATE_DIMENSIONS):].values, dtype=torch.float32)
        # Verify grid data has 1331 columns (11x11x11)
        assert grids.shape[1] == EXPANDED_GRID_SIZE * EXPANDED_GRID_SIZE * EXPANDED_GRID_SIZE, "Grid data does not have the correct number of elements (1331)."

        # Reshape grids into [batch_size, 11, 11, 11]
        self.grid_data = grids.view(-1, EXPANDED_GRID_SIZE, EXPANDED_GRID_SIZE, EXPANDED_GRID_SIZE)
        self.robot_ids = torch.tensor(dataframe.iloc[:, 0].values, dtype=torch.long)
        self.max_voxels = max_voxels
        self.coordinate_dim = COORDINATE_DIMENSIONS
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
        Normalises the coordinate values in range [0,1].
        One-hot encodes the descriptor values, voxel data column indices 3:7 correspond to descriptor values 0:4
        Includes 0 descriptor value at for padding so voxel data is the same shape for every robot.

        :param idx: Dataset index
        :return: Robot ID, Voxel data
        """
        # Get ID and single grid data (11x11x11 matrix)
        robot_id = self.robot_ids[idx]
        grid = self.grid_data[idx]

        # Non-zero coordinates and values
        non_zero_indices = grid.nonzero(as_tuple=True)  # (x, y, z)
        non_zero_values = grid[non_zero_indices]

        # Normalise coordinates
        normalised_coords = torch.stack(non_zero_indices, dim=1) / EXPANDED_GRID_SIZE  # range [0, 1] divides by grid size as padded voxels encoded as expanded grid size

        # One-hot encode descriptor values
        one_hot_values = F.one_hot(non_zero_values.long(), num_classes=NUM_CLASSES)

        num_voxels = non_zero_values.shape[0]

        # Stack indices and values (x, y, z, one-hot descriptors) to form a single tensor
        sparse_data = torch.cat([normalised_coords, one_hot_values], dim=1)  # (N, COORDINATE_DIM + NUM_CLASSES)

        # Pad to fixed shape (max_voxels, NUM_CLASSES + 3)
        if num_voxels < self.max_voxels:
            coor = torch.full((self.max_voxels - num_voxels, COORDINATE_DIMENSIONS), 1.0)  # Padding coordinates encoded as normalised expanded grid size value
            zero_descriptors = torch.zeros((self.max_voxels - num_voxels, NUM_CLASSES))  # Initialise descriptor values
            padding = torch.cat((coor, zero_descriptors), dim=1)
            padding[:, COORDINATE_DIMENSIONS] = 1  # One-hot encoding for 0 descriptor (empty space) padding
            sparse_data = torch.cat([sparse_data, padding], dim=0)

        # Shuffle order to avoid learning position bias
        if self.shuffle_voxels:
            sparse_data = sparse_data[torch.randperm(sparse_data.shape[0])]

        return robot_id, sparse_data


def sparse_to_dense(sparse_data: torch.Tensor, batched=False) -> torch.Tensor:
    """
    Converts sparse voxel representation back to a dense matrix.

    :param sparse_data: Sparse voxel data with shape (max_voxels, NUM_CLASSES + 3),
        columns 0-2 are normalised coordinates (x, y, z) in range [0,1],
        and columns 3-(NUM_CLASSES + 3) are one-hot encoded descriptor values
    :param batched: Bool denoting if sparse_data is batched
    :return: Dense voxel grid
    """
    if not batched:
        sparse_data = sparse_data.unsqueeze(0)

    shape = (sparse_data.size(0),) + (EXPANDED_GRID_SIZE,) * COORDINATE_DIMENSIONS
    dense_grid = torch.zeros(shape, dtype=torch.int64).to(DEVICE)

    for i, robot in enumerate(sparse_data):
        # Extract coordinates and descriptor values
        coor = robot[:, :COORDINATE_DIMENSIONS] * EXPANDED_GRID_SIZE  # Scale normalised coordinates back to grid indices, padded values encoded as EXPANDED GRID SIZE (outside of grid)
        coor_rounded = coor.round().long()
        coor = torch.clamp(coor_rounded, min=0, max=EXPANDED_GRID_SIZE)  # Clamp values to range [0, EXPANDED_GRID_SIZE] to avoid errors if not normalised correctly
        descriptors = robot[:, COORDINATE_DIMENSIONS:].argmax(dim=1)  # Convert one-hot encoding back to descriptor values

        for j in range(robot.shape[0]):
            x, y, z = coor[j]
            descriptor = descriptors[j]
            if descriptor != 0 and not torch.any(coor[j] == EXPANDED_GRID_SIZE):  # Skip padding descriptors and coordinates
                dense_grid[i, x, y, z] = descriptor

    if not batched:
        dense_grid = dense_grid.squeeze(0)

    return dense_grid
