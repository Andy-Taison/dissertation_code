"""
Functions to visualise robots
"""

import torch
import matplotlib
matplotlib.use('TkAgg')  # Backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # For legend
import numpy as np
from pathlib import Path
from ..config import PLOT_DIR


def load_grid_from_file(filepath: str, robot_id: int, robot_id_column_idx: int = 0) -> tuple[torch.Tensor, int]:
    """
    Retrieve grid data for 3D space data from file for robot_id specified.
    Intended to be used on raw (non-combined) csv files.
    If using for combined files, default robot_id_column_idx will refer to the first column (new combined ID),
    for original ID, use idx 2 but note there may be duplicate ID's, first found will be returned.

    :param filepath: Filepath
    :param robot_id: Robot ID to get grid data for
    :param robot_id_column_idx: Column index containing robot ID, default 0
    :return: Grid data as tensor, Row number
    """
    with open(filepath) as file:
        lines = file.readlines()
        for row_idx, line in enumerate(lines):
            robot_encoding = line.split(',')
            # Checks id and skips iteration if not the one we want
            if int(robot_encoding[robot_id_column_idx]) != robot_id:
                continue
            i = 0
            j = 0
            row = []  # x-axis
            roww = []  # y-axis
            matrix = []  # z-axis (vertical)
            for elt in robot_encoding[-1331:]:
                row.append(int(elt))
                i += 1
                if i == 11:
                    roww.append(row)
                    row = []
                    j += 1
                    i = 0
                if j == 11:
                    matrix.append(roww)
                    roww = []
                    j = 0

            return torch.tensor(matrix, dtype=torch.float32), row_idx + 1


def visualise_robot(grid_data: torch.Tensor, title: str = None, filename: str | Path = "robot") -> None:
    """
    Visualise robot from tensor of grid data.
    Saves as png in 'PLOT_DIR' as specified in config.

    :param grid_data: 11x11x11 3D space coordinate data
    :param title: Optional plot title
    :param filename: Filename to save generated visualisation. Stores in 'PLOT_DIR' as specified in config
    """
    # Verify grid data has 1331 elements (11x11x11 flattened)
    assert grid_data.numel() == 1331, "Grid data does not have the correct number of elements (1331)."

    # Reshape grid data to (11, 11, 11) and ensure integer type
    matrix = grid_data.view(11, 11, 11).numpy().astype(np.int32)

    # Comments show files with no skeleton / files with skeleton
    colours = np.where(matrix == 1, "blue", matrix)  # wheel/skeleton
    colours = np.where(colours == '2', "green", colours)  # joint/wheel
    colours = np.where(colours == '3', "red", colours)  # caster/sensor
    colours = np.where(colours == '4', "orange", colours)  # sensor/joint
    # colours = np.where(colours == '5', "pink", colours)  # not used/caster

    # Plot everything
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(matrix, facecolors=colours, edgecolor='k')

    # Set axis limits, can have rendering issues when left to automatic
    ax.set_xlim(0, 11.99)
    ax.set_ylim(0, 11.99)
    ax.set_zlim(0, 11.99)

    # Set title
    if title:
        ax.set_title(title, pad=20)

    # Create a legend
    legend_elements = [
        # When using files that include the skeleton, change to commented values or add dynamic functionality
        mpatches.Patch(color="blue", label="Wheel"),  # "Skeleton"
        mpatches.Patch(color="green", label="Joint"),  # Wheel
        mpatches.Patch(color="red", label="Caster"),  # Sensor
        mpatches.Patch(color="orange", label="Sensor")  # Joint
        # mpatches.Patch(color="pink", label="Caster")  # Caster (this value is not used with files without skeleton)
    ]

    # Add the legend to the plot
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')

    filepath = Path(PLOT_DIR) / f"{filename}.png"
    if not filepath.parent.exists():
        print(f"Creating directory '{filepath.parent}'...")
        filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath)
    print(f"Robot visualisation plot saved to '{filepath.name}.png'")

    plt.show()
    plt.close(fig)
