"""
Program to visualise robot encodings.
Intended to be run using csv files with the following structure:
[id, 3D space dimensions, encoding values...]
- The "id" column uniquely identifies each robot.
- The "3D space dimensions" column defines the maximum allowable space for the robot components (e.g., 11 = 11x11x11).
- The remaining 11x11x11 (11^3) columns represent the 3D space structure:
  - A value of 0 indicates an empty location.
  - Other values correspond to specific robot components (e.g., wheel, joint, sensor, etc.).
The program visualises the encoded structure as a 3D plot for analysis and debugging.

Command to run:
python visualize_matrix_descriptor.py <path to the csv file> <id of the robot>
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # For legend
import numpy as np
import sys


def load_matrix_desc(filename: str, id: str):
    matrix_descs = []
    with open(filename) as file:
        lines = file.readlines()
        for line in lines:
            robot_encoding = line.split(',')
            # Checks id and skips iteration if not the one we want
            if robot_encoding[0] != id:
                continue
            i = 0
            j = 0
            row = []  # x-axis
            roww = []  # y-axis
            matrix = []  # z-axis (vertical)
            for elt in robot_encoding[2:]:
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
            return matrix


if __name__ == "__main__":

    filename = sys.argv[1]
    robot_id = sys.argv[2]
    matrix = np.array(load_matrix_desc(filename, robot_id))

    if matrix.size == 1:
        print(f"Robot ID not found.")
        sys.exit(1)

    # comments show files with no skeleton / files with skeleton
    colours = np.where(matrix == 1, "blue", matrix)  # wheel/skeleton
    colours = np.where(colours == '2', "green", colours)  # joint/wheel
    colours = np.where(colours == '3', "red", colours)  # caster/sensor
    colours = np.where(colours == '4', "yellow", colours)  # sensor/joint
    # colours = np.where(colours == '5', "pink", colours)  # not used/caster

    # and plot everything
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(matrix, facecolors=colours, edgecolor='k')

    # Create a legend
    legend_elements = [
        # When using file that includes the skeleton, change to commenbted values or add dynamic functionality
        mpatches.Patch(color="blue", label="Wheel"),  # "Skeleton"
        mpatches.Patch(color="green", label="Joint"),  # Wheel
        mpatches.Patch(color="red", label="Caster"),  # Sensor
        mpatches.Patch(color="yellow", label="Sensor")  # Joint
        # mpatches.Patch(color="pink", label="Caster")  # Caster (this value is not used with files without skeleton)
    ]

    # Add the legend to the plot
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')

    plt.show()
