import os

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt


def read_pickle(directory, filename, finger_index, validate=False):
    long_file_name = f"{directory}/{filename}_{finger_index}{'_validate' if validate else ''}.pkl"

    with open(long_file_name, "rb") as file:
        output = pickle.load(file)
        print(f"Loaded {type(output)} from {long_file_name}")

    return output


def plot_chosen_sdf_points(chosen_sdf_points, time_index, condition):
    """
    Plot points with sdf < 0 for a given time index and show a histogram of SDF values.

    Args:
        load_dir (str): Directory containing the sdf_points and sdf_values pickle files.
        finger_index (int): Index of the finger to load data for.
        time_index (int): Fixed time index to select data from.
    """

    # Plot 1: 3D scatter plot for points with sdf < 0
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.scatter(chosen_sdf_points[:, 0], chosen_sdf_points[:, 1], chosen_sdf_points[:, 2], c="red", marker="o", s=1)
    ax1.set_title(f"Points with {condition} (Time Index: {time_index})")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    plt.show()

    # Plot 2: Histogram of SDF values
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    ax2.hist(values, bins=50, color="blue", alpha=0.7)
    ax2.set_title(f"Histogram of SDF Values (Time Index: {time_index})")
    ax2.set_xlabel("SDF Value")
    ax2.set_ylabel("Frequency")
    plt.show()


def main(load_dir, finger_index, time_index):
    """
    Main function to plot negative sdf points and histogram.
    """

    # Load sdf_points and sdf_values
    sdf_points = read_pickle(load_dir, "sdf_points", finger_index)
    sdf_values = read_pickle(load_dir, "sdf_values", finger_index)
    # Extract data for the fixed time index
    points = sdf_points[time_index]
    values = sdf_values[time_index]

    # Filter points with sdf < 0 and sdf >= 0
    negative_sdf_points = points[values < 0]
    positive_sdf_points = points[values >= 0]

    # Print details
    print(f"Number of negative SDF points: {negative_sdf_points.shape[0]}")
    print(f"Number of positive SDF points: {positive_sdf_points.shape[0]}")

    plot_chosen_sdf_points(negative_sdf_points, time_index, "sdf < 0 ")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot SDF points and histogram for a given finger and time index.")
    parser.add_argument(
        "--finger_index",
        type=int,
        default=730,  # Default value for finger index
        help="Index of the finger to analyze (default: 730).",
    )
    parser.add_argument(
        "--time_index",
        type=int,
        default=40,  # Default value for time index
        help="Time index to analyze (default: 40).",
    )
    args = parser.parse_args()

    # Directory containing the sdf pickle files
    LOAD_DIR = "./training_data"

    # Call the main function
    main(LOAD_DIR, args.finger_index, args.time_index)
