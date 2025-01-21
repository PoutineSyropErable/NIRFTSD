import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def read_pickle(directory, filename, finger_index, validate=False):
    long_file_name = f"{directory}/{filename}_{finger_index}{'_validate' if validate else ''}.pkl"

    with open(long_file_name, "rb") as file:
        output = pickle.load(file)
        print(f"Loaded {type(output)} from {long_file_name}")

    return output


def plot_negative_sdf_points(load_dir, finger_index, time_index):
    """
    Plot points with sdf < 0 for a given time index and show a histogram of SDF values.

    Args:
        load_dir (str): Directory containing the sdf_points and sdf_values pickle files.
        finger_index (int): Index of the finger to load data for.
        time_index (int): Fixed time index to select data from.
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

    # Plot the points with sdf < 0
    fig = plt.figure(figsize=(10, 5))

    # 3D scatter plot
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(negative_sdf_points[:, 0], negative_sdf_points[:, 1], negative_sdf_points[:, 2], c="red", marker="o", s=1)
    ax1.set_title(f"Points with sdf < 0 (Time Index: {time_index})")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Histogram of SDF values
    ax2 = fig.add_subplot(122)
    ax2.hist(values, bins=50, color="blue", alpha=0.7)
    ax2.set_title(f"Histogram of SDF Values (Time Index: {time_index})")
    ax2.set_xlabel("SDF Value")
    ax2.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Directory containing the sdf pickle files
    LOAD_DIR = "./training_data"

    # Finger index and time index to analyze
    FINGER_INDEX = 730  # Replace with your desired finger index
    TIME_INDEX = 0  # Replace with your desired time index

    # Plot the points with sdf < 0
    plot_negative_sdf_points(LOAD_DIR, FINGER_INDEX, TIME_INDEX)
