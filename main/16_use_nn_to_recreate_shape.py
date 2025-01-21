import pickle
import argparse
import os
import signal
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import copy
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D


from __TRAINING_FILE import MeshEncoder, SDFCalculator, TrainingContext, LATENT_DIM, DEFAULT_FINGER_INDEX

GRID_DIM = 30


# Directory containing the pickle files
LOAD_DIR = "./training_data"

# Directory where we save and load the neural weights
NEURAL_WEIGHTS_DIR = "./neural_weights"
FINGER_INDEX = 730


def read_pickle(directory, filename, finger_index, validate=False):
    long_file_name = f"{directory}/{filename}_{finger_index}{'_validate' if validate else ''}.pkl"
    print(long_file_name, "\n")

    with open(long_file_name, "rb") as file:
        output = pickle.load(file)
        print(f"Loaded {type(output)} from {long_file_name}")

    return output


def compute_small_bounding_box(mesh_points: np.ndarray) -> (np.ndarray, np.ndarray):
    """Compute the smallest bounding box for the vertices."""
    b_min = np.min(mesh_points, axis=0)
    b_max = np.max(mesh_points, axis=0)
    return b_min, b_max


def compute_enlarged_bounding_box(mesh_points: np.ndarray, box_ratio: float = 1.5) -> (np.ndarray, np.ndarray):
    """Compute an expanded bounding box for the vertices."""
    b_min, b_max = compute_small_bounding_box(mesh_points)

    center = (b_min + b_max) / 2
    half_lengths = (b_max - b_min) / 2

    # Expand the bounding box by the given ratio
    b_min = center - half_lengths * box_ratio
    b_max = center + half_lengths * box_ratio

    return b_min, b_max


def load_model_weights(encoder, calculator, epoch_index, time_index):
    """
    Placeholder function to load the weights for the encoder and calculator models.
    """
    # Replace with the actual mechanism to load weights, e.g., from files or a database.
    encoder_weights_path = os.path.join(NEURAL_WEIGHTS_DIR, f"encoder_epoch_{epoch_index}_time_{time_index}.pth")
    calculator_weights_path = os.path.join(NEURAL_WEIGHTS_DIR, f"calculator_epoch_{epoch_index}_time_{time_index}.pth")

    if os.path.exists(encoder_weights_path):
        encoder.load_state_dict(torch.load(encoder_weights_path))
        print(f"Loaded encoder weights from {encoder_weights_path}.")
    else:
        raise FileNotFoundError(f"Encoder weights not found at {encoder_weights_path}.")

    if os.path.exists(calculator_weights_path):
        calculator.load_state_dict(torch.load(calculator_weights_path))
        print(f"Loaded calculator weights from {calculator_weights_path}.")
    else:
        raise FileNotFoundError(f"Calculator weights not found at {calculator_weights_path}.")


def recreate_shape(mesh_encoder, sdf_calculator, time_index_visualise, vertices_tensor, sdf_points):
    """
    Recreate the shape by extracting the latent vector and using the SDF calculator.

    Args:
        mesh_encoder (MeshEncoder): Trained mesh encoder model.
        sdf_calculator (SDFCalculator): Trained SDF calculator model.
        vertices_tensor (torch.Tensor): Vertices tensor.
        faces (any): Faces information.
        time_index_visualise (int): Time index to visualize.
        sdf_points (torch.Tensor): Points for SDF computation.
    """
    # Extract the latent vector
    vertices = vertices_tensor[time_index_visualise].view(1, -1)  # Flatten the vertices
    latent_vector = mesh_encoder(vertices)

    # Convert latent vector to NumPy and print
    latent_vector_np = latent_vector.detach().cpu().numpy().flatten()
    print(f"Latent vector for time index {time_index_visualise}:")
    print(f"Shape: {latent_vector_np.shape}")
    print(f"Values:\n{latent_vector_np}")

    # Predict SDF values
    points = sdf_points[time_index_visualise].unsqueeze(0)
    predicted_sdf = sdf_calculator(latent_vector, points)  # (1, num_points, 1)

    # Convert predicted SDF values to NumPy and print
    predicted_sdf_np = predicted_sdf.detach().cpu().numpy().flatten()
    print("\nPredicted SDF values:")
    print(f"Shape: {predicted_sdf_np.shape}")
    print(f"Values:\n{predicted_sdf_np}")


def calculate_sdf_at_points(mesh_encoder, sdf_calculator, vertices_tensor, time_index, query_points_np) -> np.ndarray:
    """
    Calculate the SDF values at given query points using the trained models.

    Args:
        mesh_encoder (MeshEncoder): Trained mesh encoder model.
        sdf_calculator (SDFCalculator): Trained SDF calculator model.
        vertices_tensor (torch.Tensor): Vertices tensor (time_steps, num_vertices, 3).
        time_index (int): Time index to use for the mesh encoder.
        query_points_np (np.ndarray): Query points to calculate SDF, shape (N, 3).

    Returns:
        np.ndarray: Predicted SDF values for the query points, shape (N, 1).
    """
    # Ensure the query points are a PyTorch tensor
    query_points = torch.tensor(query_points_np, dtype=torch.float32).unsqueeze(0)  # Shape (1, N, 3)

    # Extract the latent vector from the mesh encoder
    vertices = vertices_tensor[time_index].view(1, -1)  # Flatten the vertices
    latent_vector = mesh_encoder(vertices)  # Shape (1, latent_dim)

    # Use the SDF calculator to predict the SDF values
    predicted_sdf = sdf_calculator(latent_vector, query_points)  # Shape (1, N, 1)

    # Convert the predicted SDF values to a NumPy array and reshape
    predicted_sdf_np = predicted_sdf.detach().cpu().numpy().squeeze()  # Shape (N,)

    return predicted_sdf_np.flatten()  # Shape (N, )


def create_3d_points_within_bbox(b_min, b_max, num_points_per_axis):
    """
    Create a 3D grid of points within the specified bounding box.

    Args:
        b_min (array-like): Minimum coordinates of the bounding box [x_min, y_min, z_min].
        b_max (array-like): Maximum coordinates of the bounding box [x_max, y_max, z_max].
        num_points_per_axis (int): Number of points to generate along each axis.

    Returns:
        numpy.ndarray: Array of shape (N, 3) containing the 3D points within the bounding box.
    """
    # Generate linearly spaced points along each axis
    x = np.linspace(b_min[0], b_max[0], num_points_per_axis)
    y = np.linspace(b_min[1], b_max[1], num_points_per_axis)
    z = np.linspace(b_min[2], b_max[2], num_points_per_axis)

    # Create a 3D meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Flatten the meshgrid arrays and stack them into an (N, 3) array
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    return points


def recreate_mesh(sdf_grid, N, b_min, b_max):
    """
    Recreate the 3D mesh from the SDF grid using the Marching Cubes algorithm.

    Args:
        sdf_grid (np.ndarray): Flattened array of SDF values.
        N (int): Number of points along each axis (resolution).
        b_min (np.ndarray): Minimum coordinates of the bounding box.
        b_max (np.ndarray): Maximum coordinates of the bounding box.

    Returns:
        verts (np.ndarray): Vertices of the reconstructed mesh.
        faces (np.ndarray): Faces of the reconstructed mesh.
    """
    # Reshape the flat sdf_grid into a 3D array
    sdf_3d = sdf_grid.reshape((N, N, N))

    # Apply the Marching Cubes algorithm to extract the isosurface
    verts, faces, normals, values = measure.marching_cubes(sdf_3d, level=0)

    # Scale and translate the vertices to the original bounding box
    scale = b_max - b_min
    verts = verts / (N - 1)  # Normalize to [0, 1]
    verts = verts * scale + b_min  # Scale and translate to original bbox

    return verts, faces


def visualize_mesh(verts, faces):
    """
    Visualize the 3D mesh using Matplotlib.

    Args:
        verts (np.ndarray): Vertices of the mesh.
        faces (np.ndarray): Faces of the mesh.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Create a Poly3DCollection from the vertices and faces
    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    mesh.set_edgecolor("k")
    ax.add_collection3d(mesh)

    # Set plot limits
    ax.set_xlim([verts[:, 0].min(), verts[:, 0].max()])
    ax.set_ylim([verts[:, 1].min(), verts[:, 1].max()])
    ax.set_zlim([verts[:, 2].min(), verts[:, 2].max()])

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()


def visualize_sdf_points(query_points, sdf_values, threshold=0):
    """
    Visualize the 3D points with coloring based on the SDF sign.

    Args:
        query_points (np.ndarray): 3D points of shape (N, 3).
        sdf_values (np.ndarray): SDF values of shape (N,).
        threshold (float): Threshold for separating points into two groups.
                           Default is 0 for isosurface visualization.
    """
    # Separate points based on SDF value
    inside_points = query_points[sdf_values < threshold]
    outside_points = query_points[sdf_values >= threshold]

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot inside points
    ax.scatter(inside_points[:, 0], inside_points[:, 1], inside_points[:, 2], c="blue", label=f"SDF < {threshold}", alpha=0.6, s=1)

    # Plot outside points
    ax.scatter(outside_points[:, 0], outside_points[:, 1], outside_points[:, 2], c="red", label=f"SDF >= {threshold}", alpha=0.6, s=1)

    # Add labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.tight_layout()
    plt.show()


def main(epoch_index=100, time_index=0, finger_index=DEFAULT_FINGER_INDEX):
    vertices_tensor_np = read_pickle(LOAD_DIR, "vertices_tensor", finger_index)
    faces = read_pickle(LOAD_DIR, "vertices_tensor", finger_index)
    sdf_points = read_pickle(LOAD_DIR, "sdf_points", finger_index)
    sdf_values = read_pickle(LOAD_DIR, "sdf_values", finger_index)

    # Convert inputs to PyTorch tensors
    vertices_tensor = torch.tensor(vertices_tensor_np, dtype=torch.float32)  # (time_steps, num_vertices, 3)
    sdf_points = torch.tensor(sdf_points, dtype=torch.float32)  # (time_steps, num_points, 3)
    sdf_values = torch.tensor(sdf_values, dtype=torch.float32).unsqueeze(-1)  # (time_steps, num_points, 1)
    input_dim = vertices_tensor.shape[1] * vertices_tensor.shape[2]  # num_vertices * 3

    number_of_shape_per_familly = sdf_points.shape[0]
    mesh_encoder = MeshEncoder(input_dim=input_dim, latent_dim=LATENT_DIM)
    sdf_calculator = SDFCalculator(latent_dim=LATENT_DIM)
    training_context = TrainingContext(mesh_encoder, sdf_calculator, finger_index, number_of_shape_per_familly, 0.1)
    training_context.load_model_weights(epoch_index, time_index)

    mesh_encoder = training_context.mesh_encoder
    sdf_calculator = training_context.sdf_calculator

    time_index_visualise = 0
    # recreate_shape(mesh_encoder, sdf_calculator, time_index_visualise, vertices_tensor, sdf_points)

    b_min, b_max = compute_small_bounding_box(vertices_tensor_np[time_index_visualise])
    print("\n")
    print(f"b_min = {b_min}\nb_max = {b_max}")

    query_points = create_3d_points_within_bbox(b_min, b_max, GRID_DIM)
    sdf_grid = calculate_sdf_at_points(mesh_encoder, sdf_calculator, vertices_tensor, time_index_visualise, query_points)
    print("")
    print(f"np.shape(sdf_grid) = {np.shape(sdf_grid)}")
    print(f"sdf_grid = {sdf_grid}\n")

    visualize_sdf_points(query_points, sdf_grid)
    exit(0)

    verts, faces = recreate_mesh(sdf_grid, GRID_DIM, b_min, b_max)
    visualize_mesh(verts, faces)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="use a trained model to recreate shapes")
    # Arguments for epoch and time indices
    parser.add_argument("--epoch_index", type=int, help="Specify the epoch index to recreate the shape from")
    parser.add_argument("--time_index", type=int, help="Specify the time index of processing to recreate the shape from")
    parser.add_argument("--finger_index", type=int, help="Specify the finger index where the force was applied")
    args = parser.parse_args()

    if args.epoch_index is None and args.time_index is None:
        epoch_index, time_index = 80, 0
    else:
        epoch_index, time_index = args.epoch_index, args.time_index

    if args.finger_index is None:
        finger_index = DEFAULT_FINGER_INDEX
    else:
        finger_index = args.finger_index

    ret = main(epoch_index, time_index, finger_index)
    exit(ret)
