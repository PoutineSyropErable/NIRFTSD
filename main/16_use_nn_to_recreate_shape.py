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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv


from __TRAINING_FILE import MeshEncoder, SDFCalculator, TrainingContext, LATENT_DIM, DEFAULT_FINGER_INDEX, NEURAL_WEIGHTS_DIR

GRID_DIM = 100


# Directory containing the pickle files
LOAD_DIR = "./training_data"

# Directory where we save and load the neural weights
FINGER_INDEX = 730


def save_pickle(path: str, object1):
    with open(path, "wb") as f:
        pickle.dump(object1, f)


def load_pickle(path: str):
    with open(path, "rb") as file:
        output = pickle.load(file)
        print(f"Loaded {type(output)} from {path}")

    return output


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


def calculate_sdf_at_points(mesh_encoder, sdf_calculator, vertices_tensor, I, query_points_np) -> np.ndarray:
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
    vertices = vertices_tensor[I].view(1, -1)  # Flatten the vertices
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


def create_sphere(center, radius, resolution=20):
    """
    Generate vertices for a sphere.

    Args:
        center (tuple): Coordinates of the sphere's center (x, y, z).
        radius (float): Radius of the sphere.
        resolution (int): Number of points to define the sphere (higher means smoother).

    Returns:
        tuple: Vertices (x, y, z) of the sphere.
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def visualize_mesh_old(verts, faces, finger_position, R):
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

    # Determine the cubic bounds
    x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    z_min, z_max = verts[:, 2].min(), verts[:, 2].max()

    # Find the range and center
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    center_x = (x_max + x_min) / 2.0
    center_y = (y_max + y_min) / 2.0
    center_z = (z_max + z_min) / 2.0

    # Set cubic limits
    ax.set_xlim(center_x - max_range, center_x + max_range)
    ax.set_ylim(center_y - max_range, center_y + max_range)
    ax.set_zlim(center_z - max_range, center_z + max_range)
    # Add a sphere at the finger position
    x, y, z = create_sphere(finger_position, R)
    ax.plot_surface(x, y, z, color="r", alpha=0.8)

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()


def convert_faces_to_pyvista_format(faces):
    """
    Convert face indices to PyVista-compatible format.

    Args:
        faces (np.ndarray): Array of face indices (Nx3 for triangles).

    Returns:
        np.ndarray: Flattened array with the number of vertices prepended for each face.
    """
    n_faces = faces.shape[0]
    flat_faces = np.hstack([[3] * n_faces, faces.flatten()])
    # Validate the length
    expected_size = n_faces * (3 + 1)
    if len(flat_faces) != expected_size:
        raise ValueError(f"Unexpected size of flattened faces array: {len(flat_faces)} != {expected_size}")
    return flat_faces


def visualize_mesh(verts, faces, finger_position, R):
    """
    Visualize the 3D mesh and a sphere at the finger position using PyVista.

    Args:
        verts (np.ndarray): Vertices of the mesh (Nx3).
        faces (np.ndarray): Faces of the mesh (Nx3).
        finger_position (tuple): Coordinates of the sphere's center (x, y, z).
        R (float): Radius of the sphere.
    """
    # Convert faces to PyVista-compatible format
    pyvista_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()

    # Create the PyVista mesh
    mesh = pv.PolyData(verts, pyvista_faces)

    # Create the sphere at the finger position
    sphere = pv.Sphere(radius=R, center=finger_position)

    # Initialize PyVista plotter
    plotter = pv.Plotter()

    # Add the mesh to the plotter
    plotter.add_mesh(mesh, color="lightblue", show_edges=True, opacity=0.7)

    # Add the sphere to the plotter
    plotter.add_mesh(sphere, color="red", opacity=0.8)

    # Set plotter properties
    plotter.set_background("white")
    plotter.add_axes()
    plotter.show_grid()

    # Show the plot
    plotter.show()


def visualize_mesh_list(mesh_list, finger_position, R, output_file="mesh_animation.mp4", offscreen=False):
    """
    Animate the 3D meshes as an animation and a sphere at the finger position using PyVista.

    Args:
        mesh_list (list of pv.PolyData): List of PyVista meshes.
        finger_position (tuple): Coordinates of the sphere's center (x, y, z).
        R (float): Radius of the sphere.
        output_file (str): Path to save the animation (MP4 format).
        offscreen (bool): Whether to enable offscreen rendering.
    """
    # Initialize PyVista plotter
    plotter = pv.Plotter(off_screen=offscreen)
    plotter.add_axes()
    plotter.show_grid()

    # Add a title
    # plotter.add_text("Mesh Animation", font_size=12)

    # Add the static sphere at the finger position
    finger_marker = pv.Sphere(radius=R, center=finger_position)
    plotter.add_mesh(finger_marker, color="red", label="Finger Position", opacity=0.8)

    # Add the first mesh to initialize
    actor = plotter.add_mesh(mesh_list[0], color="lightblue", show_edges=True)

    # Open the movie file for writing
    plotter.open_movie(output_file, framerate=20)

    # Animate through all meshes in the list
    for i, mesh in enumerate(mesh_list):
        # Update the actor with the current mesh's geometry
        actor.GetMapper().SetInputData(mesh)

        # Optional: Add a progress label
        plotter.add_text(f"Frame: {i + 1}/{len(mesh_list)}", name="frame-label", font_size=10)

        # Write the current frame to the movie
        plotter.write_frame()

    # Close the plotter and save the movie
    plotter.close()
    print(f"Animation saved to {output_file}")


def main(epoch_index=100, time_index=0, finger_index=DEFAULT_FINGER_INDEX, visualize_index=0):
    vertices_tensor_np = read_pickle(LOAD_DIR, "vertices_tensor", finger_index)[:-1]
    faces = read_pickle(LOAD_DIR, "vertices_tensor", finger_index)[:-1]
    sdf_points = read_pickle(LOAD_DIR, "sdf_points", finger_index)[:-1]
    sdf_values = read_pickle(LOAD_DIR, "sdf_values", finger_index)[:-1]

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

    latent_vector_list = []
    for I in range(len(vertices_tensor) - 1):
        # Extract the latent vector from the mesh encoder
        vertices = vertices_tensor[I].view(1, -1)  # Flatten the vertices
        latent_vector = mesh_encoder(vertices)  # Shape (1, latent_dim)
        latent_vector_np = latent_vector.detach().cpu().numpy().flatten()
        latent_vector_list.append(latent_vector_np)
        # print(f"Latent vector at index {I}: {latent_vector_np}")

    # Calculate the standard deviation of each latent dimension across time

    print("\n\n")
    latent_vectors = np.array(latent_vector_list).T
    print(np.shape(latent_vectors))
    print(latent_vectors)

    print("\n\n")
    stdl = []
    for latent_vector_coord in latent_vectors:
        stdl.append(np.std(latent_vector_coord))

    stdl = np.array(stdl)
    print("each element standard deviation=\n")
    print(stdl)

    # Add plot title and labels
    plt.title("First 5 Latent Vectors", fontsize=16)
    plt.xlabel("Latent Dimension Index", fontsize=12)
    plt.ylabel("Latent Value", fontsize=12)

    # Add legend and grid
    plt.legend(loc="upper right")
    plt.grid(True)

    # Show the plot
    plt.show()

    if True:
        b_min, b_max = compute_enlarged_bounding_box(vertices_tensor_np[visualize_index])
        print("\n")
        print(f"b_min = {b_min}\nb_max = {b_max}")

        # File containing finger_positions (after filtering)
        FINGER_POSITIONS_FILES = "filtered_points_of_force_on_boundary.txt"
        finger_positions = np.loadtxt(FINGER_POSITIONS_FILES, skiprows=1)
        # Swap Y and Z because poylscope uses weird data
        # finger_positions[:, [1, 2]] = finger_positions[:, [2, 1]]
        finger_position = finger_positions[finger_index]
        R = 0.003  # Radius of the FINGER

        query_points = create_3d_points_within_bbox(b_min, b_max, GRID_DIM)
        sdf_grid = calculate_sdf_at_points(mesh_encoder, sdf_calculator, vertices_tensor, visualize_index, query_points)
        print("")
        print(f"np.shape(sdf_grid) = {np.shape(sdf_grid)}")
        print(f"sdf_grid = {sdf_grid}\n")

        # visualize_sdf_points(query_points, sdf_grid)

        verts, faces = recreate_mesh(sdf_grid, GRID_DIM, b_min, b_max)

        print(f"np.shape(verts) = {np.shape(verts)}")
        print(f"np.shape(faces) = {np.shape(faces)}")
        print(f"faces = {faces}")
        print(f"verts = {verts}")
        # visualize_mesh(verts, faces, finger_position, R)

    verts_list_path = "verts_list.pkl"
    faces_list_path = "faces_list.pkl"
    n = 100
    if os.path.exists(verts_list_path) and os.path.exists(faces_list_path):
        # Load pre-processed mesh data
        verts_list = load_pickle(verts_list_path)
        faces_list = load_pickle(faces_list_path)
        I = 0
        for verts, faces in zip(verts_list, faces_list):
            # print(f"np.shape(verts) = {np.shape(verts)}")
            # print(f"np.shape(faces) = {np.shape(faces)}")
            print(f"a = {vertices_tensor[I][0][0]}")
            I += 1
            pass
    else:
        verts_list = []
        faces_list = []
        for I in range(n):
            # Compute bounding box for the current shape
            b_min, b_max = compute_enlarged_bounding_box(vertices_tensor_np[visualize_index])
            # print("\n")
            # print(f"Shape {visualize_index + 1}")
            # print(f"b_min = {b_min}\nb_max = {b_max}")

            query_points = create_3d_points_within_bbox(b_min, b_max, GRID_DIM)
            sdf_grid = calculate_sdf_at_points(mesh_encoder, sdf_calculator, vertices_tensor, I, query_points)
            # print("")
            # print(f"np.shape(sdf_grid) = {np.shape(sdf_grid)}")
            # print(f"sdf_grid = {sdf_grid}\n")

            # Recreate the mesh for the current shape
            verts, faces = recreate_mesh(sdf_grid, GRID_DIM, b_min, b_max)

            # print(f"np.shape(verts) = {np.shape(verts)}")
            # print(f"np.shape(faces) = {np.shape(faces)}")
            print(f"a = {vertices_tensor[I][0][0]}")
            verts_list.append(verts)
            faces_list.append(faces)

    # Visualize the current shape
    save_pickle(verts_list_path, verts_list)
    save_pickle(faces_list_path, faces_list)

    mesh_list = []
    for verts, faces in zip(verts_list, faces_list):
        # print(f"np.shape(verts) = {np.shape(verts)}")
        # print(f"np.shape(faces) = {np.shape(faces)}")
        pyvista_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()
        mesh = pv.PolyData(verts, pyvista_faces)
        mesh_list.append(mesh)

    save_pickle("mesh_list.pkl", mesh_list)
    visualize_mesh_list(mesh_list, finger_position, R)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="use a trained model to recreate shapes")
    # Arguments for epoch and time indices
    parser.add_argument("--epoch_index", type=int, default=2, help="Specify the epoch index to recreate the shape from")
    parser.add_argument("--time_index", type=int, default=0, help="Specify the time index of processing to recreate the shape from")
    parser.add_argument("--finger_index", type=int, default=730, help="Specify the finger index where the force was applied")
    parser.add_argument("--visualize_index", type=int, default=0, help="Specify the finger index where the force was applied")
    args = parser.parse_args()

    ret = main(args.epoch_index, args.time_index, args.finger_index, args.visualize_index)
    exit(ret)
