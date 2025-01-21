import numpy as np
import igl
import h5py
import pyvista as pv
from dolfinx.io.utils import XDMFFile
from dolfinx import mesh
from mpi4py import MPI
from scipy.optimize import root_scalar
from typing import Tuple
import argparse
import os, sys
import time
import matplotlib.pyplot as plt
import polyscope as ps
import pickle


BUNNY_FILE = "bunny.xdmf"
NUMBER_OF_POINTS_IN_VISUALISATION = 10_000


def load_file(filename: str) -> mesh.Mesh:
    """
    Load the mesh from an XDMF file.

    Parameters:
        filename (str): Path to the XDMF file.

    Returns:
        mesh.Mesh: The loaded mesh object.
    """
    with XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        domain: mesh.Mesh = xdmf.read_mesh(name="Grid")
        print("Mesh loaded successfully!")
    return domain


def extract_boundary_info(domain) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the boundary faces and vertex indices from a tetrahedral mesh.

    Args:
        domain (dolfinx.mesh.Mesh): The input tetrahedral mesh.

    Returns:
        faces (np.ndarray): Triangular faces on the boundary (each row contains 3 vertex indices).
        vertex_index (np.ndarray): Indices of the vertices on the boundary.
    return faces, boundary_vertices_index
    """
    # Step 1: Locate boundary facets
    tdim = domain.topology.dim  # Topological dimension (tetrahedra -> 3D)
    fdim = tdim - 1  # Facet dimension (boundary faces -> 2D)

    # Get facets on the boundary
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))

    # Step 2: Get facet-to-vertex connectivity
    facet_to_vertex = domain.topology.connectivity(fdim, 0)
    if facet_to_vertex is None:
        raise ValueError("Facet-to-vertex connectivity not available. Ensure the mesh is initialized correctly.")

    # Map boundary facets to vertex indices
    boundary_faces = [facet_to_vertex.links(facet) for facet in boundary_facets]

    # Step 3: Flatten and extract unique boundary vertex indices
    boundary_vertices_index = np.unique(np.hstack(boundary_faces))

    # Map original vertex indices to continuous indices (0-based for faces)
    vertex_map = {original: i for i, original in enumerate(boundary_vertices_index)}
    faces = np.array([[vertex_map[v] for v in face] for face in boundary_faces], dtype=int)

    return faces, boundary_vertices_index


def get_array_from_conn(conn) -> np.ndarray:
    """
    Convert mesh topology connectivity to a 2D numpy array.

    Parameters:
        conn: The mesh topology connectivity (dolfinx mesh.topology.connectivity).

    Returns:
        np.ndarray: A 2D numpy array where each row contains the vertex indices for a cell.
    """
    connectivity_array = conn.array
    offsets = conn.offsets

    # Convert the flat connectivity array into a list of arrays
    connectivity_2d = [connectivity_array[start:end] for start, end in zip(offsets[:-1], offsets[1:])]

    return np.array(connectivity_2d, dtype=object)


def get_mesh(filename: str) -> Tuple[mesh.Mesh, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract points and connectivity from the mesh.

    Parameters:
        filename (str): Path to the XDMF file.

    Returns:
        Tuple[mesh.Mesh, np.ndarray*4]: The mesh object, points, and connectivity array.

    return domain, points, connectivity, faces, boundary_vertices_index
    """
    domain = load_file(filename)
    points = domain.geometry.x  # Array of vertex coordinates
    conn = domain.topology.connectivity(3, 0)
    connectivity = get_array_from_conn(conn).astype(np.int64)  # Convert to 2D numpy array

    faces, boundary_vertices_index = extract_boundary_info(domain)

    return domain, points, connectivity, faces, boundary_vertices_index


def load_deformations(h5_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load deformation data from an HDF5 file.

    Parameters:
        h5_file (str): Path to the HDF5 file containing deformation data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: An array of time steps and a 3D tensor of displacements [time_index][point_index][x, y, z].
    """
    with h5py.File(h5_file, "r") as f:
        # Access the 'Function' -> 'f' group
        function_group = f["Function"]
        f_group = function_group["f"]

        # Extract time steps and displacements
        time_steps = np.array(sorted(f_group.keys(), key=lambda x: float(x)), dtype=float)
        displacements = np.array([f_group[time_step][...] for time_step in f_group.keys()])
        print(f"Loaded {len(time_steps)} time steps, Displacement tensor shape: {displacements.shape}")

    return time_steps, displacements


def load_mesh_and_deformations(
    xdmf_file: str, h5_file: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load mesh points and deformation data.

    Parameters:
        xdmf_file (str): Path to the XDMF file for the mesh.
        h5_file (str): Path to the HDF5 file for deformation data.

    Returns:
        Tuple[np.ndarray*7]: The mesh points, connectivity, time steps, and deformation tensor.

    return points, connectivity, time_steps, deformations, faces, boundary_vertices_index
    """
    # Load the mesh
    _, points, connectivity, faces, boundary_vertices_index = get_mesh(xdmf_file)

    # Load the deformations
    time_steps, deformations = load_deformations(h5_file)

    return points, connectivity, time_steps, deformations, faces, boundary_vertices_index


def compute_small_bounding_box(mesh_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the smallest bounding box for the vertices.
    b_min = [x_min, y_min, z_min]
    b_max = [x_max, y_max, z_max]

    return b_min, b_max
    """

    b_min = np.min(mesh_points, axis=0)
    b_max = np.max(mesh_points, axis=0)
    return b_min, b_max


def compute_enlarged_bounding_box(mesh_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute an enlarged bounding box for the vertices.
    It has the same center, but 1.5x bigger
    b_min = [x_min_tilda, y_min_tilda, z_min_tilda]
    b_max = [x_max_tilda, y_max_tilda, z_max_tilda]

    return b_min, b_max
    """
    b_min = mesh_points.min(axis=0)
    b_max = mesh_points.max(axis=0)

    BOX_RATIO = 1.5
    center = (b_min + b_max) / 2
    half_lengths = (b_max - b_min) / 2 * BOX_RATIO
    b_min = center - half_lengths
    b_max = center + half_lengths

    return b_min, b_max


def process_sdf_data(input_file):
    """
    Process SDF data from an HDF5 file and print uniform-sized numpy tensors.

    Args:
        input_file (str): Path to the input HDF5 file containing SDF data.
    """
    # Open the input HDF5 file
    with h5py.File(input_file, "r") as f:
        time_steps = sorted(f.keys())
        print(f"Found {len(time_steps)} time steps.")

        # Read all datasets to determine the minimum number of points
        datasets = [f[time_step][:] for time_step in time_steps]

        number_of_filtered_points_at_timestep = [data.shape[0] for data in datasets]
        min_points = min(number_of_filtered_points_at_timestep)
        print(f"The data len is {number_of_filtered_points_at_timestep}")
        print(f"Minimum number of points across time steps: {min_points}")

        # Create uniform-sized tensors
        sdf_points = np.zeros((len(time_steps), min_points, 3))
        sdf_values = np.zeros((len(time_steps), min_points))

        for t_index, data in enumerate(datasets):
            sdf_points[t_index] = data[:min_points, :3]  # Extract x, y, z
            sdf_values[t_index] = data[:min_points, 3]  # Extract sdf

        return sdf_points, sdf_values


def resize_to_minimum(sdf_points, sdf_values, sdf_points_validate, sdf_values_validate):
    """
    Resize the training and validation SDF datasets to the minimum number of points across all datasets.

    Args:
        sdf_points (np.ndarray): Training SDF points of shape (time_steps, num_points, 3).
        sdf_values (np.ndarray): Training SDF values of shape (time_steps, num_points).
        sdf_points_validate (np.ndarray): Validation SDF points of shape (time_steps, num_points, 3).
        sdf_values_validate (np.ndarray): Validation SDF values of shape (time_steps, num_points).

    Returns:
        Tuple: Resized (sdf_points, sdf_values, sdf_points_validate, sdf_values_validate).
    """
    # Determine the minimum number of points across all datasets
    min_points = min(sdf_points.shape[1], sdf_points_validate.shape[1], sdf_values.shape[1], sdf_values_validate.shape[1])

    print(f"Resizing datasets to minimum number of points: {min_points}")

    # Resize datasets to the minimum number of points
    sdf_points = sdf_points[:, :min_points, :]
    sdf_values = sdf_values[:, :min_points]
    sdf_points_validate = sdf_points_validate[:, :min_points, :]
    sdf_values_validate = sdf_values_validate[:, :min_points]

    return sdf_points, sdf_values, sdf_points_validate, sdf_values_validate


def show_mesh(faces: np.ndarray, vertices: np.ndarray):
    """
    Visualize a 3D mesh using PyVista.

    Args:
        vertices (np.ndarray): Array of shape (num_vertices, 3) representing the vertex coordinates.
        faces (np.ndarray): Array of shape (num_faces, 3) representing the triangular face indices.
    """
    if vertices.shape[1] != 3:
        raise ValueError("Vertices array must have shape (num_vertices, 3).")
    if faces.shape[1] != 3:
        raise ValueError("Faces array must have shape (num_faces, 3).")

    # PyVista requires a specific format for faces
    # Each face is prefixed with the number of vertices in the face
    pyvista_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()

    # Create a PyVista mesh
    mesh = pv.PolyData(vertices, pyvista_faces)

    # Create a PyVista plotter
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, color="lightblue")
    plotter.add_axes()
    plotter.add_title("3D Mesh Visualization")

    # Show the plot
    plotter.show()


def animate_deformation(faces: np.ndarray, deformed_boundary_points: np.ndarray):
    """
    Animate the deformation of a surface mesh using PyVista.

    Parameters:
        faces (np.ndarray): Faces of the surface mesh (connectivity array).
        deformed_boundary_points (np.ndarray): Deformed boundary points at each time step
                                               of shape (time_steps, num_points, 3).
    """
    output_file = "deformation_animation.mp4"  # Define the output file path

    # Create PyVista surface mesh for the initial frame
    initial_points = deformed_boundary_points[0]
    num_faces = faces.shape[0]
    cells = np.hstack([np.full((num_faces, 1), faces.shape[1]), faces]).flatten()
    surface_mesh = pv.PolyData(initial_points, cells)

    # Initialize PyVista plotter
    plotter = pv.Plotter(off_screen=False)
    plotter.add_mesh(surface_mesh, show_edges=True, scalars=None, colormap="coolwarm")
    plotter.add_axes()
    plotter.add_text("Deformation Animation", font_size=12)
    plotter.open_movie(output_file, framerate=20)

    # Animate through all time steps
    for t_index in range(deformed_boundary_points.shape[0]):
        # Update the mesh with the current deformation
        surface_mesh.points = deformed_boundary_points[t_index]
        plotter.write_frame()  # Save the frame to the animation

        print(f"Rendered time step {t_index + 1}/{deformed_boundary_points.shape[0]}")

    # Finalize and close the plotter
    plotter.close()
    print(f"Animation saved to {output_file}")


def plot_histograms_with_function(point_list, b_min, b_max, n, b):
    """
    Plot histograms for x, y, and z from the point list and overlay the normalized PDF for z.

    Args:
        point_list (np.ndarray): Array of shape (num_points, 3) with points (x, y, z).
        b_min (np.ndarray): Minimum bounds [min_x, min_y, min_z].
        b_max (np.ndarray): Maximum bounds [max_x, max_y, max_z].
        n (float): Exponent for the distribution function.
        b (float): Constant for the distribution function.
    """
    x, y, z = point_list[:, 0], point_list[:, 1], point_list[:, 2]

    # Create DistributionFunction for z
    z_min, z_max = b_min[2], b_max[2]
    distribution = DistributionFunction(n=n, b=b, z_min=z_min, z_max=z_max)

    # Generate z values for the function plot
    z_values = np.linspace(z_min, z_max, 1000)
    f_values = distribution.pdf(z_values)

    # Plot histograms
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    data = [x, y, z]
    labels = ["x", "y", "z"]
    bounds = [(b_min[0], b_max[0]), (b_min[1], b_max[1]), (z_min, z_max)]

    for i, ax in enumerate(axes):
        # Calculate the histogram for scaling
        hist, bin_edges = np.histogram(data[i], bins=50, range=bounds[i])
        bin_width = bin_edges[1] - bin_edges[0]

        # Plot the histogram
        ax.hist(data[i], bins=50, range=bounds[i], alpha=0.7, color="blue", label=f"Histogram of {labels[i]}")
        ax.set_title(f"{labels[i]} Histogram", fontsize=14)
        ax.set_xlabel(labels[i], fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)

        # Overlay the function for z
        if labels[i] == "z":
            # Scale f_values to match histogram scale
            f_values_scaled = f_values * len(z) * bin_width
            ax.plot(z_values, f_values_scaled, color="red", label=r"$f(z) = a(z - z_{min})^n + b$", linewidth=2)
            ax.legend(fontsize=10)

        # Add bounds as vertical lines
        ax.axvline(bounds[i][0], color="green", linestyle="--", linewidth=1.5, label=f"{labels[i]} min")
        ax.axvline(bounds[i][1], color="red", linestyle="--", linewidth=1.5, label=f"{labels[i]} max")
        ax.legend()

    plt.tight_layout()
    plt.show()


def draw_bounding_box(b_min: np.ndarray, b_max: np.ndarray, name: str, color: tuple = (0.0, 1.0, 0.0), radius: float = 0.002):
    """Draw a bounding box in Polyscope given min and max points."""
    # Create corners of the bounding box
    box_corners = np.array(
        [
            [b_min[0], b_min[1], b_min[2]],
            [b_max[0], b_min[1], b_min[2]],
            [b_max[0], b_max[1], b_min[2]],
            [b_min[0], b_max[1], b_min[2]],
            [b_min[0], b_min[1], b_max[2]],
            [b_max[0], b_min[1], b_max[2]],
            [b_max[0], b_max[1], b_max[2]],
            [b_min[0], b_max[1], b_max[2]],
        ]
    )

    # Define edges for the bounding box
    box_edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],  # Bottom face
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],  # Top face
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],  # Vertical edges
        ]
    )

    # Register the bounding box as a curve network
    ps_bounding_box = ps.register_curve_network(name, box_corners, box_edges)
    ps_bounding_box.set_radius(radius)
    ps_bounding_box.set_color(color)


def visualize_mesh_with_points(mesh_vertices, mesh_faces, sdf_points, sdf_values, NUMBER_OF_POINTS_IN_VISUALISATION):
    """
    Visualize the mesh, bounding boxes, and SDF points in Polyscope.

    Args:
        mesh_vertices (np.ndarray): Vertices of the mesh.
        mesh_faces (np.ndarray): Faces of the mesh (connectivity).
        sdf_points (np.ndarray): Points where the SDF is computed.
        sdf_values (np.ndarray): SDF values corresponding to sdf_points.
        NUMBER_OF_POINTS_IN_VISUALISATION (int): Number of SDF points to visualize.
    """
    # Ensure the number of points to visualize does not exceed the available points
    NUMBER_OF_POINTS_IN_VISUALISATION = min(NUMBER_OF_POINTS_IN_VISUALISATION, sdf_points.shape[0])

    # Select a subset of SDF points and values
    sdf_points_subset = sdf_points[:NUMBER_OF_POINTS_IN_VISUALISATION].copy()
    sdf_values_subset = sdf_values[:NUMBER_OF_POINTS_IN_VISUALISATION].copy()

    # Make copies of the mesh data and swap the y and z axes
    temp_mesh_vertices = mesh_vertices.copy()
    temp_mesh_vertices[:, [1, 2]] = temp_mesh_vertices[:, [2, 1]]  # Swap y and z axes

    # Swap y and z axes for the SDF points
    temp_sdf_points = sdf_points_subset.copy()
    temp_sdf_points[:, [1, 2]] = temp_sdf_points[:, [2, 1]]  # Swap y and z axes

    # Compute bounding boxes with swapped axes
    b_min, b_max = compute_enlarged_bounding_box(temp_mesh_vertices)
    small_b_min, small_b_max = compute_small_bounding_box(temp_mesh_vertices)

    # Initialize Polyscope
    ps.init()

    # Register the mesh
    ps_mesh = ps.register_surface_mesh("Mesh", temp_mesh_vertices, mesh_faces)

    # Register the SDF points
    ps_sdf_points = ps.register_point_cloud("SDF Points", temp_sdf_points, radius=0.0025)
    ps_sdf_points.set_color((1.0, 0.0, 0.0))  # Red for SDF points
    # Add scalar quantity to the SDF points
    ps_sdf_points.add_scalar_quantity("SDF Values", sdf_values_subset, enabled=True)

    # Draw bounding boxes
    draw_bounding_box(b_min, b_max, "Large Bounding Box", color=(0.0, 1.0, 0.0), radius=0.002)
    draw_bounding_box(small_b_min, small_b_max, "Small Bounding Box", color=(0.0, 0.0, 1.0), radius=0.001)

    # Show Polyscope
    ps.show()


def main(DISPLACEMENT_FILE, SDF_FILE, SDF_FILE_VALIDATE, index):
    mesh_points, mesh_connectivity, time_steps, deformations, faces, boundary_vertices_index = load_mesh_and_deformations(
        xdmf_file=BUNNY_FILE, h5_file=DISPLACEMENT_FILE
    )

    boundary_points = mesh_points[boundary_vertices_index]
    boundary_deformations = deformations[:, boundary_vertices_index]

    vertices_tensor = boundary_points + boundary_deformations

    def shape(t_index):
        return faces, vertices_tensor[t_index]

    sdf_points, sdf_values = process_sdf_data(SDF_FILE)
    sdf_points_validate, sdf_values_validate = process_sdf_data(SDF_FILE_VALIDATE)

    sdf_points, sdf_values, sdf_points_validate, sdf_values_validate = resize_to_minimum(
        sdf_points, sdf_values, sdf_points_validate, sdf_values_validate
    )

    print(f"\n")
    print("shape(sdf_points) =", np.shape(sdf_points))
    print("shape(sdf_values) =", np.shape(sdf_values))
    print("shape(sdf_points_validate) =", np.shape(sdf_points_validate))
    print("shape(sdf_values_validate) =", np.shape(sdf_values_validate))

    print("shape(vertices_tensor) =", np.shape(vertices_tensor))
    print("shape(faces) =", np.shape(faces))
    print("\n")

    TIME_INDEX = 80
    show_mesh(*shape(TIME_INDEX))
    animate_deformation(faces, vertices_tensor)
    visualize_mesh_with_points(
        vertices_tensor[TIME_INDEX], faces, sdf_points[TIME_INDEX], sdf_values[TIME_INDEX], NUMBER_OF_POINTS_IN_VISUALISATION
    )

    # Ensure the directory exists
    TRAINING_DIR = "./training_data"
    os.makedirs(TRAINING_DIR, exist_ok=True)
    # File paths
    files = {
        "sdf_values": os.path.join(TRAINING_DIR, f"sdf_values_{index}.pkl"),
        "sdf_points": os.path.join(TRAINING_DIR, f"sdf_points_{index}.pkl"),
        "sdf_points_validate": os.path.join(TRAINING_DIR, f"sdf_points_{index}_validate.pkl"),
        "sdf_values_validate": os.path.join(TRAINING_DIR, f"sdf_values_{index}_validate.pkl"),
        "vertices_tensor": os.path.join(TRAINING_DIR, f"vertices_tensor_{index}.pkl"),
        "faces": os.path.join(TRAINING_DIR, f"faces_{index}.pkl"),
    }

    # Save each object
    for name, path in files.items():
        with open(path, "wb") as file:
            pickle.dump(eval(name), file)  # Use eval to dynamically fetch the variable
            print(f"Saved {name} to {path}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create animation from deformation data.")
    parser.add_argument("--index", type=int, help="Index of the deformation scenario.")
    # Set up argument parsing

    args = parser.parse_args()
    index = args.index

    DISPLACEMENT_FILE = f"./deformed_bunny_files_tunned/displacement_{index}.h5"
    SDF_FILE = f"./calculated_sdf_tunned/sdf_points_{index}.h5"  # Replace with the path to your HDF5 file
    SDF_FILE_VALIDATE = f"./calculated_sdf_tunned/sdf_points_{index}_validate.h5"  # Replace with the path to your HDF5 file

    ret = main(DISPLACEMENT_FILE, SDF_FILE, SDF_FILE_VALIDATE, index)
    exit(ret)
