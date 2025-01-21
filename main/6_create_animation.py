import h5py
import numpy as np
import pyvista as pv
from dolfinx import mesh
from dolfinx.io.utils import XDMFFile
from mpi4py import MPI
from typing import Tuple
import argparse
import os, sys
from scipy.spatial import distance

# Ensure the working directory is correct
os.chdir(sys.path[0])


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
    connectivity_2d = [
        connectivity_array[start:end] for start, end in zip(offsets[:-1], offsets[1:])
    ]

    return np.array(connectivity_2d, dtype=object)


def get_mesh(filename: str) -> Tuple[mesh.Mesh, np.ndarray, np.ndarray]:
    """
    Extract points and connectivity from the mesh.

    Parameters:
        filename (str): Path to the XDMF file.

    Returns:
        Tuple[mesh.Mesh, np.ndarray, np.ndarray]: The mesh object, points, and connectivity array.
    """
    domain = load_file(filename)
    points = domain.geometry.x  # Array of vertex coordinates
    conn = domain.topology.connectivity(3, 0)
    connectivity = get_array_from_conn(conn).astype(
        np.int64
    )  # Convert to 2D numpy array

    return domain, points, connectivity


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
        time_steps = np.array(
            sorted(f_group.keys(), key=lambda x: float(x)), dtype=float
        )
        displacements = np.array(
            [f_group[time_step][...] for time_step in f_group.keys()]
        )
        print(
            f"Loaded {len(time_steps)} time steps, Displacement tensor shape: {displacements.shape}"
        )

    return time_steps, displacements


def load_mesh_and_deformations(xdmf_file: str, h5_file: str):
    """
    Load mesh points and deformation data.

    Parameters:
        xdmf_file (str): Path to the XDMF file for the mesh.
        h5_file (str): Path to the HDF5 file for deformation data.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The mesh points, connectivity, time steps, and deformation tensor.
    """
    # Load the mesh
    _, points, connectivity = get_mesh(xdmf_file)

    # Load the deformations
    time_steps, deformations = load_deformations(h5_file)

    return points, connectivity, time_steps, deformations


def get_finger_position(index: int) -> np.ndarray:
    """
    Retrieve the finger position based on the given index.

    Parameters:
        index (int): The index for the deformation scenario.

    Returns:
        np.ndarray: The finger position as a NumPy array.
    """
    FILTERED_POINTS_FILE = "./filtered_points_of_force_on_boundary.txt"
    filtered_points = np.loadtxt(FILTERED_POINTS_FILE, skiprows=1)
    finger_position = filtered_points[index]
    return finger_position


import numpy as np


def compute_barycentric_coordinates(point, vertices):
    """
    Compute the barycentric coordinates of a point with respect to a tetrahedron.

    Parameters:
        point (np.ndarray): The target point (shape: (3,)).
        vertices (np.ndarray): The vertices of the tetrahedron (shape: (4, 3)).

    Returns:
        np.ndarray: The barycentric coordinates (shape: (4,)).
    """
    if vertices.shape != (4, 3):
        raise ValueError(f"Expected vertices shape (4, 3), but got {vertices.shape}")

    T = np.hstack([vertices.T, np.ones((4, 1))])  # Add 1s for affine coordinates
    T_inv = np.linalg.inv(T)
    coords = T_inv @ np.append(point, 1)
    return coords


def is_point_in_tetrahedron(bary_coords):
    """
    Check if a point is inside a tetrahedron based on barycentric coordinates.

    Parameters:
        bary_coords (np.ndarray): The barycentric coordinates of the point.

    Returns:
        bool: True if the point is inside the tetrahedron, False otherwise.
    """
    return np.all(bary_coords >= 0) and np.all(bary_coords <= 1)


def get_closest_points(target_point, points, k=10):
    """
    Find the indices of the k closest points to the target point.

    Parameters:
        target_point (np.ndarray): The point to compare against (shape: (3,)).
        points (np.ndarray): The mesh points (shape: (n_points, 3)).
        k (int): Number of closest points to find.

    Returns:
        np.ndarray: Indices of the k closest points.
    """
    distances = np.linalg.norm(points - target_point, axis=1)
    return np.argsort(distances)[:k]


def find_cell_and_barycentric(points, connectivity, target_point):
    """
    Find the cell containing the point or the closest cell and compute barycentric coordinates.

    Parameters:
        points (np.ndarray): The mesh points (shape: (n_points, 3)).
        connectivity (np.ndarray): The cell connectivity (shape: (n_cells, 4)).
        target_point (np.ndarray): The target point (shape: (3,)).

    Returns:
        tuple: The cell index, barycentric coordinates, and fallback status (bool).
    """
    for cell_index, cell in enumerate(connectivity):
        if len(cell) != 4:
            raise ValueError(
                f"Cell at index {cell_index} does not have 4 vertices: {cell}"
            )

        vertices = points[cell]
        if vertices.shape != (4, 3):
            raise ValueError(
                f"Unexpected vertices shape {vertices.shape} for cell {cell_index}"
            )

        bary_coords = compute_barycentric_coordinates(target_point, vertices)
        if is_point_in_tetrahedron(bary_coords):
            return cell_index, bary_coords, False  # Found cell, no fallback

    # Fallback: Use closest points to find the closest cell
    closest_points_indices = get_closest_points(target_point, points)
    candidate_cells = [
        cell_index
        for cell_index, cell in enumerate(connectivity)
        if np.any(np.isin(cell, closest_points_indices))
    ]

    if not candidate_cells:
        raise RuntimeError("No candidate cells found near the finger position.")

    # Pick the first candidate cell for simplicity
    fallback_cell_index = candidate_cells[0]
    vertices = points[connectivity[fallback_cell_index]]
    bary_coords = compute_barycentric_coordinates(target_point, vertices)

    return fallback_cell_index, bary_coords, True


def get_finger_info(finger_position, points, connectivity):
    """
    Get the cell and barycentric coordinates for the finger position.

    Parameters:
        finger_position (np.ndarray): The finger position (shape: (3,)).
        points (np.ndarray): The mesh points (shape: (n_points, 3)).
        connectivity (np.ndarray): The cell connectivity (shape: (n_cells, 4)).

    Returns:
        dict: Information about the finger including cell index, barycentric coordinates,
              and whether the result is from fallback logic.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape (n_points, 3), but got {points.shape}")

    if connectivity.ndim != 2 or connectivity.shape[1] != 4:
        raise ValueError(
            f"Expected connectivity shape (n_cells, 4), but got {connectivity.shape}"
        )

    cell_index, bary_coords, fallback = find_cell_and_barycentric(
        points, connectivity, finger_position
    )
    return {
        "cell_index": cell_index,
        "barycentric_coordinates": bary_coords,
        "fallback": fallback,
    }


def animate_deformation(
    points: np.ndarray,
    connectivity: np.ndarray,
    deformations: np.ndarray,
    finger_position: np.ndarray,
    output_file: str,
    offscreen: bool,
):
    """
    Animate the deformation of the mesh using PyVista.

    Parameters:
        points (np.ndarray): The initial points of the mesh.
        connectivity (np.ndarray): The connectivity of the mesh.
        deformations (np.ndarray): The deformation tensor [time_index][point_index][x, y, z].
        finger_position (np.ndarray): The position of the finger marker.
        output_file (str): Path to save the animation.
        offscreen (bool): Whether to enable offscreen rendering.
    """
    R = 0.003  # Radius of the sphere

    # Create a PyVista UnstructuredGrid
    num_cells = connectivity.shape[0]
    cells = np.hstack(
        [np.full((num_cells, 1), connectivity.shape[1]), connectivity]
    ).flatten()
    cell_types = np.full(
        num_cells, 10, dtype=np.uint8
    )  # 10 corresponds to tetrahedrons in PyVista
    grid = pv.UnstructuredGrid(cells, cell_types, points)

    displacement_norms = np.linalg.norm(deformations[0], axis=1)
    grid.point_data["Displacement"] = displacement_norms

    # Setup PyVista plotter
    plotter = pv.Plotter(off_screen=offscreen)
    plotter.add_axes()
    plotter.add_text("Deformation Animation", font_size=12)

    # Add the mesh
    actor = plotter.add_mesh(grid, show_edges=True, colormap="coolwarm")
    plotter.show_grid()

    # Add the finger marker as a sphere
    finger_marker = pv.Sphere(radius=R, center=finger_position)
    plotter.add_mesh(finger_marker, color="green", label="Finger Position", opacity=1.0)

    # Open movie file for writing
    plotter.open_movie(output_file, framerate=20)

    # Animate through all time steps
    for t_index in range(deformations.shape[0]):
        displacement = deformations[t_index]
        grid.points = points + displacement
        # Update scalar values (displacement norms) for coloring
        displacement_norms = np.linalg.norm(deformations[t_index], axis=1)
        grid.point_data["Displacement"] = displacement_norms
        # Update the actor to reflect new scalar values
        actor.mapper.scalar_range = (
            np.min(displacement_norms),
            np.max(displacement_norms),
        )
        actor.mapper.update()

        # Write frame
        plotter.write_frame()

        # (Optional) Print progress
        # print(f"Rendered time step {t_index + 1}/{deformations.shape[0]}")

    # Close the plotter
    plotter.close()


def main(INDEX: int, OFFSCREEN: bool):
    # Fixed XDMF file
    XDMF_FILE = "bunny.xdmf"

    # Construct HDF5 file path based on index
    H5_FILE = f"./deformed_bunny_files_tunned/displacement_{INDEX}.h5"
    OUTPUT_DIR = f"./Animations_tunned"
    OUTPUT_FILE = f"{OUTPUT_DIR}/deformation_{INDEX}.mp4"

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load mesh and deformations
    points, connectivity, time_steps, deformations = load_mesh_and_deformations(
        XDMF_FILE, H5_FILE
    )

    print(f"Loaded mesh points: {points.shape}")
    print(f"Loaded connectivity: {connectivity.shape}")
    print(f"Loaded {len(time_steps)} deformation time steps.")

    if connectivity.shape[1] != 4:
        raise ValueError(
            f"Expected connectivity to define tetrahedral cells with 4 vertices, "
            f"but found {connectivity.shape[1]} vertices per cell."
        )

    # Get finger position
    print("\n\n")
    finger_position = get_finger_position(INDEX)
    if False:
        finger_info = get_finger_info(finger_position, points, connectivity)

        print(f"Finger position info: {finger_info}")
        if not finger_info["fallback"]:
            print("The finger is inside a cell.")
        else:
            print("The finger is outside the mesh. Using the closest cell as fallback.")
            print("\n\n")

    # Animate deformation
    animate_deformation(
        points, connectivity, deformations, finger_position, OUTPUT_FILE, OFFSCREEN
    )


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Create animation from deformation data."
    )
    parser.add_argument("--index", type=int, help="Index of the deformation scenario.")
    parser.add_argument(
        "index_pos",
        type=int,
        nargs="?",
        help="Index of the deformation scenario (positional).",
    )
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Enable offscreen rendering for the animation.",
    )
    args = parser.parse_args()

    # Determine the index from either --index or positional argument
    if args.index is not None:
        INDEX = args.index
    elif args.index_pos is not None:
        INDEX = args.index_pos
    else:
        parser.error(
            "Index must be provided either as '--index' or as a positional argument."
        )

    main(INDEX, args.offscreen)
