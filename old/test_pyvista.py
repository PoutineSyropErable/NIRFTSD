import numpy as np
import pyvista as pv
import polyscope as ps
import meshio
import os
import sys

def get_tetra_mesh_data(file_path):
    """
    Extracts points and tetrahedral connectivity from a mesh file.

    Args:
        file_path (str): Path to the input mesh file.

    Returns:
        tuple: A tuple (points, connectivity), where:
            - points: numpy.ndarray of shape (N, 3), the mesh vertex coordinates.
            - connectivity: numpy.ndarray of shape (M, 4), the tetrahedral cell indices.

    Raises:
        ValueError: If no tetrahedral cells are found in the mesh.
        FileNotFoundError: If the file does not exist.
    """
    # Load the mesh file
    try:
        mesh = meshio.read(file_path)
    except Exception as e:
        raise FileNotFoundError(f"Error loading mesh file: {e}")

    # Extract points
    points = mesh.points

    # Find tetrahedral cells
    connectivity = mesh.cells_dict.get("tetra", None)

    # Raise error if no tets are found
    if connectivity is None:
        raise ValueError("No tetrahedral cells found in the mesh file.")

    return points, connectivity


def extract_surface_mesh(points, connectivity):
    """
    Extracts the surface triangular mesh from a tetrahedral mesh using PyVista.

    Args:
        points (numpy.ndarray): Nx3 array of vertex positions.
        connectivity (numpy.ndarray): Mx4 array of tetrahedral cell indices.

    Returns:
        tuple: (surface_points, surface_faces)
            - surface_points: Nx3 array of vertex positions.
            - surface_faces: Kx3 array of triangular face indices.
    """
    # Add cell sizes as a prefix to the connectivity
    cells = np.hstack([[4, *tet] for tet in connectivity])  # Prefix 4 indicates tetrahedron
    cell_types = [pv.CellType.TETRA] * len(connectivity)

    # Create an unstructured grid using PyVista
    grid = pv.UnstructuredGrid(cells, cell_types, points)

    # Extract the surface mesh
    surface_mesh = grid.extract_surface()

    # Surface points and faces
    surface_points = surface_mesh.points
    surface_faces = surface_mesh.faces.reshape(-1, 4)[:, 1:]  # Drop the face size prefix

    return surface_points, surface_faces


def visualize_with_polyscope(points, faces, title="Surface Mesh"):
    """
    Visualize a triangular surface mesh using Polyscope.

    Args:
        points (numpy.ndarray): Nx3 array of vertex positions.
        faces (numpy.ndarray): Kx3 array of triangular face indices.
        title (str): Title for the visualization.
    """
    # Initialize Polyscope
    ps.init()

    # Register the surface mesh with Polyscope
    ps.register_surface_mesh(title, points, faces, smooth_shade=False)

    # Show the visualization
    ps.show()


def main():
    # Set working directory to script directory
    os.chdir(sys.path[0])

    # Load the tetrahedral mesh
    mesh_path = "Bunny/bunny.mesh"
    print("Loading tetrahedral mesh...")
    points, connectivity = get_tetra_mesh_data(mesh_path)

    print(f"Loaded mesh with {len(points)} vertices and {len(connectivity)} tetrahedra.")

    # Extract the surface mesh
    print("Extracting surface mesh...")
    surface_points, surface_faces = extract_surface_mesh(points, connectivity)

    print(f"Extracted surface mesh with {len(surface_points)} vertices and {len(surface_faces)} faces.")

    # Visualize the surface mesh using Polyscope
    print("Visualizing surface mesh with Polyscope...")
    visualize_with_polyscope(surface_points, surface_faces, title="Bunny Surface Mesh")


if __name__ == "__main__":
    main()
