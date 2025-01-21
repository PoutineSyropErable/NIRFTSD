import polyscope as ps
import meshio
import os, sys
import numpy as np


# Function to check if an array is empty
def is_empty(array):
    return array is None or array.size == 0


def read_mesh_file(mesh_file):
    """
    Reads a .mesh file and assumes tetrahedral connectivity.

    Args:
        mesh_file (str): Path to the .mesh file.

    Returns:
        tuple: (points, connectivity)
    """
    mesh = meshio.read(mesh_file)
    points = mesh.points
    tets = mesh.cells_dict.get("tetra", None)

    if is_empty(tets):
        raise ValueError("No tetrahedral connectivity found in the .mesh file.")

    print(f"Loaded .mesh file: {mesh_file}")
    print(f"Vertices: {len(points)} | Tetrahedra: {len(tets)}")
    return points, tets


def read_obj_file(obj_file):
    """
    Reads a .obj file and assumes triangular connectivity.

    Args:
        obj_file (str): Path to the .obj file.

    Returns:
        tuple: (points, connectivity)
    """
    mesh = meshio.read(obj_file)
    points = mesh.points
    triangles = mesh.cells_dict.get("triangle", None)

    if is_empty(triangles):
        raise ValueError("No triangular connectivity found in the .obj file.")

    print(f"Loaded .obj file: {obj_file}")
    print(f"Vertices: {len(points)} | Triangles: {len(triangles)}")
    return points, triangles


def display_meshes(mesh_data_list):
    """
    Displays multiple meshes in Polyscope.

    Args:
        mesh_data_list (list): List of tuples (name, points, connectivity).
    """
    ps.init()

    for name, points, connectivity in mesh_data_list:
        ps.register_surface_mesh(name, points, connectivity)

    ps.show()


def main():
    print("\n\n\n----------- Start of Program ---------------------\n")
    os.chdir(sys.path[0])  # Change to the script directory

    # Default file paths
    mesh_file = "Bunny/bunny.mesh"  # Tetrahedral mesh
    obj_file = "Bunny/bunny.obj"    # Triangular mesh

    # Use command-line arguments if provided
    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    if len(sys.argv) > 2:
        obj_file = sys.argv[2]

    # Load meshes
    mesh_points, mesh_connectivity = read_mesh_file(mesh_file)
    obj_points, obj_connectivity = read_obj_file(obj_file)

    # Combine and display both meshes
    mesh_data = [
        ("Bunny - Tetrahedral", mesh_points, mesh_connectivity),
        ("Bunny - Triangular", obj_points, obj_connectivity),
    ]
    display_meshes(mesh_data)


if __name__ == "__main__":
    main()
