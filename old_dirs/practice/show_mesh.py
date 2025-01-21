import polyscope as ps
import meshio
import os, sys
import numpy as np


# Function to check if an array is empty
def is_empty(array):
    return array is None or array.size == 0


def load_mesh(file_path: str):

    # Load mesh using meshio
    mesh = meshio.read(file_path)

    # Extract vertices and faces
    points = mesh.points
    
    # Swap y (index 1) and z (index 2)
    points[:, [1, 2]] = points[:, [2, 1]]

    dict_result = mesh.cells_dict
    faces = mesh.cells_dict.get("triangle", None)  # Extract triangle faces
    connectivity = mesh.cells_dict.get("tetra", None)  # Extract triangle faces

    print(f"Types of faces and conectivity: {type(faces)}, {type(connectivity)}\n")
    print(f"Faces = \n{faces}\n")
    print(f"Points = \n{points}\n")
    print(f"Connectivity = \n{connectivity}\n")

    if not is_empty(connectivity) and is_empty(faces):
        edges = connectivity
        print("It's a tetra shape")
    elif is_empty(connectivity) and not is_empty(faces):
        edges = faces
        print("It's a triangle shape\n")
    else:
        AssertionError("No face or edges")

    return points, edges


def compute_cell_center(points, connectivity, cell_id):
    """
    Computes the center of a tetrahedral cell.

    Args:
        points (numpy.ndarray): Nx3 array of vertex positions.
        connectivity (numpy.ndarray): Mx4 array of tetrahedral cell indices.
        cell_id (int): Index of the tetrahedral cell in the connectivity.

    Returns:
        numpy.ndarray: Coordinates of the center of the specified cell.
    """
    # Extract the vertex indices for the given cell
    vertex_indices = connectivity[cell_id]

    # Retrieve the positions of the vertices
    cell_vertices = points[vertex_indices]

    # Compute the center as the average of the vertices
    center = np.mean(cell_vertices, axis=0)
    return center


def show_mesh(points, edges):
    # Start Polyscope
    ps.init()

    # Register the mesh with Polyscope
    ps_mesh = ps.register_surface_mesh("Mesh Viewer", points, edges)

    # Register the vector quantity on the mesh
    ps_mesh.add_vector_quantity("Vectors to Center", points, defined_on="vertices")

    # ------------------- Show the 3D Axis
    origin = np.array([[0, 0.01, 0]])

    vector_length = 10
    # Vector length is useless
    x_axis = vector_length * np.array([[1, 0, 0]])  # Vector in +X direction
    y_axis = vector_length * np.array([[0, 1, 0]])  # Vector in +Y direction
    z_axis = vector_length * np.array([[0, 0, 1]])  # Vector in +Z direction

    # Combine into a single "point cloud" for visualization
    axes_points = np.vstack([origin, origin, origin])  # Origin repeated 3 times
    axes_vectors = np.vstack([x_axis, y_axis, z_axis])  # X, Y, Z vectors

    ps_axis_x = ps.register_point_cloud("X-axis Origin", origin, radius=0.01)
    ps_axis_x.add_vector_quantity("X-axis Vector", x_axis, enabled=True, color=(1, 0, 0))  # Red

    ps_axis_y = ps.register_point_cloud("Y-axis Origin", origin, radius=0.01)
    ps_axis_y.add_vector_quantity("Y-axis Vector", y_axis, enabled=True, color=(0, 1, 0))  # Green

    ps_axis_z = ps.register_point_cloud("Z-axis Origin", origin, radius=0.01)
    ps_axis_z.add_vector_quantity("Z-axis Vector", z_axis, enabled=True, color=(0, 0, 1))  # Blue
    # Show the mesh in the viewer
    ps.show()


# Example usage
def main():

    print("\n\n\n-----------Start of Program---------------------\n")
    # Change directory to script location
    os.chdir(sys.path[0])

    # Default path to the mesh file
    file_path = "Bunny/bunny.mesh"

    # Use command-line argument if provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]

    points, edges = load_mesh(file_path)
    show_mesh(points, edges)


if __name__ == "__main__":
    main()
