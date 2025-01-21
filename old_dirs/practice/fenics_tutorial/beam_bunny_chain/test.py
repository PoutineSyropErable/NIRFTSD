import numpy as np
from dolfinx.io import XDMFFile
from dolfinx import fem
from scipy.spatial import Delaunay
from mpi4py import MPI

def load_file(filename):
    # Load the mesh from the XDMF file
    with XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        print("Mesh loaded successfully!")
    return domain


def get_array_from_conn(conn) -> np.ndarray:
    """
    Convert mesh topology connectivity to a 2D numpy array.

    Parameters:
    - conn: The mesh topology connectivity (dolfinx mesh.topology.connectivity).

    Returns:
    - connectivity_2d: A 2D numpy array where each row contains the vertex indices for a cell.
    """
    connectivity_array = conn.array
    offsets = conn.offsets

    # Convert the flat connectivity array into a list of arrays
    connectivity_2d = [
        connectivity_array[start:end]
        for start, end in zip(offsets[:-1], offsets[1:])
    ]

    # Convert to numpy array with dtype=object to handle variable-length rows
    return np.array(connectivity_2d, dtype=object)

def get_function_value_at_point(domain, u, delaunay, query_point):
    """
    Evaluate the function `u` at a given query point.

    Parameters:
    - domain: The mesh domain.
    - u: The function to evaluate.
    - delaunay: Delaunay triangulation of the mesh points.
    - query_point: The point (x, y, z) where the function is to be evaluated.

    Returns:
    - Value of the function at the query point.
    """
    # Find the simplex (cell) containing the point
    simplex = delaunay.find_simplex(query_point)
    if simplex == -1:
        raise ValueError(f"Point {query_point} is outside the mesh.")
    
    # Identify the cell containing the query point
    point = np.atleast_2d(query_point)
    containing_cell = np.array([simplex], dtype=np.int32)
    print(f"containing_cell = {containing_cell}")
    
    # Compute the function value at the query point
    function_value = u.eval(point, containing_cell)
    return function_value

# Load the mesh
domain = load_file("bunny.xdmf")

# Define a function space and function over the domain
V = fem.functionspace(domain, ("Lagrange", 1))
u = fem.Function(V)

# Correctly interpolate a function
def interpolation_function(x):
    return x[0]**2 + x[1] - x[2]

# Apply the interpolation
u.interpolate(interpolation_function)

# Extract points and cells from the mesh
points = domain.geometry.x  # Array of vertex coordinates
topology = domain.topology.connectivity(3, 0)
cells = get_array_from_conn(topology)  # 2D numpy array of cell connectivity

# Create a Delaunay triangulation
delaunay = Delaunay(points)

# Example query point
query_point = np.array([0.1, 0, 0])
try:
    value = get_function_value_at_point(domain, u, delaunay, query_point)
    print(f"Value of the function at {query_point}: {value[0]}")
    print(f"Exact value of the function at {query_point}: {interpolation_function(query_point)}")
except ValueError as e:
    print(e)

