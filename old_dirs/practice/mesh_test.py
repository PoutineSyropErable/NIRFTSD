import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import mesh

# Define 3D points and connectivity (example data; replace with your own)
points = np.array([
    [0.0, 0.0, 0.0],  # Point 0
    [1.0, 0.0, 0.0],  # Point 1
    [0.0, 1.0, 0.0],  # Point 2
    [0.0, 0.0, 1.0],  # Point 3
    [1.0, 1.0, 0.0],  # Point 4
    [1.0, 0.0, 1.0],  # Point 5
    [0.0, 1.0, 1.0],  # Point 6
    [1.0, 1.0, 1.0],  # Point 7
], dtype=np.float64)

connectivity = np.array([
    [0, 1, 2, 3],
    [1, 2, 4, 7],
    [1, 3, 5, 7],
    [2, 3, 6, 7]
], dtype=np.int64)



import basix.ufl

# Define the coordinate element
c_el = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(points.shape[1],)))
# Create the serial mesh
tetra_mesh = mesh.create_mesh(MPI.COMM_SELF, connectivity, points, c_el)

# Output basic mesh information
print("\nDOLFINx mesh created successfully!")
print(f"Number of vertices: {tetra_mesh.geometry.x.shape[0]}")
print(f"Number of tetrahedra: {tetra_mesh.topology.index_map(3).size_local}")


