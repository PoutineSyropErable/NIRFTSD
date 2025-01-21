import meshio
import numpy as np
from scipy.spatial import KDTree
from dolfin import *


# Function to extract mesh points and connectivity from file
def get_tetra_mesh_data(file_path):
    """
    Extracts points and tetrahedral connectivity from a mesh file.

    Args:
        file_path (str): Path to the input mesh file.

    Returns:
        tuple: (points, connectivity)
    """
    try:
        mesh = meshio.read(file_path)
    except Exception as e:
        raise FileNotFoundError(f"Error loading mesh file: {e}")

    points = mesh.points
    connectivity = mesh.cells_dict.get("tetra", None)

    if connectivity is None:
        raise ValueError("No tetrahedral cells found in the mesh file.")
    return points, connectivity


# Load bunny mesh
mesh_file = "Bunny/bunny.mesh"
points, connectivity = get_tetra_mesh_data(mesh_file)
print(f"Mesh loaded: {len(points)} points, {len(connectivity)} tetrahedra.")

# Convert to FEniCS mesh
mesh = Mesh()
editor = MeshEditor()
editor.open(mesh, "tetrahedron", 3, 3)
editor.init_vertices(len(points))
editor.init_cells(len(connectivity))

for i, point in enumerate(points):
    editor.add_vertex(i, point)

for i, cell in enumerate(connectivity):
    editor.add_cell(i, cell)

editor.close()
print("Mesh successfully converted to FEniCS format.")

# Function space
V = VectorFunctionSpace(mesh, "Lagrange", 1)


# Boundary condition (fix displacement at boundaries)
def boundary(x, on_boundary):
    return on_boundary


bc = DirichletBC(V, Constant((0.0, 0.0, 0.0)), boundary)

# Define force location (nearest vertex to a chosen point)
force_point = np.array([0.0, 0.05, 0.0])  # Example 3D point
kdtree = KDTree(points)
force_vertex_id = kdtree.query(force_point)[1]
force_vertex_coords = points[force_vertex_id]
print(f"Applying force at vertex {force_vertex_id}, location: {force_vertex_coords}")

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant((0.0, -0.1, 0.0))  # Downward force vector
a = inner(sym(grad(u)), sym(grad(v))) * dx  # Use sym instead of symmetric
L = dot(f, v) * ds

# Solve
u_sol = Function(V)
solve(a == L, u_sol, bc)
print("Deformation computed successfully.")

# Save results
file = XDMFFile("bunny_deformation.xdmf")
file.write(u_sol)
print("Results saved to bunny_deformation.xdmf.")

# Output displacement at force point
displacement = u_sol(force_vertex_coords)
print(f"Displacement at force point {force_vertex_coords}: {displacement}")
