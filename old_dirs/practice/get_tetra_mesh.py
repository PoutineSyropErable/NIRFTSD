#----------------------get_tetra_mesh.py
from dolfinx.fem import coordinate_element
from dolfinx import plot
import meshio
import numpy as np
from mpi4py import MPI
from dolfinx import mesh
import ufl
import basix.ufl
import pyvista as pv
import polyscope as ps

# Function to extract mesh points and connectivity
def get_tetra_mesh_data(file_path):
    mesh = meshio.read(file_path)
    points = mesh.points
    connectivity = mesh.cells_dict.get("tetra", None)
    if connectivity is None:
        raise ValueError("No tetrahedral cells found in the mesh file.")
    return points, connectivity


# Load mesh
mesh_file = "../Bunny/bunny.mesh"
points, connectivity = get_tetra_mesh_data(mesh_file)
points[:, [1, 2]] = points[:, [2, 1]] # Switch y and z
points[:, 0] = -points[:, 0] # So the bunny face forward

# Get the min and max for each column
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()
z_min, z_max = points[:, 2].min(), points[:, 2].max()


# Print the results
print("____________before_______________")
print(f"x_min: {x_min}, x_max: {x_max}")
print(f"y_min: {y_min}, y_max: {y_max}")
print(f"z_min: {z_min}, z_max: {z_max}")

# Calculate required shifts
# dz = 0.0015
dz = 0.007
x_shift = -x_min
z_shift = -z_min - dz
y_shift = -(y_min + y_max) / 2  # Center y around 0

# Apply transformations
points[:, 0] += x_shift  # Shift x so x_min = 0
points[:, 1] += y_shift  # Center y
points[:, 2] += z_shift  # Shift z so z_min = 0

# Print the results
print("____________after____________")
# Get the min and max for each column
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()
z_min, z_max = points[:, 2].min(), points[:, 2].max()
print(f"x_min: {x_min}, x_max: {x_max}")
print(f"y_min: {y_min}, y_max: {y_max}")
print(f"z_min: {z_min}, z_max: {z_max}")

print("\n\n--------START OF PROGRAM---------\n\n")

print("Mesh name = Bunny")
print("Mesh path = {mesh_file}\n")

print("Mesh = (points, connectivity)\n")

print(f"type(points) = {type(points)}")
print(f"type(connectivity) = {type(connectivity)}\n")

print(f"shape(points) = {np.shape(points)}")
print(f"shape(connectivity) = {np.shape(connectivity)}\n\n")

print(f"points = \n{points}\n")
print(f"connectivity = \n{connectivity}\n")



#---------------------------------
print("\n------Doing dolfinx things-----\n")

points = points.astype(np.float64)
connectivity = connectivity.astype(np.int64)

# Define the coordinate element
coordinate_element = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(points.shape[1],)))


# Create the DOLFINx mesh
tetra_mesh = mesh.create_mesh(MPI.COMM_WORLD, connectivity, points, coordinate_element)


# Output mesh information
print(f"Mesh topology cell types:\n{tetra_mesh.topology.cell_type}\n")
""" See load_bunny for information and testing with getting a numpy array of points 
Of points and connectivity from this, and outputing it with meshio to a file
"""



# Extract mesh data from DOLFINx for Pyvista
topology, cell_types, geometry = plot.vtk_mesh(tetra_mesh)

# Create a Pyvista UnstructuredGrid
grid = pv.UnstructuredGrid(topology, cell_types, geometry)

# Plot the mesh
plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True, color="lightblue", label="Tetrahedral Mesh")
plotter.add_axes()
plotter.show()


# Save the mesh to XDMF
meshio.write_points_cells( "bunny.xdmf",    points, [("tetra", connectivity)] )
print("Mesh saved to bunny.xdmf")





#- Polyscope section:
def show_mesh(points, edges):
    # Start Polyscope
    ps.init()

    # Register the mesh with Polyscope
    ps_mesh = ps.register_surface_mesh("Mesh Viewer", points, edges)

    # Register the vector quantity on the mesh
    ps_mesh.add_vector_quantity("Vectors to Center", points, defined_on="vertices")

    # ------------------- Show the 3D Axis
    origin = np.array([[0, 0.01, 0]])


    # ------------------- Add a Plane at z = 0 -------------------
    dy = y_max - y_min
    dx = x_max - x_min
    dmax = max(dy, dx)
    print("dmax=",dmax)
    # Define the plane's vertices (rectangle on z = 0)
    plane_vertices = np.array([
        [-1, -1, 0],  # Bottom-left
        [1, -1, 0],   # Bottom-right
        [1, 1, 0],    # Top-right
        [-1, 1, 0],   # Top-left
    ]) * dmax  # Scale the plane size as needed

    # Define the plane's triangular faces
    plane_faces = np.array([
        [0, 1, 2],  # First triangle
        [0, 2, 3],  # Second triangle
    ])

    # Register the plane
    ps.register_surface_mesh("z = 0 Plane", plane_vertices, plane_faces, color=(0.8, 0.8, 0.8), transparency=0.5)

    # ------------------- Show the 3D Axis -------------------
    vector_length = 10
    # Vector length is useless
    x_axis = vector_length * np.array([[1, 0, 0]])  # Vector in +X direction
    y_axis = vector_length * np.array([[0, 1, 0]])  # Vector in +Y direction
    z_axis = vector_length * np.array([[0, 0, 1]])  # Vector in +Z direction

    ps_axis_x = ps.register_point_cloud("X-axis Origin", origin, radius=0.01)
    ps_axis_x.add_vector_quantity("X-axis Vector", x_axis, enabled=True, color=(1, 0, 0))  # Red

    ps_axis_y = ps.register_point_cloud("Y-axis Origin", origin, radius=0.01)
    ps_axis_y.add_vector_quantity("Y-axis Vector", y_axis, enabled=True, color=(0, 1, 0))  # Green

    ps_axis_z = ps.register_point_cloud("Z-axis Origin", origin, radius=0.01)
    ps_axis_z.add_vector_quantity("Z-axis Vector", z_axis, enabled=True, color=(0, 0, 1))  # Blue

    



    # Show the mesh in the viewer
    ps.show()

show_mesh(points, connectivity)


# Clip the mesh using gmsh
