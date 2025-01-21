from dolfinx.io.utils import XDMFFile
from mpi4py import MPI
from dolfinx import mesh

print("\n\n------Start of Program------\n\n")


# Load the mesh from the XDMF file
with XDMFFile(MPI.COMM_WORLD, "bunny.xdmf", "r") as xdmf:
    tetra_mesh = xdmf.read_mesh(name="Grid")
    print("Mesh loaded successfully!")

# Output mesh information
print(f"Number of vertices: {tetra_mesh.geometry.x.shape[0]}")
print(f"Number of tetrahedra: {tetra_mesh.topology.index_map(3).size_local}\n")


import pyvista as pv
from dolfinx import plot

# Extract topology, cell types, and geometry for visualization
topology, cell_types, geometry = plot.vtk_mesh(tetra_mesh)

# Create a Pyvista UnstructuredGrid
grid = pv.UnstructuredGrid(topology, cell_types, geometry)

# Visualize the mesh using Pyvista
plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True, color="lightblue", label="Bunny Mesh")
plotter.add_axes()
plotter.add_title("Bunny Mesh Visualization")
plotter.show()




