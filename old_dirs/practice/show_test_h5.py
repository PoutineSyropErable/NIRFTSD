from dolfinx.io.utils import XDMFFile
from mpi4py import MPI
from dolfinx import mesh, plot
import pyvista as pv
import sys

# Handle dynamic file input
if len(sys.argv) != 2:
    print("Usage: python bunny_visualizer.py <path_to_xdmf>")
    sys.exit(1)

mesh_path = sys.argv[1]

print("\n\n------Start of Program------\n\n")

# Load the mesh
with XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as xdmf:
    tetra_mesh = xdmf.read_mesh(name="Grid")
    print("Mesh loaded successfully!")

# Output mesh information
print(f"Number of vertices: {tetra_mesh.geometry.x.shape[0]}")
print(f"Number of tetrahedra: {tetra_mesh.topology.index_map(3).size_local}\n")

# Extract topology, cell types, and geometry for visualization
topology, cell_types, geometry = plot.vtk_mesh(tetra_mesh)

# Create a Pyvista UnstructuredGrid
grid = pv.UnstructuredGrid(topology, cell_types, geometry)

# Visualize the mesh
plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True, color="lightblue", label="Bunny Mesh")
plotter.add_axes()
plotter.add_title("Bunny Mesh Visualization")

# Add cell picking for interaction
def on_click(mesh, event):
    picked_cell = plotter.picked_cell
    if picked_cell is not None:
        face_center = grid.cell_centers().points[picked_cell]
        print(f"Face {picked_cell} Center: {face_center}")
        face_points = grid.extract_cells(picked_cell).points
        print(f"Face {picked_cell} Points: {face_points}")

plotter.enable_cell_picking(callback=on_click, show_message=True)

# Show or save the plot
plotter.show()

