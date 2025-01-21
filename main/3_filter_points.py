#-------------------------Load Points and normal to apply force---------------------------------
import numpy as np
# Load the data back
loaded_data = np.loadtxt("closest_points_and_normals.txt")

# Split the data into `closest_points` and `normals`
loaded_closest_points = loaded_data[:, :3]  # First 3 columns
loaded_normals = loaded_data[:, 3:]         # Last 3 columns

print("\n\n-----Start of programs------\n\n")

# It's called closest points due to generating at Boundary + Delta, and then projecting to boundary. To avoid off by zero error
# That was to calculate sdf and force at that point, but it turns out its not needed
print(f"Shape(normals)        = {np.shape(loaded_normals)}")
print(f"Shape(closest_points) = {np.shape(loaded_closest_points)}\n")

print("Loaded points to apply force:")
print(loaded_closest_points)

print("\nLoaded normals:")
print(loaded_normals)


#-------------------------Load Mesh-----------------------------
from dolfinx.io.utils import XDMFFile
from mpi4py import MPI

def load_file(filename):
    # Load the mesh from the XDMF file
    with XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        print("\n\nMesh loaded successfully!")
    return domain

domain = load_file("bunny.xdmf")
points = domain.geometry.x  # Array of vertex coordinates
print("\n")
print(f"np.shape(points) = {np.shape(points)}")
print(f"points = {points}\n")


#-------------------------GET Filter points numbers--------------------------
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()
z_min, z_max = points[:, 2].min(), points[:, 2].max()

pos_min = np.array([x_min, y_min , z_min])
pos_max = np.array([x_max, y_max , z_max])

center = (pos_min + pos_max)/2 
bbox_size = (pos_max - pos_min)
print(f"center = {center}")
print(f"bbox_size = {bbox_size}")

DELTA_Z = bbox_size[2]
Z_FRACTION = 0.1
FILTER_Z = Z_FRACTION*DELTA_Z
print(f"We'll be removing every force point bellow z={FILTER_Z}")
#----------------------------Actually filter the points-----------------------------
filter_indices = np.where(loaded_closest_points[:, 2] >= FILTER_Z)[0]
print("")
print(f"np.shape(filter_mask) = {np.shape(filter_indices)}")
print(f"filter_mask = \n{filter_indices}\n")

filtered_points = loaded_closest_points[filter_indices]
print(f"np.shape(filtered_points) = {np.shape(filtered_points)}")
print(f"filtered_points = \n{filtered_points}\n")

for point in filtered_points:
    if point[2] < FILTER_Z:
        print("bad filtering",point)


#-------------------------------- Writting the filtered points to a file--------------------------
# Define the output file name
output_file = "filtered_points_of_force_on_boundary.txt"
np.savetxt(output_file, filtered_points, fmt="%.6f", header="x y z", comments="")




