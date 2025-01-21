from dolfinx.io.utils import XDMFFile
from mpi4py import MPI
from dolfinx import mesh
import numpy as np
import pyvista as pv
import igl


num_points = 3000
show_points_generator = False

# For use_circle == False, using points near surface to get normals
generating_offset_length = 0.05 #The distance between the points on bunny, and point in the air
# used to calculate the normal


# For use_circle == True, using points on Bounding Sphere to get normals 
use_circle = False #Use the sphere to generate, or nearby surface
radius_ratio = 1.1
# radius = np.linalg.norm(center-min_pos)/radius_ratio
""" Where center and min_pos are the center and min in x,y,z of the bunny mesh"""




print("\n\n------Start of Program------\n\n")


def get_data_from_mesh(tetra_mesh, save_to_file=False):
    conn = tetra_mesh.topology.connectivity(3,0)
    # print("type conn = ",type(conn))
    # print(help(type(conn)))
    # Extract the connectivity data and offsets
    connectivity_array = conn.array
    offsets = conn.offsets

    # Convert the flat connectivity array into a 2D array
    connectivity_2d = np.array([
        connectivity_array[start:end]
        for start, end in zip(offsets[:-1], offsets[1:])
    ])

    # Print the result, checkthing the data we GOT(obtained) from dolfinx
    got_points = tetra_mesh._geometry.x
    got_connectivity = connectivity_2d

    print(f"shape(points) = {np.shape(got_points)}")
    print(f"shape(connectivity) = {np.shape(got_connectivity)}\n")

    print(f"type(points) = {type(got_points)}")
    print(f"type(connectivity) = {type(got_connectivity)}\n\n")

    print(f"Mesh geometry (Points):\n{got_points}\n\n")
    print(f"Mesh Topology Connectivity (np array):\n{got_connectivity}")


    return got_points, got_connectivity 


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


def compute_sdf_and_normals(points, faces, query_points):
    """
    Computes the signed distance field (SDF) and surface normals at query points.
    Args:
        points (numpy.ndarray): Nx3 array of vertex positions (float).
        faces (numpy.ndarray): Mx3 array of surface triangle indices (int).
        query_points (numpy.ndarray): Qx3 array of points where SDF and normals are computed.
    Returns:
        tuple: SDF values and normals at query points.
    """
    # Ensure data types are correct
    points = points.astype(np.float64)  # Vertex positions
    faces = faces.astype(np.int32)  # Triangle indices
    query_points = query_points.astype(np.float64)  # Query points

    # Print the input arrays for debugging
    print("\n--- Input Data for Signed Distance Calculation ---\n")
    print(f"Query Points Shape: {query_points.shape}")
    print(f"Query Points:\n{query_points}")

    print(f"\nSurface Vertices Shape: {points.shape}")
    print(f"Surface Vertices:\n{points}")

    print(f"\nSurface Faces Shape: {faces.shape}")
    print(f"Surface Faces:\n{faces}")

    # Compute SDF and normals using libigl
    output = igl.signed_distance(query_points, points, faces, return_normals=True)
    # Extract the results
    sdfs = output[0]  # Signed distances
    indices = output[1]  # Indices of closest faces
    closest_points = output[2]  # Closest points on the mesh
    normals = output[3]  #
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    # Print the results in a formatted way
    print("\n--- Signed Distance Output ---\n")
    print(f"Signed Distances (SDF):\n{np.array2string(sdfs, precision=4, separator=', ', threshold=10)}\n")
    print(f"Closest Face Indices:\n{np.array2string(indices, separator=', ', threshold=10)}\n")
    print(f"Closest Points on Mesh:\n{np.array2string(closest_points, precision=4, separator=', ', threshold=10)}\n")
    print(f"Normals :\n{np.array2string(normals, precision=4, separator=', ', threshold=10)}\n")

    # Normalize normals to ensure unit vectors

    return closest_points, normals, sdfs




# Load the mesh from the XDMF file
with XDMFFile(MPI.COMM_WORLD, "bunny.xdmf", "r") as xdmf:
    tetra_mesh = xdmf.read_mesh(name="Grid")
    print("Mesh loaded successfully!")

# Output mesh information
print(f"Number of vertices: {tetra_mesh.geometry.x.shape[0]}")
print(f"Number of tetrahedra: {tetra_mesh.topology.index_map(3).size_local}\n")

points, connectivity = get_data_from_mesh(tetra_mesh) # Just to see we have correctly obtained the code mesh

# Get the min and max for each column
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()
z_min, z_max = points[:, 2].min(), points[:, 2].max()

x_center = np.mean([x_min, x_max])
y_center = np.mean([y_min, y_max])
z_center = np.mean([z_min, z_max])

min_pos = np.array([x_min, y_min, z_min])
center = np.array([x_center,y_center,z_center])

radius = np.linalg.norm(center-min_pos)/radius_ratio


# Print the results
print("\n\n")
print(f"x_min: {x_min}, x_max: {x_max}")
print(f"y_min: {y_min}, y_max: {y_max}")
print(f"z_min: {z_min}, z_max: {z_max}\n")

print(f"x_center: {x_center}")
print(f"y_center: {y_center}")
print(f"z_center: {z_center}\n")



def direction_vector(phi, theta):
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.array([x, y, z])

def get_position(phi,theta):
    return center + radius*direction_vector(phi,theta)

print(f"Radius of the inscribed circle: {center}")


def generate_uniform_angles(num_points):
    """
    Generate uniform angles phi (polar) and theta (azimuthal) for points on a sphere.
    
    Parameters:
    num_points (int): Number of points to generate.
    
    Returns:
    list[tuple]: List of (phi, theta) tuples.
    """
    angles = []
    offset = 2 / num_points
    increment = np.pi * (3 - np.sqrt(5))  # Golden angle in radians

    for i in range(num_points):
        z = 1 - (i + 0.5) * offset  # z ranges from 1 to -1
        theta = (i * increment) % (2 * np.pi)  # Azimuthal angle
        phi = np.arccos(z)  # Polar angle
        angles.append((phi, theta))

    return angles


# Generate positions
print(f"Positons are {center} + {radius}*direction_vector(phi,theta)\n\n")
# Generate uniform angles
angles = generate_uniform_angles(num_points)
positions = np.array([get_position(phi, theta) for phi, theta in angles])

print(f"The positions are: \n {positions}\n")

def expand_points(points, center, radius):
    """
    Expands points outward from a center by a specified radius.

    Parameters:
        points (numpy.ndarray): Array of shape (N, 3) representing the points.
        center (numpy.ndarray): Array of shape (3,) representing the center point.
        radius (float): Scalar radius to scale the outward expansion.

    Returns:
        numpy.ndarray: Transformed points array of shape (N, 3).
    """
    # Calculate the direction vector from the center to each point
    directions = points - center

    # Expand the points by adding the scaled direction vector
    expanded_points = points + directions * radius

    return expanded_points

def sample_random_points(points, n):
    """
    Randomly sample n points from a given array of points.

    Parameters:
        points (numpy.ndarray): Array of shape (N, 3) representing the points.
        n (int): Number of random points to sample.

    Returns:
        numpy.ndarray: Array of shape (n, 3) containing the sampled points.
    """
    if n > points.shape[0]:
        n = points.shape[0]
    
    # Generate n random indices
    random_indices = np.random.choice(points.shape[0], n, replace=False)
    
    # Select points at the random indices
    sampled_points = points[random_indices, :]
    
    return sampled_points

surface_points, surface_faces = extract_surface_mesh(points, connectivity)
surface_points_offsets = expand_points(surface_points, center, generating_offset_length)
if use_circle:
    generating_points = positions
    print("\nUsing the points on the circle\n")
else:
    generating_points = sample_random_points(surface_points_offsets, num_points)
    print("\nUsing the points near the surface\n")
closest_points, normals, sdfs =  compute_sdf_and_normals(surface_points, surface_faces, generating_points)



# unique_cp = np.unique(closest_points, axis=0)
# print(len(closest_points), len(unique_cp))


def create_line_mesh(positions, closest_points):
    """
    Create a PyVista line mesh connecting points in `positions` to `closest_points`.

    Parameters:
    positions (np.ndarray): Array of shape (n, 3) containing start points of lines.
    closest_points (np.ndarray): Array of shape (n, 3) containing end points of lines.

    Returns:
    pv.PolyData: PyVista PolyData object representing the line mesh.
    """
    if len(positions) != len(closest_points):
        raise ValueError("The positions and closest_points arrays must have the same length.")

    # Combine all points for the line mesh
    all_points = np.vstack((positions, closest_points))

    # Create line connectivity
    num_lines = len(positions)
    lines = []
    for i in range(num_lines):
        lines.append([2, i, i + num_lines])  # Format: [2, start_index, end_index]

    # Convert lines to a numpy array
    lines = np.array(lines, dtype=np.int64).flatten()

    # Create PyVista PolyData for the lines
    line_mesh = pv.PolyData()
    line_mesh.points = all_points
    line_mesh.lines = lines

    return line_mesh


#-------------
# Create a PyVista Sphere for visualization
sphere = pv.Sphere(radius=float(radius), center=center)

# Create a PyVista Point Cloud of the generated points
bunny_cloud = pv.PolyData(points)

generating_point_cloud = pv.PolyData(generating_points)

closest_points_cloud = pv.PolyData(closest_points)

line_mesh = create_line_mesh(generating_points, closest_points)
# line_mesh = create_line_mesh(closest_points, closest_points+normals*0.01)

# Visualization with PyVista
plotter = pv.Plotter()
if show_points_generator and use_circle:
    plotter.add_mesh(sphere, color="lightblue", opacity=0.5, show_edges=True, label="Sphere")

if show_points_generator:
    plotter.add_points(generating_point_cloud, color="red", point_size=10, label="Generating P.C.")
    plotter.add_mesh(line_mesh, color="cyan", line_width=2, label="Lines")


plotter.add_points(bunny_cloud, color="blue", point_size=2, label="Bunny Points")
plotter.add_points(closest_points_cloud, color="green", point_size=10, label="Close on Bunny")

plotter.add_axes()
plotter.add_legend()
plotter.show()


#--------------
 # Combine the arrays for saving
data_to_save = np.hstack((closest_points, normals))

# Save to a text file
np.savetxt("closest_points_and_normals.txt", data_to_save, header="x y z nx ny nz", fmt="%.6f")
print("Data saved to 'closest_points_and_normals.txt'")


#----------------
""" Get the mesh volume"""
print("\n\n")

# Calculate volume
# PyVista requires a `cells` array where the first value is the number of nodes per cell
num_cells = connectivity.shape[0]
num_nodes_per_cell = connectivity.shape[1]
cells = np.hstack([np.full((num_cells, 1), num_nodes_per_cell), connectivity]).flatten()

# Cell types: 10 corresponds to tetrahedrons in PyVista
cell_type = np.full(num_cells, 10, dtype=np.uint8)

# Create the PyVista UnstructuredGrid
tetra_grid = pv.UnstructuredGrid(cells, cell_type, points)
# Calculate the volume
volume = tetra_grid.volume
print(f"Volume of tetrahedral mesh: {volume}")


