import meshio
import igl
from collections import defaultdict
import numpy as np
import os, sys
import pyvista as pv
import polyscope as ps
from scipy.spatial import KDTree


np.set_printoptions(suppress=True, precision=8)

# master_venv on windows
# conda_venv on linux


def get_tetra_mesh_data(file_path):
    """
    Extracts points and tetrahedral connectivity from a mesh file.

    Args:
        file_path (str): Path to the input mesh file.

    Returns:
        tuple: A tuple (points, connectivity), where:
            - points: numpy.ndarray of shape (N, 3), the mesh vertex coordinates.
            - connectivity: numpy.ndarray of shape (M, 4), the tetrahedral cell indices.

    Raises:
        ValueError: If no tetrahedral cells are found in the mesh.
        FileNotFoundError: If the file does not exist.
    """
    # Load the mesh file
    try:
        mesh = meshio.read(file_path)
    except Exception as e:
        raise FileNotFoundError(f"Error loading mesh file: {e}")

    # Extract points
    points = mesh.points

    # Find tetrahedral cells
    connectivity = mesh.cells_dict.get("tetra", None)

    # Raise error if no tets are found
    if connectivity is None:
        raise ValueError("No tetrahedral cells found in the mesh file.")

    return points, connectivity


def normalize_points(points):
    """
    Normalizes the mesh points to fit inside a unit cube centered at the origin.

    Args:
        points (np.ndarray): Nx3 array of vertex positions.

    Returns:
        np.ndarray: Normalized points centered at origin and scaled to fit in unit cube.
    """
    # Find the bounding box of the points
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    print(f"\n\nDimensions= {max_coords- min_coords}\n\n")

    # Normalize the points to the range [0, 1]
    normalized_points = (points - min_coords) / (max_coords - min_coords)

    print("PolyFEM Normalization:")
    print("Min coords (should be ~0):", np.min(normalized_points, axis=0))
    print("Max coords (should be ~1):", np.max(normalized_points, axis=0))

    return normalized_points


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
    sdf = output[0]  # Signed distances
    indices = output[1]  # Indices of closest faces
    closest_points = output[2]  # Closest points on the mesh
    normals = output[3]  #

    # Print the results in a formatted way
    print("\n--- Signed Distance Output ---\n")
    print(f"Signed Distances (SDF):\n{np.array2string(sdf, precision=4, separator=', ', threshold=10)}\n")
    print(f"Closest Face Indices:\n{np.array2string(indices, separator=', ', threshold=10)}\n")
    print(f"Closest Points on Mesh:\n{np.array2string(closest_points, precision=4, separator=', ', threshold=10)}\n")
    print(f"Normals :\n{np.array2string(normals, precision=4, separator=', ', threshold=10)}\n")

    # Normalize normals to ensure unit vectors
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    return sdf, normals


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


def compute_cells_center(points, connectivity, cell_ids):
    """
    Computes the centers of multiple tetrahedral cells.

    Args:
        points (numpy.ndarray): Nx3 array of vertex positions.
        connectivity (numpy.ndarray): Mx4 array of tetrahedral cell indices.
        cell_ids (list or numpy.ndarray): List of indices of tetrahedral cells.

    Returns:
        numpy.ndarray: Qx3 array of cell centers, where Q is the number of cells.
    """
    # Extract vertex indices for all specified cells
    vertex_indices = connectivity[cell_ids]

    # Retrieve positions of vertices for all cells
    cell_vertices = points[vertex_indices]  # Shape: (Q, 4, 3)

    # Compute the centers as the mean of the vertices along axis 1
    centers = np.mean(cell_vertices, axis=1)  # Shape: (Q, 3)

    return centers


def compute_location(points, connectivity, cell_id, barycentric_coords):
    """
    Computes the Cartesian coordinates of a point given its barycentric coordinates
    in a specified tetrahedral cell.

    Args:
        points (numpy.ndarray): Nx3 array of vertex positions.
        connectivity (numpy.ndarray): Mx4 array of tetrahedral cell indices.
        cell_id (int): Index of the tetrahedral cell in the connectivity.
        barycentric_coords (list or numpy.ndarray): Barycentric coordinates [w0, w1, w2, w3].

    Returns:
        numpy.ndarray: Cartesian coordinates of the point.
    """
    # Extract vertex indices for the cell
    vertex_indices = connectivity[cell_id]

    # Retrieve the positions of the vertices
    cell_vertices = points[vertex_indices]  # Shape: (4, 3)

    # Check if barycentric coordinates sum to approximately 1
    if not np.isclose(np.sum(barycentric_coords), 1.0):
        raise ValueError("Barycentric coordinates must sum to 1.")

    # Compute the Cartesian coordinates using the barycentric formula. T for
    location = np.dot(barycentric_coords, cell_vertices)

    return location


def find_surface_cells(points, connectivity):
    """
    Finds the surface triangular faces and the bordering tetrahedral cells in a tetrahedral mesh.

    Args:
        points (numpy.ndarray): Nx3 array of vertex positions.
        connectivity (numpy.ndarray): Mx4 array of tetrahedral cell indices.

    Returns:
        tuple:
            - surface_faces: List of unique surface triangular faces.
            - surface_cells: Set of indices of tetrahedral cells that share surface faces.
    """
    face_count = defaultdict(int)  # Track the number of times each face appears
    face_to_cell = defaultdict(list)  # Map faces to the tetrahedral cells they belong to

    # Function to sort a face (ensures consistent representation)
    def sorted_face(v1, v2, v3):
        return tuple(sorted([v1, v2, v3]))

    # Iterate over all tetrahedral cells
    for cell_idx, cell in enumerate(connectivity):
        # Extract the 4 faces of the tetrahedron
        faces = [
            sorted_face(cell[0], cell[1], cell[2]),
            sorted_face(cell[0], cell[1], cell[3]),
            sorted_face(cell[0], cell[2], cell[3]),
            sorted_face(cell[1], cell[2], cell[3]),
        ]

        # Update the face count and map faces to cells
        for face in faces:
            face_count[face] += 1
            face_to_cell[face].append(cell_idx)

    # Identify faces that appear only once (surface faces)
    surface_faces = [face for face, count in face_count.items() if count == 1]

    # Identify cells that share these surface faces
    surface_cells = set()
    for face in surface_faces:
        surface_cells.update(face_to_cell[face])

    surface_cells = np.array(list(surface_cells))
    surface_faces = np.array(surface_faces)

    return surface_faces, surface_cells


def compute_mesh_center(points, connectivity):
    """
    Computes the geometric center of a tetrahedral mesh.

    Args:
        points (numpy.ndarray): Nx3 array of vertex positions.
        connectivity (numpy.ndarray): Mx4 array of tetrahedral cell indices.

    Returns:
        numpy.ndarray: 1x3 array representing the coordinates of the mesh center.
    """
    # Retrieve all vertices for all tetrahedral cells
    all_cell_vertices = points[connectivity]  # Shape: (M, 4, 3)

    # Compute the center of each tetrahedral cell
    cell_centers = np.mean(all_cell_vertices, axis=1)  # Shape: (M, 3)

    # Compute the overall mesh center as the mean of all cell centers
    mesh_center = np.mean(cell_centers, axis=0)  # Shape: (3,)

    return mesh_center


def find_closest_cell(points, connectivity, target_point):
    """
    Finds the tetrahedral cell whose center is closest to a given point.

    Args:
        points (numpy.ndarray): Nx3 array of vertex positions.
        connectivity (numpy.ndarray): Mx4 array of tetrahedral cell indices.
        target_point (numpy.ndarray): 1x3 array representing the target point.

    Returns:
        int: Index of the closest tetrahedral cell in the connectivity array.
    """
    # Retrieve all vertices for all tetrahedral cells
    all_cell_vertices = points[connectivity]  # Shape: (M, 4, 3)

    # Compute the center of each tetrahedral cell
    cell_centers = np.mean(all_cell_vertices, axis=1)  # Shape: (M, 3)

    # Compute the Euclidean distance between each cell center and the target point
    distances = np.linalg.norm(cell_centers - target_point, axis=1)

    # Find the index of the cell with the minimum distance
    closest_cell_id = np.argmin(distances)

    return closest_cell_id


class ShapeInfo:
    """
    A class to store all mesh-related shape information.
    """

    def __init__(self, points, connectivity, surface_cells, surface_cells_center, sdfs_surface, normals_surface):
        self.points = points
        self.connectivity = connectivity
        self.surface_cells = surface_cells
        self.surface_cells_center = surface_cells_center
        self.sdfs_surface = sdfs_surface
        self.normals_surface = normals_surface


def rotate_vector_phi_theta(vector, phi, theta):
    """
    Rotates a 3D vector by phi degrees around the Z-axis and then theta degrees
    from the Z-axis (elevation).

    Args:
        vector (numpy.ndarray): Input 3D vector [x, y, z].
        phi (float): Angle in degrees for rotation around the Z-axis (XY-plane).
        theta (float): Angle in degrees for elevation from the Z-axis.

    Returns:
        numpy.ndarray: Rotated 3D vector.
    """
    # Convert degrees to radians
    phi_rad = np.radians(phi)  # Rotation around Z-axis
    theta_rad = np.radians(theta)  # Tilt from Z-axis

    # Rotation matrix around Z-axis (phi)
    rot_z = np.array([[np.cos(phi_rad), -np.sin(phi_rad), 0], [np.sin(phi_rad), np.cos(phi_rad), 0], [0, 0, 1]])

    # Rotation matrix for tilt from Z-axis (theta)
    rot_theta = np.array([[np.cos(theta_rad), 0, np.sin(theta_rad)], [0, 1, 0], [-np.sin(theta_rad), 0, np.cos(theta_rad)]])

    # Combine the rotations: Z-axis rotation first, then tilt
    combined_rotation = rot_theta @ rot_z

    # Apply the rotation to the vector
    rotated_vector = combined_rotation @ vector

    return rotated_vector


def debug_array_info(array, array_name):
    """
    Prints debug information about a 2D array [N, 3] (points or displacements):
        - Shape
        - Rows with non-zero values
        - Number of non-zero rows
        - Maximal and minimal L2 norms (per row) if available

    Args:
        array (np.ndarray): Input 2D array (N points, 3 dimensions).
        array_name (str): Name of the array for debugging purposes.
    """
    # Ensure the input is a numpy array
    array = np.asarray(array)

    # Compute row-wise L2 norms
    norms = np.linalg.norm(array, axis=1)

    # Identify non-zero rows (at least one non-zero value in a row)
    non_zero_mask = np.any(array != 0, axis=1)
    non_zero_values = array[non_zero_mask]
    non_zero_norms = norms[non_zero_mask]

    # Debug outputs
    print("\n\n\n\n\n")
    print(f"\n--- Debug Info for Array: {array_name} ---\n")
    print(f"Shape: {array.shape}")
    print(f"Number of Non-Zero Rows: {non_zero_values.shape[0]}")

    if non_zero_values.shape[0] > 0:
        print("\nFirst 5 Non-Zero Rows:")
        print(non_zero_values[:5])

        print("\nNorms of Non-Zero Rows:")
        print(f"Max Norm: {np.max(non_zero_norms):.8f}")
        print(f"Min Norm: {np.min(non_zero_norms):.8f}")
    else:
        print("\nNo Non-Zero Rows Found. All values are zero.")

    print("\n\n\n\n\n")


def get_deformed_mesh(solver, points):
    """
    Retrieves the deformed mesh from the PolyFEM solver.

    Args:
        solver (pf.Solver): PolyFEM solver instance after solving.
        vertices (np.ndarray): Original vertex positions, shape (N, 3).

    Returns:
        np.ndarray: Deformed vertex positions, shape (N, 3).
    """
    # Retrieve displacement field from the solver
    points, connectivity, displacement = solver.get_sampled_solution()

    debug_array_info(displacement, "displacement")

    # Add displacement to original vertices to get the deformed mesh
    displaced_points = points + displacement

    return displaced_points, connectivity


def setup_and_solve_polyfem(shape_info: ShapeInfo, border_cell_index: int, total_time=1.0, dt=0.01, force_norm=1.0, phi=0.0, theta=0.0):
    """
    Sets up and solves the PolyFEM problem for a given mesh with forces applied to a specified border cell.

    Args:
        shape_info (ShapeInfo): Contains all mesh-related data.
        border_cell_index (int): Index of the border cell to apply the force.
        force_norm (float): Magnitude of the force to apply.

    Returns:
        None
    """

    # Extract the border cell data from ShapeInfo
    some_border_cell_id = int(shape_info.surface_cells[border_cell_index])
    some_border_cell_position = shape_info.surface_cells_center[border_cell_index]
    some_border_cell_sdf = shape_info.sdfs_surface[border_cell_index]
    some_border_cell_normal = shape_info.normals_surface[border_cell_index]

    # Compute the force to apply
    INWARD_FORCE_MULTIPLIER = -1
    force_on_some_border_cell = INWARD_FORCE_MULTIPLIER * force_norm * some_border_cell_normal
    force_on_some_border_cell = rotate_vector_phi_theta(force_on_some_border_cell, phi, theta)
    force_on_some_border_cell = list(force_on_some_border_cell)

    # Print debug information
    print(f"\nTaking the {border_cell_index}th Border Cell")
    print(f"some_border_cell_id = {some_border_cell_id}")
    print(f"some_border_cell_position = {some_border_cell_position}")
    print(f"some_border_cell_sdf = {some_border_cell_sdf:.4f}")
    print(f"some_border_cell_normal = {some_border_cell_normal}")
    print(f"force_on_some_border_cell = {force_on_some_border_cell}")

    # Compute mesh center and closest cell
    mesh_center = compute_mesh_center(shape_info.points, shape_info.connectivity)
    mesh_center_cell_id = find_closest_cell(shape_info.points, shape_info.connectivity, mesh_center)

    # Find closest vertices to fixed_point and force_point using KDTree
    kdtree = KDTree(shape_info.points)
    fixed_vertex_id = kdtree.query(mesh_center)[1]
    force_vertex_id = kdtree.query(some_border_cell_position)[1]

    # Define a marker function to assign sideset IDs dynamically
    def marker_function(v_ids, is_boundary):
        """
        Tag:
        - Sideset 1 for the fixed point.
        - Sideset 2 for the force application point.
        """
        if fixed_vertex_id in v_ids:
            return 1  # Sideset ID 1 for fixing displacement
        elif force_vertex_id in v_ids:
            return 2  # Sideset ID 2 for applying force
        return 0  # Default (no tag

    print("\n\n\n_______COMPUTED FORCES_______________\n\n")
    center = compute_cell_center(shape_info.points, shape_info.connectivity, 100)
    print("__ The center won't move __")
    print(f"mesh_center_cell_id = {mesh_center_cell_id}")
    print(f"Center = {center}")

    print("\n\n\n_______START OF SOLVER_______________\n\n")

    return shape_info.points, shape_info.connectivity, some_border_cell_position, force_on_some_border_cell
    if FALSE:
        # Initialize the PolyFEM solver
        solver = pf.Solver()

        # Set the mesh using vertices and faces
        solver.set_mesh(vertices=shape_info.points, connectivity=shape_info.connectivity, normalize_mesh=True)
        print("Mesh successfully loaded into PolyFEM.")

        # Set solver settings
        settings = pf.Settings(
            discr_order=1, pressure_discr_order=1, pde="LinearElasticity", nl_solver_rhs_steps=1, tend=10, time_steps=1000
        )
        settings.set_pde(pf.PDEs.LinearElasticity)  # Linear Elasticity problem
        settings.set_material_params("E", 210000)
        settings.set_material_params("nu", 0.3)

        # Define boundary conditions
        problem = pf.Problem()

        print("\n\n")

        # Add boundary conditions
        # problem.add_dirichlet_value(id=0, value=[0.0, 0.0, 0.0])  # Fix the center
        # problem.add_neumann_value(id=1, value=force_on_some_border_cell)  # Apply inward force

        problem.set_displacement(id=1, value=[0.0, 0.0, 0.0])
        problem.add_neumann_value(id=2, value=force_on_some_border_cell)  # Apply inward force

        # Assign problem and settings
        settings.set_problem(problem)
        solver.set_settings(settings)

        # Solve the problem
        print("\n\n\nSolving the problem...")
        solver.solve()
        print("\n\nProblem solved.")

        EXPORT = False
        if EXPORT:
            # Export results to visualize (VTU file)
            filename = "./Bunny/deformed_bunny_" + str(border_cell_index) + "_" + str(force_norm) + ".vtu"
            solver.export_vtu(filename)
            print(f"\n\nSolution exported to '{filename}' .")

        displaced_points, new_connectivity = get_deformed_mesh(solver, shape_info.points)
        print("Deformed mesh vertices retrieved successfully.")

        return displaced_points, new_connectivity, some_border_cell_position, force_on_some_border_cell


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


def show_two_meshes(points_1, connectivity_1, points_2, connectivity_2, deformation_point, force_vector):
    """
    Visualize two meshes simultaneously using Polyscope.

    Args:
        points_1 (np.ndarray): Nx3 array of vertex positions for the first mesh.
        connectivity_1 (np.ndarray): Mx3 array of face/edge connectivity for the first mesh.
        points_2 (np.ndarray): Px3 array of vertex positions for the second mesh.
        connectivity_2 (np.ndarray): Qx3 array of face/edge connectivity for the second mesh.
        deformation_point (np.ndarray): 1x3 array representing the point of deformation.
        force_vector (np.ndarray): 1x3 array representing the force direction.
    """
    # Start Polyscope
    ps.init()

    # Register the first mesh
    ps_mesh_1 = ps.register_surface_mesh("Mesh Original", points_1, connectivity_1)
    ps_mesh_1.add_vector_quantity("Vertex Positions - Original", points_1, defined_on="vertices")

    # Register the second mesh
    ps_mesh_2 = ps.register_surface_mesh("Mesh Deformed", points_2, connectivity_2)
    ps_mesh_2.add_vector_quantity("Vertex Positions - Deformed", points_2, defined_on="vertices")

    # Highlight the deformation point
    deformation_point_cloud = np.array([deformation_point])  # Ensure it's a 1x3 array
    ps_force_point = ps.register_point_cloud("Deformation Point", deformation_point_cloud, radius=0.01, color=(1, 0, 0))

    # Add the force vector at the deformation point
    force_vector /= np.linalg.norm(force_vector)
    force_vector *= 1
    force_vector_array = np.array([force_vector])  # 1x3 force vector
    ps_force_point.add_vector_quantity("Force Vector", force_vector_array, enabled=True, color=(1, 0.5, 0), length=1.0)

    # ------------------- Show the 3D Axis
    origin = np.array([[0, 0.01, 0]])

    vector_length = 10
    x_axis = vector_length * np.array([[1, 0, 0]])  # Vector in +X direction
    y_axis = vector_length * np.array([[0, 1, 0]])  # Vector in +Y direction
    z_axis = vector_length * np.array([[0, 0, 1]])  # Vector in +Z direction

    # Register X-axis
    ps_axis_x = ps.register_point_cloud("X-axis Origin", origin, radius=0.01)
    ps_axis_x.add_vector_quantity("X-axis Vector", x_axis, enabled=True, color=(1, 0, 0))  # Red

    # Register Y-axis
    ps_axis_y = ps.register_point_cloud("Y-axis Origin", origin, radius=0.01)
    ps_axis_y.add_vector_quantity("Y-axis Vector", y_axis, enabled=True, color=(0, 1, 0))  # Green

    # Register Z-axis
    ps_axis_z = ps.register_point_cloud("Z-axis Origin", origin, radius=0.01)
    ps_axis_z.add_vector_quantity("Z-axis Vector", z_axis, enabled=True, color=(0, 0, 1))  # Blue

    # Show the meshes
    ps.show()


def main():
    print("\n\n---------------------Start of Program-----------------\n")

    os.chdir(sys.path[0])

    # --- Clean up Polyscope configuration files ---
    config_files = [".polyscope.ini", "imgui.ini"]
    for file in config_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed {file}")

    # Load mesh using meshio
    mesh_path = "Bunny/bunny.mesh"  # Replace with actual input mesh path
    points, connectivity = get_tetra_mesh_data(mesh_path)
    # points = normalize_points(points)
    normalize_points(points)

    print(f"\nPoints.shape() = {np.shape(points)}")
    print(f"Connectivity.shape() = {np.shape(connectivity)}\n")

    print(f"\nPoints = \n{points}\n")
    print(f"\nConnectivity = \n{connectivity}\n")

    # This is to calcualate sdf and sdf gradient.
    # Sfgt gradient = normal
    obj_surface_points, obj_surface_faces = extract_surface_mesh(points, connectivity)
    print(f"\nobj_surface_faces = \n{obj_surface_faces}")
    print(f"\nobj_surface_points = \n{obj_surface_points}\n")

    # Here we won't really use surface_face. Surface cells is just obtained so we can apply force on them.
    surface_face, surface_cells = find_surface_cells(points, connectivity)

    surface_cells_center = compute_cells_center(points, connectivity, surface_cells)
    sdfs_surface, normals_surface = compute_sdf_and_normals(obj_surface_points, obj_surface_faces, surface_cells_center)
    print("\n\n\n_______COMPUTED SDF_______________\n")

    # See 12 lines above : find_surface_cells
    print(f"The number of surface cells is: \n{len(surface_cells)}\n")
    print(f"surface_cells = \n{surface_cells}\n")
    print("\n_______COMPUTED SURFACE CELLS_______________\n")

    # Create ShapeInfo instance
    shape_info = ShapeInfo(points, connectivity, surface_cells, surface_cells_center, sdfs_surface, normals_surface)

    # Call the solver setup function
    deformed_points, deformed_connectivity, deformation_point, force = setup_and_solve_polyfem(
        shape_info, border_cell_index=10, force_norm=4
    )

    # Print all deformed vertices
    print("\n--- Deformed Vertices ---\n")
    print(f"Shape: {deformed_points.shape}")
    print(deformed_points)

    # Print all deformed connectivity
    print("\n--- Deformed Connectivity ---\n")
    print(f"Shape: {deformed_connectivity.shape}")
    print(deformed_connectivity)

    # Print the point of pressure
    print("\ndeformation point:", deformation_point)
    print("Force:", force, "\n\n")

    # show_mesh(deformed_points, deformed_connectivity)

    show_two_meshes(points, connectivity, deformed_points, deformed_connectivity, deformation_point, force)


if __name__ == "__main__":
    main()
