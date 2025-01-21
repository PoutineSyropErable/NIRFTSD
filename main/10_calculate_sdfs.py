import numpy as np
import igl
import h5py
import pyvista as pv
from dolfinx.io.utils import XDMFFile
from dolfinx import mesh
from mpi4py import MPI
from scipy.optimize import root_scalar
from typing import Tuple
import argparse
import os, sys
import time
import matplotlib.pyplot as plt
import polyscope as ps

DEBUG_ = True
DEBUG_TIMER = False
os.chdir(sys.path[0])

BOX_RATIO = 1.5

NUM_POINTS = 1_000_000
NUM_PRECOMPUTED_CDF = 1000  # Dont make this too big
BUNNY_FILE = "bunny.xdmf"
DISPLACMENT_DIR = "./deformed_bunny_files_tunned"
OUTPUT_DIR = "./calculated_sdf_tunned"


Z_EXPONENT = 0.3
Z_OFFSET = 0.01

NUMBER_OF_POINTS_IN_VISUALISATION = 10_000
START_INDEX = 0


def load_file(filename: str) -> mesh.Mesh:
    """
    Load the mesh from an XDMF file.

    Parameters:
        filename (str): Path to the XDMF file.

    Returns:
        mesh.Mesh: The loaded mesh object.
    """
    with XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        domain: mesh.Mesh = xdmf.read_mesh(name="Grid")
        print("Mesh loaded successfully!")
    return domain


def get_array_from_conn(conn) -> np.ndarray:
    """
    Convert mesh topology connectivity to a 2D numpy array.

    Parameters:
        conn: The mesh topology connectivity (dolfinx mesh.topology.connectivity).

    Returns:
        np.ndarray: A 2D numpy array where each row contains the vertex indices for a cell.
    """
    connectivity_array = conn.array
    offsets = conn.offsets

    # Convert the flat connectivity array into a list of arrays
    connectivity_2d = [connectivity_array[start:end] for start, end in zip(offsets[:-1], offsets[1:])]

    return np.array(connectivity_2d, dtype=object)


def get_mesh(filename: str) -> Tuple[mesh.Mesh, np.ndarray, np.ndarray]:
    """
    Extract points and connectivity from the mesh.

    Parameters:
        filename (str): Path to the XDMF file.

    Returns:
        Tuple[mesh.Mesh, np.ndarray, np.ndarray]: The mesh object, points, and connectivity array.
    """
    domain = load_file(filename)
    points = domain.geometry.x  # Array of vertex coordinates
    conn = domain.topology.connectivity(3, 0)
    connectivity = get_array_from_conn(conn).astype(np.int64)  # Convert to 2D numpy array

    return domain, points, connectivity


def load_deformations(h5_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load deformation data from an HDF5 file.

    Parameters:
        h5_file (str): Path to the HDF5 file containing deformation data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: An array of time steps and a 3D tensor of displacements [time_index][point_index][x, y, z].
    """
    with h5py.File(h5_file, "r") as f:
        # Access the 'Function' -> 'f' group
        function_group = f["Function"]
        f_group = function_group["f"]

        # Extract time steps and displacements
        time_steps = np.array(sorted(f_group.keys(), key=lambda x: float(x)), dtype=float)
        displacements = np.array([f_group[time_step][...] for time_step in f_group.keys()])
        print(f"Loaded {len(time_steps)} time steps, Displacement tensor shape: {displacements.shape}")

    return time_steps, displacements


def load_mesh_and_deformations(xdmf_file: str, h5_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load mesh points and deformation data.

    Parameters:
        xdmf_file (str): Path to the XDMF file for the mesh.
        h5_file (str): Path to the HDF5 file for deformation data.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The mesh points, connectivity, time steps, and deformation tensor.
        return points, connectivity, time_steps, deformations
    """
    # Load the mesh
    _, points, connectivity = get_mesh(xdmf_file)

    # Load the deformations
    time_steps, deformations = load_deformations(h5_file)

    return points, connectivity, time_steps, deformations


def load_displacement_data_old(h5_file):
    """Load displacement data from an HDF5 file."""
    """ Needed to load data in the not tunned directory"""
    with h5py.File(h5_file, "r") as f:
        displacements = f["displacements"][:]
        print(f"Loaded displacement data with shape: {displacements.shape}")
    return displacements


def load_displacement_data(h5_file):
    """
    Load deformation data from an HDF5 file.

    Parameters:
        h5_file (str): Path to the HDF5 file containing deformation data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: An array of time steps and a 3D tensor of displacements [time_index][point_index][x, y, z].
        return time_steps, displacements
    """
    return load_deformations(h5_file)


def extract_boundary_info(domain) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the boundary faces and vertex indices from a tetrahedral mesh.

    Args:
        domain (dolfinx.mesh.Mesh): The input tetrahedral mesh.

    Returns:
        faces (np.ndarray): Triangular faces on the boundary (each row contains 3 vertex indices).
        vertex_index (np.ndarray): Indices of the vertices on the boundary.
    return faces, boundary_vertices_index
    """
    # Step 1: Locate boundary facets
    tdim = domain.topology.dim  # Topological dimension (tetrahedra -> 3D)
    fdim = tdim - 1  # Facet dimension (boundary faces -> 2D)

    # Get facets on the boundary
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))

    # Step 2: Get facet-to-vertex connectivity
    facet_to_vertex = domain.topology.connectivity(fdim, 0)
    if facet_to_vertex is None:
        raise ValueError("Facet-to-vertex connectivity not available. Ensure the mesh is initialized correctly.")

    # Map boundary facets to vertex indices
    boundary_faces = [facet_to_vertex.links(facet) for facet in boundary_facets]

    # Step 3: Flatten and extract unique boundary vertex indices
    boundary_vertices_index = np.unique(np.hstack(boundary_faces))

    # Map original vertex indices to continuous indices (0-based for faces)
    vertex_map = {original: i for i, original in enumerate(boundary_vertices_index)}
    faces = np.array([[vertex_map[v] for v in face] for face in boundary_faces], dtype=int)

    return faces, boundary_vertices_index


def get_surface_mesh(points: np.ndarray, connectivity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the surface mesh from the tetrahedral mesh.

    return vertices, faces
    """
    cells = np.hstack([np.full((connectivity.shape[0], 1), 4), connectivity]).flatten()
    cell_types = np.full(connectivity.shape[0], 10, dtype=np.uint8)  # Tetrahedron type
    tetra_mesh = pv.UnstructuredGrid(cells, cell_types, points)

    # Extract surface mesh
    surface_mesh = tetra_mesh.extract_surface()

    # Get vertices and faces
    vertices = surface_mesh.points
    faces = surface_mesh.faces.reshape(-1, 4)[:, 1:]  # Remove face size prefix

    np_vertices = vertices.view(np.ndarray)

    return np_vertices, faces


def compute_small_bounding_box(mesh_points: np.ndarray) -> (np.ndarray, np.ndarray):
    """Compute the smallest bounding box for the vertices."""
    b_min = np.min(mesh_points, axis=0)
    b_max = np.max(mesh_points, axis=0)
    return b_min, b_max


def compute_enlarged_bounding_box(mesh_points: np.ndarray):
    """Compute the bounding box for the vertices."""
    b_min = mesh_points.min(axis=0)
    b_max = mesh_points.max(axis=0)

    center = (b_min + b_max) / 2
    half_lengths = (b_max - b_min) / 2 * BOX_RATIO
    b_min = center - half_lengths
    b_max = center + half_lengths

    return b_min, b_max


def compute_bounding_box_and_volume(points: np.ndarray, connectivity: np.ndarray) -> tuple:
    """
    Compute the volume of the tetrahedral mesh and its bounding box.

    Args:
        points (np.ndarray): An array of shape (n_points, 3) representing the 3D points.
        connectivity (np.ndarray): An array of shape (n_cells, nodes_per_cell) representing the connectivity.

    Returns:
        tuple: A tuple containing the volume of the mesh and the volume of the bounding box.
    """
    # Get the min and max for each column
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    pos_min = np.array([x_min, y_min, z_min])
    pos_max = np.array([x_max, y_max, z_max])

    center = (pos_min + pos_max) / 2
    bbox_size = pos_max - pos_min
    print(f"Center of bounding box: {center}")
    print(f"Bounding box size: {bbox_size}")

    # Calculate the bounding box volume
    bbox_volume = bbox_size[0] * bbox_size[1] * bbox_size[2]
    print(f"Bounding box volume: {bbox_volume}")

    # Calculate volume of the mesh
    # PyVista requires a `cells` array where the first value is the number of nodes per cell
    num_cells = connectivity.shape[0]
    print(f"Number of cells: {num_cells}")
    num_nodes_per_cell = connectivity.shape[1]
    print(f"Number of nodes per cell: {num_nodes_per_cell}")

    cells = np.hstack([np.full((num_cells, 1), num_nodes_per_cell), connectivity]).flatten().astype(np.int32)

    # Check the shape and contents of the cells array
    print(f"Shape of cells array: {cells.shape}")
    print(f"First 20 elements of cells array: {cells[:20]}")

    # Cell types: 10 corresponds to tetrahedrons in PyVista
    cell_type = np.full(num_cells, 10, dtype=np.uint8)

    # Create the PyVista UnstructuredGrid
    tetra_grid = pv.UnstructuredGrid(cells, cell_type, points)

    # Calculate the volume
    mesh_volume = tetra_grid.volume
    print(f"Volume of tetrahedral mesh: {mesh_volume}")

    # Estimate the average cell size
    if num_cells > 0:
        estimate_avg_cell_size = np.cbrt(mesh_volume / num_cells)
        print(f"Estimated average cell size: {estimate_avg_cell_size}")
    else:
        print("No cells found in the mesh. Cannot estimate cell size.")
        estimate_avg_cell_size = None

    return mesh_volume, bbox_volume


class DistributionFunction:
    def __init__(self, n: float, b: float, z_min: float, z_max: float, num_precompute: int = 1000):
        """
        Initialize the distribution function f(z) = a * (z - z_min)^n + b.
        The constant `a` is normalized to make the PDF integrate to 1.
        """
        self.n = n
        self.b = b
        self.z_min = z_min
        self.z_max = z_max
        self.num_precompute = num_precompute
        self.delta_u = 1 / (self.num_precompute - 1)
        self.a = self._calculate_normalization_constant()
        # print(f"a = {self.a}")
        self.validate_range()
        self._precompute_inverse_cdf()

    def validate_range(self):
        # Precompute CDF values at z_min and z_max
        self.cdf_at_z_min = self.cdf(self.z_min)
        self.cdf_at_z_max = self.cdf(self.z_max)

        # Compute cdf_minus_u range
        self.f_a_range = [self.cdf_at_z_min - 1, self.cdf_at_z_min - 0]
        self.f_b_range = [self.cdf_at_z_max - 1, self.cdf_at_z_max - 0]

        # print(f"f_a_range = {self.f_a_range}")
        # print(f"f_b_range = {self.f_b_range}")
        # print("")
        # Validate ranges
        if max(self.f_a_range) * min(self.f_b_range) > 0:
            raise ValueError(
                f"Invalid range: CDF ranges at z_min and z_max lead to f(a) * f(b) > 0.\n"
                f"f(a) range: {self.f_a_range}\n"
                f"f(b) range: {self.f_b_range}"
            )

    def __str__(self):
        return (
            f"\t\t\tz_min = {self.z_min}\n"
            f"\t\t\tz_max = {self.z_max}\n"
            f"\t\t\tn = {self.n}\n"
            f"\t\t\tb = {self.b}\n"
            f"\t\t\ta = {self.a}"
        )

    def _calculate_normalization_constant(self):
        """
        Calculate the normalization constant `a` to make the PDF integrate to 1.
        """
        n, b, z_min, z_max = self.n, self.b, self.z_min, self.z_max
        length = z_max - z_min

        # Check if normalization is possible
        if b * length > 1:
            raise ValueError(f"Normalization not possible: b * (z_max - z_min) = {b * length} > 1")

        # Calculate normalization constant `a`
        integral_zn = (length ** (n + 1)) / (n + 1)
        a = (1 - b * length) / integral_zn

        return a

    def pdf(self, z):
        """Probability density function f(z)."""
        return self.a * (z - self.z_min) ** self.n + self.b

    def cdf(self, z):
        """Cumulative distribution function F(z)."""
        if z < self.z_min:
            return 0
        if z > self.z_max:
            return 1

        n, b, z_min = self.n, self.b, self.z_min
        integral_zn = ((z - z_min) ** (n + 1)) / (n + 1)
        integral_constant = z - z_min
        return self.a * integral_zn + b * integral_constant

    def _precompute_inverse_cdf(self):
        """
        Precompute inverse CDF for evenly spaced values of u in [0, 1].
        """
        u_values = np.linspace(0, 1, self.num_precompute)
        z_values = []

        # Start with an initial guess at the lower bound
        z_calc = self.z_min
        for u in u_values:
            try:
                z_calc = self._find_inverse_cdf(u, z_calc)  # Use previous z as the next initial guess
            except ValueError as e:
                print(f"\033[31mError in precomputing inverse CDF for u={u}: {e}\033[0m")
                z_calc = self.z_min if u < 0.5 else self.z_max  # Fallback to bounds
            z_values.append(z_calc)

        self.precomputed_u = u_values
        self.precomputed_z = np.array(z_values)

    def _find_inverse_cdf(self, u, z_init=None):
        """
        Perform root-finding to compute inverse CDF for a single value of u.

        Args:
            u (float): Target CDF value.
            z_init (float): Initial guess for z. If None, uses the middle of the range.
        """

        # Handle edge cases
        if u <= 0:
            print(f"    z_min case: u = {u}")
            return self.z_min
        if u >= 1:
            print(f"    z_max case: u = {u}")
            return self.z_max

        def cdf_minus_u(z):
            return self.cdf(z) - u

        x0 = z_init if z_init is not None else (self.z_min + self.z_max) / 2
        # Validate bracket endpoints

        # Slightly offset the brackets to avoid boundary issues
        bracket = [self.z_min + 1e-10, self.z_max - 1e-10]
        # Validate bracket endpoints
        f_a = cdf_minus_u(bracket[0])
        f_b = cdf_minus_u(bracket[1])
        if f_a * f_b > 0:
            raise ValueError(
                f"Invalid bracket for u={u}: CDF(z_min)={f_a + u}, CDF(z_max)={f_b + u}. "
                f"f(a)={f_a}, f(b)={f_b}. Brackets must have different signs."
            )
        solution = root_scalar(cdf_minus_u, bracket=bracket, x0=x0, method="brentq")
        return solution.root

    def _refine_inverse_cdf(self, u, z_guess):
        """
        Refine the inverse CDF calculation using a single Newton iteration.
        """
        # Compute the CDF and PDF at z_guess
        F_z = self.cdf(z_guess)
        f_z = self.pdf(z_guess)

        # Newton iteration
        z_new = z_guess - (F_z - u) / f_z
        return z_new

    def inverse_cdf(self, u, previous_guess=None):
        """
        Inverse CDF to generate random samples from the PDF.
        Combines precomputation with optional refinement using root-finding.
        """
        # Locate the nearest interval in precomputed values
        idx = int(u / self.delta_u)
        idx = np.clip(idx, 0, self.num_precompute - 2)

        # Linearly interpolate between precomputed points
        u1, u2 = self.precomputed_u[idx], self.precomputed_u[idx + 1]
        z1, z2 = self.precomputed_z[idx], self.precomputed_z[idx + 1]
        z_guess = z1 + (u - u1) * (z2 - z1) / (u2 - u1)

        # Optionally refine with root-finding, using either z_guess or previous_guess
        initial_guess = previous_guess if previous_guess is not None else z_guess
        return self._refine_inverse_cdf(u, initial_guess)


def generate_random_points(
    b_min: np.ndarray,
    b_max: np.ndarray,
    num_points: int,
    distribution: "DistributionFunction",
):
    """
    Generate random points within the bounding box, with z generated from a custom distribution.

    Args:
        b_min (np.ndarray): Array containing [min_x, min_y, min_z].
        b_max (np.ndarray): Array containing [max_x, max_y, max_z].
        num_points (int): Number of points to generate.
        distribution (DistributionFunction): Custom distribution for z.

    Returns:
        np.ndarray: Array of shape (num_points, 3) with random points.
    """
    # Generate x and y uniformly
    xy_points = np.random.uniform(b_min[:2], b_max[:2], size=(num_points, 2))

    # Generate z using the custom distribution
    uniform_samples = np.random.uniform(0, 1, size=num_points)
    z_points = np.empty(num_points)  # Preallocate array for z-values

    if DEBUG_TIMER:
        start_time = time.time()
    if not DEBUG_:
        # Fast computation for non-debug mode
        z_points = np.array([distribution.inverse_cdf(u) for u in uniform_samples])
    else:
        batch_size = 1000
        num_batches = (num_points + batch_size - 1) // batch_size  # Total number of batches
        print(f"        Starting computation of {num_points} z-points in {num_batches} batches of {batch_size}...")

        for batch_num, i in enumerate(range(0, num_points, batch_size), start=1):
            batch_start_time = time.time()

            # Get the batch of uniform samples
            batch_end = min(i + batch_size, num_points)
            batch_uniform_samples = uniform_samples[i:batch_end]

            # Compute z-points for the batch
            z_points[i:batch_end] = [distribution.inverse_cdf(u) for u in batch_uniform_samples]

            # Log batch timing
            batch_time = time.time() - batch_start_time
            if DEBUG_TIMER:
                print(f"        Processed batch {batch_num}/{num_batches}: {batch_end - i} points in {batch_time:.6f} seconds.")

    if DEBUG_TIMER:
        total_time = time.time() - start_time
        print(f"    Finished generation of {num_points} points in {total_time:.2f} seconds.")

    # Combine x, y, and z
    output = np.hstack([xy_points, z_points.reshape(-1, 1)])
    return output


def generate_random_points_old(b_min: np.ndarray, b_max: np.ndarray, num_points: int):
    """Generate random points uniformly within the bounding box."""
    return np.random.uniform(b_min, b_max, size=(num_points, 3))


def compute_signed_distances(point_list: np.ndarray, mesh_points: np.ndarray, mesh_faces: np.ndarray):
    """Compute signed distances from points to the triangle mesh."""
    signed_distances, nearest_face, nearest_points = igl.signed_distance(point_list, mesh_points, mesh_faces)
    return signed_distances, nearest_face, nearest_points


def weight_function(signed_distance: float, weight_exponent: float = 10) -> float:
    """Takes a signed_distances and return a probability of taking said points"""
    multiplier = 3 if signed_distance < 0 else 0.8
    return multiplier * (1 + abs(signed_distance)) ** (-weight_exponent)


def filter_function(signed_distance: float, weight_exponent: float) -> bool:
    "Returns a bool or not, deciding weither or not to take the point"

    random_number = np.random.rand()
    return random_number < weight_function(signed_distance, weight_exponent)


def filter_points(signed_distances: np.ndarray, weight_exponent: float) -> np.ndarray:
    """Filter points based on their signed distances."""

    filtered_index = np.array([i for i in range(len(signed_distances)) if filter_function(signed_distances[i], weight_exponent)])
    return filtered_index


def generate_sdf_points_from_boundary_points(NUM_POINTS, deformed_boundary_points_t, n, b):
    """f = a*z^n + b
    a is calculated so int_min^max f dz = 1"""

    b_min, b_max = compute_enlarged_bounding_box(deformed_boundary_points_t)
    z_min, z_max = b_min[2], b_max[2]

    # print(f"z_min, z_max = {z_min}, {z_max}")
    print("    creating a point generator:")
    tuned_z_generator = DistributionFunction(n, b, z_min, z_max, NUM_PRECOMPUTED_CDF)
    print("        tuned_z_generator:")
    print(tuned_z_generator)
    print("    calculating the point list:")
    point_list = generate_random_points(b_min, b_max, NUM_POINTS, tuned_z_generator)
    return point_list


def show_mesh(faces: np.ndarray, vertices: np.ndarray):
    """
    Visualize a 3D mesh using PyVista.

    Args:
        vertices (np.ndarray): Array of shape (num_vertices, 3) representing the vertex coordinates.
        faces (np.ndarray): Array of shape (num_faces, 3) representing the triangular face indices.
    """
    if vertices.shape[1] != 3:
        raise ValueError("Vertices array must have shape (num_vertices, 3).")
    if faces.shape[1] != 3:
        raise ValueError("Faces array must have shape (num_faces, 3).")

    # PyVista requires a specific format for faces
    # Each face is prefixed with the number of vertices in the face
    pyvista_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()

    # Create a PyVista mesh
    mesh = pv.PolyData(vertices, pyvista_faces)

    # Create a PyVista plotter
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, color="lightblue")
    plotter.add_axes()
    plotter.add_title("3D Mesh Visualization")

    # Show the plot
    plotter.show()


def animate_deformation(faces: np.ndarray, deformed_boundary_points: np.ndarray):
    """
    Animate the deformation of a surface mesh using PyVista.

    Parameters:
        faces (np.ndarray): Faces of the surface mesh (connectivity array).
        deformed_boundary_points (np.ndarray): Deformed boundary points at each time step
                                               of shape (time_steps, num_points, 3).
    """
    output_file = "deformation_animation.mp4"  # Define the output file path

    # Create PyVista surface mesh for the initial frame
    initial_points = deformed_boundary_points[0]
    num_faces = faces.shape[0]
    cells = np.hstack([np.full((num_faces, 1), faces.shape[1]), faces]).flatten()
    surface_mesh = pv.PolyData(initial_points, cells)

    # Initialize PyVista plotter
    plotter = pv.Plotter(off_screen=False)
    plotter.add_mesh(surface_mesh, show_edges=True, scalars=None, colormap="coolwarm")
    plotter.add_axes()
    plotter.add_text("Deformation Animation", font_size=12)
    plotter.open_movie(output_file, framerate=20)

    # Animate through all time steps
    for t_index in range(deformed_boundary_points.shape[0]):
        # Update the mesh with the current deformation
        surface_mesh.points = deformed_boundary_points[t_index]
        plotter.write_frame()  # Save the frame to the animation

        print(f"Rendered time step {t_index + 1}/{deformed_boundary_points.shape[0]}")

    # Finalize and close the plotter
    plotter.close()
    print(f"Animation saved to {output_file}")


def plot_histograms_with_function(point_list, b_min, b_max, n, b):
    """
    Plot histograms for x, y, and z from the point list and overlay the normalized PDF for z.

    Args:
        point_list (np.ndarray): Array of shape (num_points, 3) with points (x, y, z).
        b_min (np.ndarray): Minimum bounds [min_x, min_y, min_z].
        b_max (np.ndarray): Maximum bounds [max_x, max_y, max_z].
        n (float): Exponent for the distribution function.
        b (float): Constant for the distribution function.
    """
    x, y, z = point_list[:, 0], point_list[:, 1], point_list[:, 2]

    # Create DistributionFunction for z
    z_min, z_max = b_min[2], b_max[2]
    distribution = DistributionFunction(n=n, b=b, z_min=z_min, z_max=z_max)

    # Generate z values for the function plot
    z_values = np.linspace(z_min, z_max, 1000)
    f_values = distribution.pdf(z_values)

    # Plot histograms
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    data = [x, y, z]
    labels = ["x", "y", "z"]
    bounds = [(b_min[0], b_max[0]), (b_min[1], b_max[1]), (z_min, z_max)]

    for i, ax in enumerate(axes):
        # Calculate the histogram for scaling
        hist, bin_edges = np.histogram(data[i], bins=50, range=bounds[i])
        bin_width = bin_edges[1] - bin_edges[0]

        # Plot the histogram
        ax.hist(
            data[i],
            bins=50,
            range=bounds[i],
            alpha=0.7,
            color="blue",
            label=f"Histogram of {labels[i]}",
        )
        ax.set_title(f"{labels[i]} Histogram", fontsize=14)
        ax.set_xlabel(labels[i], fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)

        # Overlay the function for z
        if labels[i] == "z":
            # Scale f_values to match histogram scale
            f_values_scaled = f_values * len(z) * bin_width
            ax.plot(
                z_values,
                f_values_scaled,
                color="red",
                label=r"$f(z) = a(z - z_{min})^n + b$",
                linewidth=2,
            )
            ax.legend(fontsize=10)

        # Add bounds as vertical lines
        ax.axvline(
            bounds[i][0],
            color="green",
            linestyle="--",
            linewidth=1.5,
            label=f"{labels[i]} min",
        )
        ax.axvline(
            bounds[i][1],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"{labels[i]} max",
        )
        ax.legend()

    plt.tight_layout()
    plt.show()


def draw_bounding_box(
    b_min: np.ndarray,
    b_max: np.ndarray,
    name: str,
    color: tuple = (0.0, 1.0, 0.0),
    radius: float = 0.002,
):
    """Draw a bounding box in Polyscope given min and max points."""
    # Create corners of the bounding box
    box_corners = np.array(
        [
            [b_min[0], b_min[1], b_min[2]],
            [b_max[0], b_min[1], b_min[2]],
            [b_max[0], b_max[1], b_min[2]],
            [b_min[0], b_max[1], b_min[2]],
            [b_min[0], b_min[1], b_max[2]],
            [b_max[0], b_min[1], b_max[2]],
            [b_max[0], b_max[1], b_max[2]],
            [b_min[0], b_max[1], b_max[2]],
        ]
    )

    # Define edges for the bounding box
    box_edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],  # Bottom face
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],  # Top face
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],  # Vertical edges
        ]
    )

    # Register the bounding box as a curve network
    ps_bounding_box = ps.register_curve_network(name, box_corners, box_edges)
    ps_bounding_box.set_radius(radius)
    ps_bounding_box.set_color(color)


def visualize_mesh_with_points(mesh_points, mesh_faces, sdf_points):
    """
    Visualize the mesh, bounding boxes, and SDF points in Polyscope.

    Args:
        mesh_points (np.ndarray): Vertices of the mesh.
        mesh_faces (np.ndarray): Faces of the mesh (connectivity).
        sdf_points (np.ndarray): Points where the SDF is computed.
    """
    # Make copies of the data and swap the y and z axes
    temp_mesh_points = mesh_points.copy()
    temp_mesh_points[:, [1, 2]] = temp_mesh_points[:, [2, 1]]  # Swap y and z axes

    temp_sdf_points = sdf_points.copy()
    temp_sdf_points[:, [1, 2]] = temp_sdf_points[:, [2, 1]]  # Swap y and z axes

    # Compute bounding boxes with swapped axes
    b_min, b_max = compute_enlarged_bounding_box(temp_mesh_points)
    small_b_min, small_b_max = compute_small_bounding_box(temp_mesh_points)

    # Initialize Polyscope
    ps.init()

    # Register the mesh
    ps_mesh = ps.register_surface_mesh("Mesh", temp_mesh_points, mesh_faces)

    # Register the SDF points
    ps_sdf_points = ps.register_point_cloud("SDF Points", temp_sdf_points, radius=0.0025)
    ps_sdf_points.set_color((1.0, 0.0, 0.0))  # Red for SDF points

    # Draw bounding boxes
    draw_bounding_box(b_min, b_max, "Large Bounding Box", color=(0.0, 1.0, 0.0), radius=0.002)
    draw_bounding_box(
        small_b_min,
        small_b_max,
        "Small Bounding Box",
        color=(0.0, 0.0, 1.0),
        radius=0.001,
    )

    # Show Polyscope
    ps.show()


def main(INDEX, SDF_ONLY, REPLACE, VALIDATE):
    print(f"\n\n{'-'*10} Start of Program{'-'*10}\n\n")

    # Input files
    DISPLACEMENT_FILE = f"{DISPLACMENT_DIR}/displacement_{INDEX}.h5"
    OUTPUT_FILE = f"{OUTPUT_DIR}/sdf_points_{INDEX}{'_sdf_only' if SDF_ONLY else ''}{'_validate' if VALIDATE else ''}.h5"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.isfile(OUTPUT_FILE):
        if REPLACE:
            os.remove(OUTPUT_FILE)
        else:
            print(f"The file {OUTPUT_FILE} already exist. The sdf and points were already calculated")
            print(f"So, we won't calculate it. Skipping calculation and exiting (5)")
            return 5

    points, connectivity, time_steps, deformations = load_mesh_and_deformations(xdmf_file=BUNNY_FILE, h5_file=DISPLACEMENT_FILE)
    print(f"np.shape(points) = {np.shape(points)}")
    print(f"np.shape(connectivity) = {np.shape(connectivity)}")
    print(f"np.shape(time_steps) = {np.shape(time_steps)}")
    print(f"np.shape(deformations) = {np.shape(deformations)}")

    deformed_vertices_list, deformed_faces_list = [], []
    for deformation in deformations:
        deformed_surface_vertices, deformed_surfaces_faces = get_surface_mesh(points + deformation, connectivity)
        deformed_vertices_list.append(deformed_surface_vertices)
        deformed_faces_list.append(deformed_surfaces_faces)
        print(np.shape(deformed_surface_vertices), np.shape(deformed_surfaces_faces))

    deformed_vertices_array = np.array(deformed_vertices_list)
    deformed_faces_array = np.array(deformed_faces_list)

    print(f"type(deformed_vertices_array) = {type(deformed_vertices_array)}")
    print(f"np.shape(deformed_vertices_array) = {np.shape(deformed_vertices_array)}")

    print(f"type(deformed_faces_array) = {type(deformed_faces_array)}")
    print(f"np.shape(deformed_vertices_list) = {np.shape(deformed_faces_array)}")

    init_vertex = deformed_vertices_array[0, :, :]
    init_face = deformed_faces_array[0, :, :]
    max_fl, max_vl = [], []
    min_fl, min_vl = [], []
    for i in range(len(deformed_faces_array)):
        comp_vertex = deformed_vertices_array[i, :, :]
        comp_face = deformed_faces_array[i, :, :]

        diff_vertex = comp_vertex - init_vertex
        diff_face = comp_face - init_face
        max_fl.append(np.max(diff_face))
        max_vl.append(np.max(diff_vertex))

        min_fl.append(np.min(diff_face))
        min_vl.append(np.min(diff_vertex))

    print("\n\n")
    print(f"min_fl = {min_fl}\n")
    print(f"max_fl = {max_fl}\n")

    print("\n")
    print(f"min_vl = {min_vl}\n")
    print(f"max_vl = {max_vl}\n")

    faces = deformed_faces_array = deformed_faces_list[0]
    deformed_boundary_points = deformed_vertices_array
    # deformed_boundary_points = boundary_points + boundary_deformations

    def shape(t_index):
        return faces, deformed_boundary_points[t_index]

    if DEBUG_:
        show_mesh(*shape(0))
        animate_deformation(faces, deformed_boundary_points)
        pass

    mesh_volume, box_volume = compute_bounding_box_and_volume(points, connectivity)
    mesh_volume1, box_volume1 = compute_bounding_box_and_volume(deformed_boundary_points[0], faces)

    ratio = mesh_volume / box_volume
    inside_points = ratio * NUM_POINTS

    ratio1 = mesh_volume1 / box_volume1
    inside_points1 = ratio1 * NUM_POINTS

    print(f"mesh_volume = {mesh_volume}, box_volume = {box_volume}, ratio = {ratio}, inside_points = {inside_points}")
    print(f"mesh_volume1 = {mesh_volume1}, box_volume1 = {box_volume1}, ratio = {ratio1}, inside_points = {inside_points1}")

    # Prepare to store SDF results
    print("Starting the loop")
    with h5py.File(OUTPUT_FILE, "w") as f:
        # for t_index in range(3):
        for t_index in range(START_INDEX, len(time_steps)):
            print(f"Processing time step {t_index}/{len(time_steps) -1}.")
            # Deform points using displacement
            # Compute signed distances
            # deformed_boundary_points[t_index] is the vertex of the surface mesh (.obj style) of the deformed bunny

            b_min, b_max = compute_enlarged_bounding_box(deformed_boundary_points[t_index])
            print(f"    Bounding box:\n    Min: {b_min}\n    Max: {b_max}\n")

            point_list = generate_sdf_points_from_boundary_points(NUM_POINTS, deformed_boundary_points[t_index], n=Z_EXPONENT, b=Z_OFFSET)
            # point_list = generate_random_points_old(b_min, b_max, NUM_POINTS)
            print(f"\n    type(point_list) = {type(point_list)}")
            print(f"    np.shape(point_list) = {np.shape(point_list)}")
            print(f"    point_list = \n{point_list}\n")

            b_min, b_max = compute_enlarged_bounding_box(deformed_boundary_points[t_index])
            print(f"    Bounding box:\n    Min: {b_min}\n    Max: {b_max}\n")

            if DEBUG_ and t_index == 0:
                plot_histograms_with_function(point_list, b_min, b_max, Z_EXPONENT, Z_OFFSET)
                visualize_mesh_with_points(
                    deformed_boundary_points[t_index],
                    faces,
                    point_list[0:NUMBER_OF_POINTS_IN_VISUALISATION],
                )
            signed_distances, _, _ = compute_signed_distances(point_list, deformed_boundary_points[t_index], faces)
            print(f"obtained signed distances")
            if t_index == 0 and DEBUG_:
                print("\n\n-----------------------ORIGINAL SIGNED DISTANCES----------------------------\n\n")
                print(f"Shape of signed_distances: {np.shape(signed_distances)}")
                print(f"Minimum SDF value: {min(signed_distances):.6f}")
                print(f"Filtered SDF values: {signed_distances}")

                sdf_negative = signed_distances[signed_distances < 0]
                sdf_positive = signed_distances[signed_distances >= 0]

                print("")
                print(f"SDF < 0: \n{sdf_negative}")
                print(f"SDF > 0: \n{sdf_positive}")
                print("")

                print(f"Number SDF < 0: {len(sdf_negative)}, Number SDF > 0: {len(sdf_positive)}")

                plt.figure()
                plt.grid()
                plt.title("Signed distance occurance before filtering")
                plt.xlabel("signed distance value")
                plt.ylabel("frequencies")
                plt.hist(signed_distances)
                plt.show()

            # Filter points based on signed distances
            filtered_index = filter_points(signed_distances, weight_exponent=20)
            filtered_points = point_list[filtered_index]

            filtered_signed_distances = signed_distances[filtered_index]

            if t_index == 0 and DEBUG_:
                print("\n\n-----------------------FILTERED SIGNED DISTANCES----------------------------\n\n")
                print(f"Shape of filtered signed_distances: {np.shape(filtered_signed_distances)}")
                print(f"Minimum SDF value: {min(filtered_signed_distances):.6f}")
                print(f"Filtered SDF values: {filtered_signed_distances}")

                sdf_negative = filtered_signed_distances[filtered_signed_distances < 0]
                sdf_positive = filtered_signed_distances[filtered_signed_distances >= 0]

                print(f"SDF < 0 count: {sdf_negative}")
                print(f"SDF > 0 count: {sdf_positive}")

                print(f"Number SDF < 0: {len(sdf_negative)}, Number SDF > 0: {len(sdf_positive)}")

                plt.figure()
                plt.grid()
                plt.title("Signed distance occurance after filtering")
                plt.xlabel("signed distance value")
                plt.ylabel("frequencies")
                plt.hist(filtered_signed_distances)
                plt.show()

            print(f"np.shape(filtered_points) = {np.shape(filtered_points)}")
            print(f"np.shape(signed_distances) = {np.shape(signed_distances)}")
            print(f"np.shape(filtered_signed_distances) = {np.shape(filtered_signed_distances)}")

            # Combine points and signed distances
            sdf_with_points = np.hstack((filtered_points, filtered_signed_distances[:, None]))
            f.create_dataset(f"time_{t_index}", data=sdf_with_points)

            print(f"Processed time step {t_index}/{len(time_steps) -1}")
            print("----------------------------------------------------")
            print("\n")

        print(f"Saved SDF results to {OUTPUT_FILE}.")
    # --- end of main------------------------------------------------------------------
    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate SDF for deformed meshes.")
    parser.add_argument("--index", type=int, required=True, help="Index of the deformation file.")
    parser.add_argument("--sdf_only", action="store_true", help="Only save the SDF values.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Recalculate the sdf if it already is calculated, and replace the output file",
    )
    parser.add_argument("--validate", action="store_true", help="Generate points for validation")
    args = parser.parse_args()

    INDEX = args.index
    SDF_ONLY = args.sdf_only
    REPLACE = args.replace
    VALIDATE = args.validate
    print(f"\n\n{'-' * 10} Processing INDEX: {INDEX} {'-' * 10}\n\n")
    print(f"SDF only mode: {'Enabled' if SDF_ONLY else 'Disabled'}")
    ret = main(INDEX, SDF_ONLY, REPLACE, VALIDATE)
    exit(ret)
