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
import inspect
import pickle

from __SDF_VIS import plot_histograms_with_function


DEBUG_ = False
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


def write_function_debug(var, var_name=None):
    """
    Prints debug information for a variable including its type, shape, and content.

    Args:
        var: The variable to debug.
        var_name (str): Optional. Name of the variable. If not provided, it will be inferred.
    """
    if var_name is None:
        # Infer the variable name if not explicitly provided
        frame = inspect.currentframe().f_back
        var_name = [name for name, value in frame.f_locals.items() if value is var]
        var_name = var_name[0] if var_name else "<unknown>"

    print(f"type({var_name}) = {type(var)}")
    try:
        print(f"np.shape({var_name}) = {np.shape(var)}")
    except Exception as e:
        print(f"Some error about not having a shape: {e}")
    print(f"{var_name} = \n{var}\n")


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


def weight_function(signed_distance: float, weight_exponent: float = 10) -> float:
    """
    Calculate the weight/probability of selecting a point based on its signed distance.

    Args:
        signed_distance (float): The signed distance of the point.
        weight_exponent (float): The exponent used to control the weighting.

    Returns:
        float: Weight/probability for the given signed distance.
    """
    multiplier = 3 if signed_distance < 0 else 0.8  # Higher priority for negative distances
    return multiplier * (1 + abs(signed_distance)) ** (-weight_exponent)


def filter_points(signed_distances: np.ndarray, weight_exponent: float) -> np.ndarray:
    """
    Filter points based on their signed distances.

    Args:
        signed_distances (np.ndarray): Array of signed distances for the points.
        weight_exponent (float): The exponent used to control the weighting.

    Returns:
        np.ndarray: Indices of the points that pass the filtering.
    """
    # Vectorized weight computation
    weights = np.array([weight_function(sd, weight_exponent) for sd in signed_distances])

    # Generate random numbers for comparison
    random_numbers = np.random.rand(len(signed_distances))

    # Return indices of points where random number < weight
    return np.where(random_numbers < weights)[0]


def generate_sdf_points_from_boundary_points(NUM_POINTS, b_min, b_max, n, b):
    """f = a*z^n + b
    a is calculated so int_min^max f dz = 1"""

    z_min, z_max = b_min[2], b_max[2]

    # print(f"z_min, z_max = {z_min}, {z_max}")
    print("    creating a point generator:")
    tuned_z_generator = DistributionFunction(n, b, z_min, z_max, NUM_PRECOMPUTED_CDF)
    print("        tuned_z_generator:")
    print(tuned_z_generator)
    print("    calculating the point list:")
    point_list = generate_random_points(b_min, b_max, NUM_POINTS, tuned_z_generator)
    return point_list


def generate_and_save_sdf(
    deformed_vertices_array: np.ndarray,
    faces: np.ndarray,
    index: int,
    validate: bool = False,
):
    """
    Generate points, calculate signed distances, filter points, and save to .pkl files.

    Args:
        deformed_vertices_array (np.ndarray): Array of deformed vertices for each time step.
        faces (np.ndarray): Array of faces for the mesh.
        index (int): Index for the file naming.
        validate (bool): Whether to save as validation files (appends `_validate` to filenames).
    """
    suffix = "_validate" if validate else ""

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate points within the bounding box
    b_min = np.full(3, np.inf)
    b_max = np.full(3, -np.inf)

    # Compute collective bounding box
    for t_index in range(len(deformed_vertices_array)):
        temp_b_min, temp_b_max = compute_enlarged_bounding_box(deformed_vertices_array[t_index])
        b_min = np.minimum(b_min, temp_b_min)
        b_max = np.maximum(b_max, temp_b_max)

    b_min, b_max = b_min * BOX_RATIO, b_max * BOX_RATIO
    point_list = generate_sdf_points_from_boundary_points(NUM_POINTS, b_min, b_max, n=Z_EXPONENT, b=Z_OFFSET)
    plot_histograms_with_function(point_list, b_min, b_max, Z_EXPONENT, Z_OFFSET)

    filtered_points_list = []
    filtered_sd_list = []

    # Loop over each time step
    for t_index in range(len(deformed_vertices_array)):
        print(f"Processing time step {t_index}/{len(deformed_vertices_array) - 1}...")

        # Compute signed distances
        signed_distances, _, _ = igl.signed_distance(point_list, deformed_vertices_array[t_index], faces)

        # Filter points based on signed distances
        filtered_index = filter_points(signed_distances, weight_exponent=10)
        filtered_points = point_list[filtered_index]
        filtered_signed_distances = signed_distances[filtered_index]

        # Debugging output
        if DEBUG_:
            write_function_debug(filtered_points)
            write_function_debug(filtered_signed_distances)

        filtered_points_list.append(filtered_points)
        filtered_sd_list.append(filtered_signed_distances)

    # Crop all filtered points and signed distances to the minimum length
    min_length = min(len(points) for points in filtered_points_list)
    filtered_points_array = np.array([points[:min_length] for points in filtered_points_list])
    filtered_sd_array = np.array([sd[:min_length] for sd in filtered_sd_list])

    # Save to .pkl files
    points_file = f"{OUTPUT_DIR}/sdf_points_{index}{suffix}.pkl"
    values_file = f"{OUTPUT_DIR}/sdf_values_{index}{suffix}.pkl"

    with open(points_file, "wb") as pf, open(values_file, "wb") as vf:
        pickle.dump(filtered_points_array, pf)
        pickle.dump(filtered_sd_array, vf)

    print(f"Saved filtered points to {points_file}")
    print(f"Saved filtered signed distances to {values_file}")

    return filtered_points_list, filtered_sd_list


def main():
    # Ensure the directory exists
    TRAINING_DIR = "./training_data"
    os.makedirs(TRAINING_DIR, exist_ok=True)

    print(f"\n\n{'-'*10} Start of Program{'-'*10}\n\n")
    INDEX = 730

    # Input files
    DISPLACEMENT_FILE = f"{DISPLACMENT_DIR}/displacement_{INDEX}.h5"
    OUTPUT_FILE = f"{OUTPUT_DIR}/sdf_points_{INDEX}.h5"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    points, connectivity, time_steps, deformations = load_mesh_and_deformations(xdmf_file=BUNNY_FILE, h5_file=DISPLACEMENT_FILE)

    deformed_vertices_list, deformed_faces_list = [], []
    for deformation in deformations:
        deformed_surface_vertices, deformed_surfaces_faces = get_surface_mesh(points + deformation, connectivity)
        deformed_vertices_list.append(deformed_surface_vertices)
        deformed_faces_list.append(deformed_surfaces_faces)

    deformed_vertices_array = np.array(deformed_vertices_list)
    deformed_faces_array = np.array(deformed_faces_list)
    faces = deformed_faces_array[0]
    # animate_deformation(faces, deformed_vertices_array)

    b_min = np.full(3, np.inf)  # Start with very large values for minimum
    b_max = np.full(3, -np.inf)  # Start with very small values for maximum

    for t_index in range(len(time_steps)):
        # Compute the temporary bounding box for the current time step
        temp_b_min, temp_b_max = compute_enlarged_bounding_box(deformed_vertices_array[t_index])

        # Update the collective bounding box
        b_min = np.minimum(b_min, temp_b_min)  # Update with the smallest coordinates
        b_max = np.maximum(b_max, temp_b_max)  # Update with the largest coordinates

    print("\nCollective largest 1.5*bounding box:")
    print(f"b_min: {b_min}")
    print(f"b_max: {b_max}")

    filtered_points_list, filtered_sd_list = generate_and_save_sdf(deformed_vertices_array, faces, INDEX, validate=False)
    generate_and_save_sdf(deformed_vertices_array, faces, INDEX, validate=True)
    # --- end of main------------------------------------------------------------------

    # Save vertices_tensor and faces as pickle files
    vertices_file = f"{OUTPUT_DIR}/vertices_tensor_{INDEX}.pkl"
    faces_file = f"{OUTPUT_DIR}/faces_{INDEX}.pkl"

    with open(vertices_file, "wb") as vf:
        pickle.dump(deformed_vertices_array, vf)

    with open(faces_file, "wb") as ff:
        pickle.dump(faces, ff)

    print(f"Saved vertices tensor to {vertices_file}")
    print(f"Saved faces to {faces_file}")
    return 0


if __name__ == "__main__":
    main()
