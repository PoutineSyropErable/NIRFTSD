import argparse
import pickle
from skimage import measure
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import pyvista as pv

LOAD_DIR = "./training_data"
GRID_DIM = 100

R = 0.003  # Radius of the FINGER
KNN_DIR = "./near_neighbor"


def read_pickle(directory, filename, finger_index, validate=False):
    long_file_name = f"{directory}/{filename}_{finger_index}{'_validate' if validate else ''}.pkl"

    with open(long_file_name, "rb") as file:
        output = pickle.load(file)
        print(f"Loaded {type(output)} from {long_file_name}")

    return output


def load_pickle(path: str):
    with open(path, "rb") as file:
        output = pickle.load(file)
        print(f"Loaded {type(output)} from {path}")

    return output


def save_pickle(path: str, object1):
    with open(path, "wb") as f:
        pickle.dump(object1, f)


def train_nn_models(sdf_points, sdf_values):
    """
    Train a 1-Nearest Neighbor model for each time index.

    Args:
        sdf_points (list of np.ndarray): List of 3D points arrays for each time index.
        sdf_values (list of np.ndarray): List of SDF value arrays for each time index.

    Returns:
        list: List of trained NearestNeighbors models (one per time index).
        list: List of corresponding SDF value arrays (one per time index).
    """
    nn_models = []
    sdf_values_list = []

    for t_index in range(len(sdf_points)):
        points = sdf_points[t_index]  # Shape: (num_points, 3)
        values = sdf_values[t_index]  # Shape: (num_points,)

        # Train 1-NN model for this time index
        nn_model = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn_model.fit(points)

        nn_models.append(nn_model)
        sdf_values_list.append(values)
        print(f"trained {t_index}th nn model")

    print("Trained 1-NN models for all time indices.")
    return nn_models, sdf_values_list


def f_i(nn_models, sdf_values_list, t_index, query_points):
    """
    Query the SDF value for given 3D points at a specific time index.

    Args:
        nn_models (list of NearestNeighbors): Trained 1-NN models (one per time index).
        sdf_values_list (list of np.ndarray): SDF values corresponding to each time index.
        t_index (int): Time index to query.
        query_points (np.ndarray): Array of 3D points to query (num_query_points, 3).

    Returns:
        np.ndarray: SDF values for the query points.
    """
    nn_model = nn_models[t_index]
    sdf_values = sdf_values_list[t_index]

    # Find the nearest neighbor for each query point
    distances, indices = nn_model.kneighbors(query_points)

    # Retrieve the corresponding SDF values
    queried_sdf_values = sdf_values[indices.flatten()]

    return queried_sdf_values


def compute_small_bounding_box(mesh_points: np.ndarray) -> (np.ndarray, np.ndarray):
    """Compute the smallest bounding box for the vertices."""
    b_min = np.min(mesh_points, axis=0)
    b_max = np.max(mesh_points, axis=0)
    return b_min, b_max


def compute_enlarged_bounding_box(mesh_points: np.ndarray, box_ratio: float = 1.5) -> (np.ndarray, np.ndarray):
    """Compute an expanded bounding box for the vertices."""
    b_min, b_max = compute_small_bounding_box(mesh_points)

    center = (b_min + b_max) / 2
    half_lengths = (b_max - b_min) / 2

    # Expand the bounding box by the given ratio
    b_min = center - half_lengths * box_ratio
    b_max = center + half_lengths * box_ratio

    return b_min, b_max


def find_global_bounding_box(vertices_tensor):
    """
    Compute the minimum-sized bounding box (b_min, b_max) for all vertices.

    Args:
        vertices_tensor (list of np.ndarray): List of vertex arrays, where each array is of shape (num_vertices, 3).
        compute_enlarged_bounding_box (function): Function to compute bounding box for a single vertex array.

    Returns:
        tuple: (b_min, b_max), where:
            b_min (np.ndarray): Minimum corner of the global bounding box [x_min, y_min, z_min].
            b_max (np.ndarray): Maximum corner of the global bounding box [x_max, y_max, z_max].
    """
    # Initialize global bounds
    global_b_min = np.array([np.inf, np.inf, np.inf])  # Large initial value for min
    global_b_max = np.array([-np.inf, -np.inf, -np.inf])  # Small initial value for max

    # Iterate over each vertex array
    for verts in vertices_tensor:
        # Compute the bounding box for the current vertices
        b_min_temp, b_max_temp = compute_enlarged_bounding_box(verts)

        # Update the global bounds
        global_b_min = np.minimum(global_b_min, b_min_temp)  # Element-wise min
        global_b_max = np.maximum(global_b_max, b_max_temp)  # Element-wise max

    return global_b_min, global_b_max


def create_3d_points_within_bbox(b_min, b_max, num_points_per_axis):
    """
    Create a 3D grid of points within the specified bounding box.

    Args:
        b_min (array-like): Minimum coordinates of the bounding box [x_min, y_min, z_min].
        b_max (array-like): Maximum coordinates of the bounding box [x_max, y_max, z_max].
        num_points_per_axis (int): Number of points to generate along each axis.


    Returns:
        numpy.ndarray: Array of shape (N, 3) containing the 3D points within the bounding box.
    """
    # Generate linearly spaced points along each axis
    x = np.linspace(b_min[0], b_max[0], num_points_per_axis)
    y = np.linspace(b_min[1], b_max[1], num_points_per_axis)
    z = np.linspace(b_min[2], b_max[2], num_points_per_axis)

    # Create a 3D meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Flatten the meshgrid arrays and stack them into an (N, 3) array
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    return points


def get_trained_models(sdf_points, sdf_values):
    """
    Load pre-trained 1-NN models and SDF values if they exist.
    Otherwise, train the models and save them.

    Args:
        sdf_points (list of np.ndarray): List of 3D points arrays for each time index.
        sdf_values (list of np.ndarray): List of SDF value arrays for each time index.

    Returns:
        tuple: (nn_models, sdf_values_list)
            nn_models: List of trained NearestNeighbors models.
            sdf_values_list: List of corresponding SDF value arrays.
    """
    # Ensure the KNN directory exists
    os.makedirs(KNN_DIR, exist_ok=True)

    # Paths for saving/loading variables
    nn_models_path = os.path.join(KNN_DIR, "nn_models.pkl")
    sdf_values_list_path = os.path.join(KNN_DIR, "sdf_values_list.pkl")

    # Check if pre-trained models and SDF values already exist
    if os.path.exists(nn_models_path) and os.path.exists(sdf_values_list_path):
        # Load pre-trained models and SDF values
        nn_models = load_pickle(nn_models_path)
        sdf_values_list = load_pickle(sdf_values_list_path)
        print("Loaded pre-trained 1-NN models and SDF values.")
    else:
        # Train the 1-NN models
        nn_models, sdf_values_list = train_nn_models(sdf_points, sdf_values)

        # Save the trained models and SDF values
        save_pickle(nn_models_path, nn_models)
        save_pickle(sdf_values_list_path, sdf_values_list)
        print("Trained and saved 1-NN models and SDF values.")

    print("Created models.")
    return nn_models, sdf_values_list


def recreate_meshes(nn_models, sdf_values_list, query_points, b_min, b_max, grid_dim):
    """
    Recreate meshes for all time indices using Marching Cubes, with manual scaling for vertices.

    Args:
        nn_models (list): List of trained 1-NN models for each time index.
        sdf_values_list (list): List of SDF value arrays for each time index.
        query_points (np.ndarray): 3D grid of query points.
        b_min (np.ndarray): Minimum corner of the bounding box [x_min, y_min, z_min].
        b_max (np.ndarray): Maximum corner of the bounding box [x_max, y_max, z_max].
        grid_dim (int): Number of points along each dimension of the grid.

    Returns:
        tuple: (verts_list, faces_list)
            verts_list (list of np.ndarray): List of vertices for each time index.
            faces_list (list of np.ndarray): List of faces for each time index.
    """
    verts_list, faces_list = [], []

    for t_index in range(len(nn_models)):
        # Query SDF values for the current time index
        sdf_values_query = f_i(nn_models, sdf_values_list, t_index, query_points)

        # Reshape SDF values to fit the grid for Marching Cubes
        sdf_grid = sdf_values_query.reshape((grid_dim, grid_dim, grid_dim))

        # Apply Marching Cubes to generate the mesh (without spacing argument)
        verts, faces, normals, values = measure.marching_cubes(sdf_grid, level=0)

        # Manually scale and translate the vertices
        scale = (b_max - b_min) / (grid_dim - 1)
        verts = verts * scale + b_min  # ax + b transformation

        # Debugging: Print bounds for the current mesh
        verts_min = verts.min(axis=0)
        verts_max = verts.max(axis=0)
        print(
            f"Time Index {t_index}: Recreated mesh with {len(verts)} vertices and {len(faces)} faces."
            f" Mesh Bounds: min = {verts_min}, max = {verts_max}"
        )

        # Append the results to the list
        verts_list.append(verts)
        faces_list.append(faces)

    return verts_list, faces_list


def get_recreated_meshes(nn_models, sdf_values_list, query_points, b_min, b_max, grid_dim, output_dir="meshes"):
    """
    Load or recreate meshes for all time indices and save them to disk.

    Args:
        nn_models (list): List of trained 1-NN models for each time index.
        sdf_values_list (list): List of SDF value arrays for each time index.
        query_points (np.ndarray): 3D grid of query points.
        b_min (np.ndarray): Minimum corner of the bounding box [x_min, y_min, z_min].
        b_max (np.ndarray): Maximum corner of the bounding box [x_max, y_max, z_max].
        grid_dim (int): Number of points along each dimension of the grid.
        output_dir (str): Directory to save/load the recreated meshes.

    Returns:
        tuple: (verts_list, faces_list)
            verts_list (list of np.ndarray): List of vertices for each time index.
            faces_list (list of np.ndarray): List of faces for each time index.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    verts_path = os.path.join(output_dir, "verts_list.pkl")
    faces_path = os.path.join(output_dir, "faces_list.pkl")

    if os.path.exists(verts_path) and os.path.exists(faces_path):
        # Load pre-saved meshes
        verts_list = load_pickle(verts_path)
        faces_list = load_pickle(faces_path)
        print("Loaded precomputed meshes from disk.")
    else:
        # Recreate meshes if they do not exist
        verts_list, faces_list = recreate_meshes(nn_models, sdf_values_list, query_points, b_min, b_max, grid_dim)

        # Save the recreated meshes to disk
        save_pickle(verts_path, verts_list)
        save_pickle(faces_path, faces_list)
        print("Recreated and saved meshes to disk.")

    return verts_list, faces_list


def generate_mesh_list(verts_list, faces_list):
    """
    Generate a list of PyVista meshes from vertices and faces.

    Args:
        verts_list (list of np.ndarray): List of vertex arrays (one per time index).
        faces_list (list of np.ndarray): List of face arrays (one per time index).

    Returns:
        list: List of PyVista PolyData meshes.
    """
    mesh_list = []
    for verts, faces in zip(verts_list, faces_list):
        print(f"np.shape(verts) = {np.shape(verts)}, np.shape(faces) = {np.shape(faces)}")
        pyvista_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()
        mesh = pv.PolyData(verts, pyvista_faces)
        mesh_list.append(mesh)
    return mesh_list


def get_mesh_list(verts_list, faces_list, output_dir="meshes"):
    """
    Load or generate the list of PyVista meshes and save them to disk.

    Args:
        verts_list (list of np.ndarray): List of vertex arrays (one per time index).
        faces_list (list of np.ndarray): List of face arrays (one per time index).
        output_dir (str): Directory to save/load the mesh list.

    Returns:
        list: List of PyVista PolyData meshes.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    mesh_list_path = os.path.join(output_dir, "mesh_list.pkl")

    if os.path.exists(mesh_list_path):
        # Load precomputed mesh list
        mesh_list = load_pickle(mesh_list_path)
        print("Loaded precomputed mesh list from disk.")
    else:
        # Generate the mesh list if it does not exist
        mesh_list = generate_mesh_list(verts_list, faces_list)

        # Save the mesh list to disk
        save_pickle(mesh_list_path, mesh_list)
        print("Generated and saved mesh list to disk.")

    return mesh_list


def visualize_mesh_list(mesh_list, finger_position, R, output_file="mesh_animation.mp4", offscreen=False):
    """
    Animate the 3D meshes as an animation and a sphere at the finger position using PyVista.

    Args:
        mesh_list (list of pv.PolyData): List of PyVista meshes.
        finger_position (tuple): Coordinates of the sphere's center (x, y, z).
        R (float): Radius of the sphere.
        output_file (str): Path to save the animation (MP4 format).
        offscreen (bool): Whether to enable offscreen rendering.
    """
    # Initialize PyVista plotter
    plotter = pv.Plotter(off_screen=offscreen)
    plotter.add_axes()
    plotter.show_grid()

    # Add a title
    # plotter.add_text("Mesh Animation", font_size=12)

    # Add the static sphere at the finger position
    finger_marker = pv.Sphere(radius=R, center=finger_position)
    plotter.add_mesh(finger_marker, color="red", label="Finger Position", opacity=0.8)

    # Add the first mesh to initialize
    actor = plotter.add_mesh(mesh_list[0], color="lightblue", show_edges=True)

    # Open the movie file for writing
    plotter.open_movie(output_file, framerate=5)

    # Set the initial camera position, focal point, and view up
    plotter.camera_position = [
        (-0.20319936762932952, 0.23791062551755582, 0.38748501954049164),  # Camera position
        (0.07723370113892855, 0.0004967873193695294, 0.06903945672066888),  # Focal point
        (0.15444222902094137, -0.7220822613237896, 0.6743476890866948),  # View up
    ]

    # Animate through all meshes in the list
    for i, mesh in enumerate(mesh_list):
        # Update the actor with the current mesh's geometry
        actor.GetMapper().SetInputData(mesh)

        # Optional: Add a progress label
        plotter.add_text(f"Frame: {i + 1}/{len(mesh_list)}", name="frame-label", font_size=10)

        # Write the current frame to the movie
        plotter.write_frame()
        position, focal_point, view_up = plotter.camera_position
        print(f"  Camera Position: {position}")
        print(f"  Focal Point: {focal_point}")
        print(f"  View Up: {view_up}")
        print()
        # key = input("Press something to continue to next frame").strip().lower()

    # Close the plotter and save the movie
    plotter.close()
    print(f"Animation saved to {output_file}")


def do_all(sdf_points, sdf_values, vertices_tensor, finger_index):
    nn_models, sdf_values_list = get_trained_models(sdf_points, sdf_values)
    print("created models")

    # Get SDF value at grid points for all time_index.
    b_min, b_max = find_global_bounding_box(vertices_tensor)
    query_points = create_3d_points_within_bbox(b_min, b_max, GRID_DIM)
    print("got query points")
    # Get or recreate meshes
    verts_list, faces_list = get_recreated_meshes(nn_models, sdf_values_list, query_points, b_min, b_max, GRID_DIM)

    # Get or generate the mesh list
    mesh_list = get_mesh_list(verts_list, faces_list)

    # File containing finger_positions (after filtering)
    FINGER_POSITIONS_FILES = "filtered_points_of_force_on_boundary.txt"
    finger_positions = np.loadtxt(FINGER_POSITIONS_FILES, skiprows=1)
    # Swap Y and Z because poylscope uses weird data
    # finger_positions[:, [1, 2]] = finger_positions[:, [2, 1]]
    finger_position = finger_positions[finger_index]

    visualize_mesh_list(mesh_list, finger_position, R)


def main(finger_index):
    end = 101
    vertices_tensor = read_pickle(LOAD_DIR, "vertices_tensor", finger_index)[0:end]
    sdf_points = read_pickle(LOAD_DIR, "sdf_points", finger_index)[0:end]
    sdf_values = read_pickle(LOAD_DIR, "sdf_values", finger_index)[0:end]
    sdf_points_validate = read_pickle(LOAD_DIR, "sdf_points", finger_index, validate=True)[0:end]
    sdf_values_validate = read_pickle(LOAD_DIR, "sdf_values", finger_index, validate=True)[0:end]

    nn_models, sdf_values_list = get_trained_models(sdf_points, sdf_values)
    print("created models")

    # Get SDF value at grid points for all time_index.
    b_min, b_max = find_global_bounding_box(vertices_tensor)
    query_points = create_3d_points_within_bbox(b_min, b_max, GRID_DIM)
    print("got query points")
    # Get or recreate meshes
    verts_list, faces_list = get_recreated_meshes(nn_models, sdf_values_list, query_points, b_min, b_max, GRID_DIM)
    verts = verts_list[0]
    a_min, a_max = compute_small_bounding_box(verts)
    c_min, c_max = compute_small_bounding_box(vertices_tensor[0])
    print(f"Original: c_min = {c_min}, a_max = {c_max}")
    print(f"recreated: a_min = {a_min}, a_max = {a_max}")

    # Get or generate the mesh list
    mesh_list = get_mesh_list(verts_list, faces_list)

    # File containing finger_positions (after filtering)
    FINGER_POSITIONS_FILES = "filtered_points_of_force_on_boundary.txt"
    finger_positions = np.loadtxt(FINGER_POSITIONS_FILES, skiprows=1)
    # Swap Y and Z because poylscope uses weird data
    # finger_positions[:, [1, 2]] = finger_positions[:, [2, 1]]
    finger_position = finger_positions[finger_index]

    visualize_mesh_list(mesh_list, finger_position, R)


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process and train models for a specific finger index.")
    parser.add_argument(
        "--finger_index",
        type=int,
        default=730,
        help="Index of the finger for which data will be processed (default: 730)",
    )
    args = parser.parse_args()

    main(args.finger_index)
