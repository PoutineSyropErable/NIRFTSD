import pickle
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt


from __TRAINING_FILE import (
    MeshEncoder,
    SDFCalculator,
    CustomLRScheduler,
    DummyScheduler,
    TrainingContext,
    LATENT_DIM,
    DEFAULT_FINGER_INDEX,
    NEURAL_WEIGHTS_DIR,
)

GRID_DIM = 100


# Directory containing the pickle files
LOAD_DIR = "./training_data"

# Directory where we save and load the neural weights
FINGER_INDEX = 730


def read_pickle(directory, filename, finger_index, validate=False):
    long_file_name = f"{directory}/{filename}_{finger_index}{'_validate' if validate else ''}.pkl"
    print(long_file_name, "\n")

    with open(long_file_name, "rb") as file:
        output = pickle.load(file)
        print(f"Loaded {type(output)} from {long_file_name}")

    return output


def compute_latent_vector_stats(mesh_encoder, vertices_tensor):
    """
    Compute latent vectors, their relative changes, and orders based on the relative standard deviation.

    Args:
        mesh_encoder (MeshEncoder): The trained mesh encoder model.
        vertices_tensor (torch.Tensor): Tensor of vertices (time_steps, num_vertices, 3).

    Returns:
        tuple: (latent_vectors, order, change_rel, change_rel_order)
            latent_vectors: np.ndarray of shape (latent_dim, time_steps)
            order: Indices of latent vector coordinates ordered by relative change.
            change_rel: Relative change (std / mean) of each latent vector coordinate.
            change_rel_order: Relative change values in descending order.
    """
    latent_vector_list = []

    for i in range(len(vertices_tensor)):
        vertices = vertices_tensor[i].view(1, -1)  # Flatten vertices
        latent_vector = mesh_encoder(vertices)  # Extract latent vector
        latent_vector_np = latent_vector.detach().cpu().numpy().flatten()
        print(f"{i}: latent_vector_np = {latent_vector_np}")
        latent_vector_list.append(latent_vector_np)

    latent_vectors = np.array(latent_vector_list).T  # Transpose for easier computation
    print(f"np.shape(latent_vectors) = {np.shape(latent_vectors)}")
    print(f"latent_vectors = \n{latent_vectors}")
    stdl = np.std(latent_vectors, axis=1)  # Standard deviation of each coordinate
    meanl = np.mean(latent_vectors, axis=1)  # Mean of each coordinate

    print("\n")
    print(f"stdl = {stdl}")
    print(f"np.shape(stdl) = {np.shape(stdl)}")

    change_rel = stdl / np.abs(meanl)  # Compute relative change (std / mean)
    change_rel[np.isnan(change_rel)] = 0  # Handle division by zero if mean is zero

    order = np.argsort(change_rel)[::-1]  # Descending order
    change_rel_order = change_rel[order]

    print("\n")
    return latent_vectors, order, change_rel, change_rel_order


def plot_latent_vector_stats(latent_vectors, order, change_rel, change_rel_order, epoch_index):
    """
    Visualize the latent vectors and their relative changes.

    Args:
        latent_vectors (np.ndarray): Latent vectors of shape (latent_dim, time_steps).
        order (np.ndarray): Indices of latent vector coordinates ordered by relative change.
        change_rel (np.ndarray): Relative change of each latent vector coordinate.
        change_rel_order (np.ndarray): Relative change values in descending order.
        epoch_index (int): Epoch index for labeling the plots.
    """
    # Plot relative change (unordered)
    plt.figure()
    plt.title(f"Relative Change (std / mean) of Latent Vector Coordinates [Unordered] (Epoch {epoch_index})", fontsize=16)
    plt.xlabel("Latent Vector Coordinate", fontsize=12)
    plt.ylabel("Relative Change", fontsize=12)
    plt.plot(change_rel, marker="o")
    plt.grid(True)
    plt.show()

    # Plot relative change (ordered)
    plt.figure()
    plt.title(f"Relative Change (std / mean) of Latent Vector Coordinates [Ordered] (Epoch {epoch_index})", fontsize=16)
    plt.xlabel("Latent Vector Coordinate (sorted by relative change)", fontsize=12)
    plt.ylabel("Relative Change", fontsize=12)
    plt.plot(change_rel_order, marker="o")
    plt.grid(True)
    plt.show()

    # Plot trajectory of top 5 latent vector coordinates with highest relative change
    plt.figure()
    plt.title(f"Top 5 Latent Vector Coordinates with Highest Relative Change (Epoch {epoch_index})", fontsize=16)
    plt.xlabel("Time index", fontsize=12)
    plt.ylabel("Latent Value", fontsize=12)
    for i in range(5):
        plt.plot(latent_vectors[order[i]], label=f"Coord {order[i]}, Rel_Change: {change_rel_order[i]:.4f}")
    # plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()


def main(finger_index=DEFAULT_FINGER_INDEX, epoch_index=100):
    """
    Track the output of the MeshEncoder across 13 epochs and generate statistics and plots.

    Args:
        finger_index (int): Index of the finger for which data will be processed.
    """
    vertices_tensor_np = read_pickle(LOAD_DIR, "vertices_tensor", finger_index)[:-1]
    vertices_tensor = torch.tensor(vertices_tensor_np, dtype=torch.float32)  # Convert to PyTorch tensor

    input_dim = vertices_tensor.shape[1] * vertices_tensor.shape[2]  # num_vertices * 3
    mesh_encoder = MeshEncoder(input_dim=input_dim, latent_dim=LATENT_DIM)
    sdf_calculator = SDFCalculator(latent_dim=LATENT_DIM)
    training_context = TrainingContext(mesh_encoder, sdf_calculator, finger_index, len(vertices_tensor), 0.1, 0.01)

    mean_rel_change = []
    max_rel_change = []

    rel_changes_all_epochs = []
    # Iterate through 13 epochs
    # Add the sequence 12, 22, ..., 592

    print(f"\nProcessing Epoch {epoch_index}...\n")

    # Load MeshEncoder weights for the current epoch
    training_context.load_model_weights(epoch_index, time_index=0)

    # Compute latent vector statistics
    latent_vectors_tp, order, change_rel, change_rel_ordered = compute_latent_vector_stats(training_context.mesh_encoder, vertices_tensor)
    mean_rel_change.append(np.mean(change_rel))
    max_rel_change.append(change_rel_ordered[0])

    rel_changes_all_epochs.append(change_rel_ordered)

    # Plot latent vector statistics for the current epoch
    plot_latent_vector_stats(latent_vectors_tp, order, change_rel, change_rel_ordered, epoch_index)

    latent_vectors = latent_vectors_tp.T
    cos_sim = np.zeros(len(latent_vectors))
    latent1 = latent_vectors[0]
    print(np.shape(latent1))
    for i in range(len(latent_vectors)):
        latent2 = latent_vectors[i]
        dot_product = np.dot(latent1, latent2)
        norm_latent1 = np.linalg.norm(latent1)
        norm_latent2 = np.linalg.norm(latent2)
        cos_sim[i] = dot_product / (norm_latent1 * norm_latent2)

    plt.figure()
    plt.title("Cosine Similarity over time")
    plt.grid()
    plt.ylabel(r"$\frac{q[i] \cdot q[0]}{\|q[i]\| \cdot \|q[0]\|}$")
    plt.plot(cos_sim, ".b", ms=5)
    plt.plot(cos_sim, "-", color="orange", linewidth=1)
    plt.show()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="use a trained model to recreate shapes")
    # Arguments for epoch and time indices
    parser.add_argument("--finger_index", type=int, default=730, help="Specify the finger index where the force was applied")
    parser.add_argument(
        "--epoch_index",
        type=int,
        help="Specify the epoch index to continue processing from.",
    )
    args = parser.parse_args()

    ret = main(args.finger_index, args.epoch_index)
    exit(ret)
