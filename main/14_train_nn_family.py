import pickle
import argparse
import os
import signal
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import copy
from typing import Optional

from enum import Enum

# Directory containing the pickle files
LOAD_DIR = "./training_data"

# Directory where we save and load the neural weights
NEURAL_WEIGHTS_DIR = "./neural_weights"
LATENT_DIM = 64
SCHEDULER_SWITCH_EPOCH = 1001
DEFAULT_FINGER_INDEX = 730

# Global values for signals
stop_time_signal = False
stop_epoch_signal = False


class SaveMode(Enum):
    NowTime = 1
    NowEpoch = 2
    NextTimeItteration = 3
    NextEpochItteration = 4
    End = 5


# Signal handlers
def handle_stop_time_signal(signum, frame):
    global stop_time_signal
    stop_time_signal = True
    print("Received stop time signal. Will stop after the current time iteration.")


def handle_stop_epoch_signal(signum, frame):
    global stop_epoch_signal
    stop_epoch_signal = True
    print("Received stop epoch signal. Will stop after the current epoch iteration.")


def handle_save_epoch_signal(signum, frame):
    """
    Handle termination signals (SIGTERM, SIGINT).
    Save weights at the current epoch and time index before exiting.
    """
    global training_context
    print("Received save previous epoch signal")
    training_context.save_model_weights(SaveMode.NowEpoch)


def handle_save_time_signal(signum, frame):
    """
    Handle termination signals (SIGTERM, SIGINT).
    Save weights at the current epoch and time index before exiting.
    """
    print("Received save previous time signal")
    global training_context
    training_context.save_model_weights(SaveMode.NowTime)


def handle_termination_time(signum, frame):
    """
    Handle termination signals (SIGTERM, SIGINT).
    Save weights at the current epoch and time index before exiting.
    """
    print("\nhandling time termination, save.NowTime, then exit\n")
    global training_context
    training_context.save_model_weights(SaveMode.NowTime)
    exit(1)


def handle_termination_epoch(signum, frame):
    """
    Handle termination signals (SIGTERM, SIGINT).
    Save weights at the current epoch and time index before exiting.
    """
    print("\nhandling Epoch termination, save.NowEpoch, then exit\n")
    global training_context
    training_context.save_model_weights(SaveMode.NowEpoch)
    exit(2)


def get_latest_saved_indices():
    """
    Fetch the latest saved epoch and time index from the saved weights directory.
    Returns:
        (int, int): Latest epoch index and time index.
    """
    weights_files = [f for f in os.listdir(NEURAL_WEIGHTS_DIR) if f.endswith(".pth")]
    if not weights_files:
        return 0, 0  # No weights saved yet
    epochs_times = [tuple(map(int, f.split("_")[2:5:2])) for f in weights_files]  # Extract epoch and time indices
    return max(epochs_times)  # Return the latest epoch and time index


def get_latest_saved_time_for_epoch(epoch):
    """
    Fetch the latest saved time index for a specific epoch.
    Args:
        epoch (int): The epoch for which to find the latest time index.
    Returns:
        int: Latest time index for the given epoch.
    """
    weights_files = [f for f in os.listdir(NEURAL_WEIGHTS_DIR) if f.endswith(".pth") and f"epoch_{epoch}_" in f]
    if not weights_files:
        raise ValueError(f"No saved weights found for epoch {epoch}.")

    time_indices = []
    print(f"weights_files = \n{weights_files}\n")
    try:
        for w in weights_files:
            # print(f"w = {w}")
            ind = w.split("_")
            # print(f"ind = {ind}")
            ind4 = ind[4]
            # print(f"ind4 = {ind4}")
            # print("\n")
            time_indices.append(int(ind4))
    except:
        pass
    return max(time_indices)


def read_pickle(directory, filename, finger_index, validate=False):
    long_file_name = f"{directory}/{filename}_{finger_index}{'_validate' if validate else ''}.pkl"

    with open(long_file_name, "rb") as file:
        output = pickle.load(file)
        print(f"Loaded {type(output)} from {long_file_name}")

    return output


def save_pickle(path: str, object1):
    with open(path, "wb") as f:
        pickle.dump(object1, f)


def compute_small_bounding_box(mesh_points: np.ndarray) -> (np.ndarray, np.ndarray):
    """Compute the smallest bounding box for the vertices."""
    b_min = np.min(mesh_points, axis=0)
    b_max = np.max(mesh_points, axis=0)
    return b_min, b_max


class MeshEncoder(nn.Module):
    """
    Neural network that encodes a 3D mesh into a latent vector.
    Args:
        input_dim (int): Dimensionality of the input vertices.
        latent_dim (int): Dimensionality of the latent vector.
    """

    def __init__(self, input_dim: int = 9001, latent_dim: int = 64):
        super(MeshEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, latent_dim)

    def forward(self, vertices):
        """
        Forward pass for encoding vertices into a latent vector.
        Args:
            vertices (torch.Tensor): Input tensor of shape (batch_size, num_vertices, input_dim).
        Returns:
            torch.Tensor: Latent vector of shape (batch_size, latent_dim).
        """
        x = F.relu(self.fc1(vertices))
        x = F.relu(self.fc2(x))
        latent_vector = self.fc3(x)  # Aggregate over vertices
        return latent_vector


class SDFCalculator(nn.Module):
    """
    Neural network that calculates SDF values from a latent vector and 3D coordinates.
    Args:
        latent_dim (int): Dimensionality of the latent vector.
        input_dim (int): Dimensionality of the 3D coordinates (default 3 for x, y, z).
    """

    def __init__(self, latent_dim: int = 256, input_dim: int = 3):
        super(SDFCalculator, self).__init__()
        self.fc1 = nn.Linear(latent_dim + input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, latent_vector, coordinates):
        """
        Forward pass to calculate SDF values.
        Args:
            latent_vector (torch.Tensor): Latent vector of shape (batch_size, latent_dim).
            coordinates (torch.Tensor): Input tensor of shape (batch_size, num_points, input_dim).
        Returns:
            torch.Tensor: SDF values of shape (batch_size, num_points, 1).
        """
        batch_size, num_points, _ = coordinates.size()
        latent_repeated = latent_vector.unsqueeze(1).repeat(1, num_points, 1)  # Repeat latent vector for each point
        inputs = torch.cat([latent_repeated, coordinates], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        sdf_values = self.fc3(x)
        return sdf_values


def get_path(name: str, epoch_index: int, time_index: int, finger_index: int, extension="pth"):
    return os.path.join(NEURAL_WEIGHTS_DIR, f"{name}_epoch_{epoch_index}_time_{time_index}_finger_{finger_index}.{extension}")


def load_dict_from_path(object1, path):
    if os.path.exists(path):
        object1.load_state_dict(torch.load(path))
        print(f"Loaded encoder weights from {path}.")
    else:
        raise FileNotFoundError(f"Weights/State file not found: {path} Doesn't exist")


class TrainingContext:
    def __init__(
        self, encoder: MeshEncoder, sdf_calculator: SDFCalculator, finger_index: int, number_shape_per_familly: int, learning_rate: float
    ):
        self.finger_index = finger_index

        self.previous_time_index: Optional[int] = None
        self.previous_epoch_index: Optional[int] = None

        self.previous_encoder_weights_epoch = None
        self.previous_calculator_weights_epoch = None
        self.previous_encoder_weights_time = None
        self.previous_calculator_weights_time = None
        self.previous_time_index = None
        self.previous_epoch_index = None

        self.previous_optimizer_state_epoch = None
        self.previous_scheduler_state_epoch = None
        self.previous_optimizer_state_time = None
        self.previous_scheduler_state_time = None

        self.mesh_encoder = encoder
        self.sdf_calculator = sdf_calculator

        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            list(self.mesh_encoder.parameters()) + list(self.sdf_calculator.parameters()),
            lr=learning_rate,
        )
        self.scheduler: ReduceLROnPlateau = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.8, patience=20, verbose=True)

        self.loss_tracker: list[np.ndarray] = [np.zeros(number_shape_per_familly)]
        self.loss_tracker_validate: list[np.ndarray] = [np.zeros(number_shape_per_familly)]

    def load_model_weights(self, epoch_index, time_index):
        encoder_weights_path = get_path("encoder", epoch_index, time_index, self.finger_index)
        calculator_weights_path = get_path("sdf_calculator", epoch_index, time_index, self.finger_index)
        optimizer_state_path = get_path("optimizer", epoch_index, time_index, self.finger_index)
        scheduler_state_path = get_path("scheduler", epoch_index, time_index, self.finger_index)
        loss_tracker_path = get_path("loss_tracker", epoch_index, time_index, self.finger_index, extension="pkl")
        loss_tracker_validate_path = get_path("loss_tracker_validate", epoch_index, time_index, self.finger_index, extension="pkl")

        load_dict_from_path(self.mesh_encoder, encoder_weights_path)
        load_dict_from_path(self.sdf_calculator, calculator_weights_path)
        print(f"loading optimizer from {optimizer_state_path}")
        load_dict_from_path(self.optimizer, optimizer_state_path)
        print(f"loading scheduler from {scheduler_state_path}")
        load_dict_from_path(self.scheduler, scheduler_state_path)

        with open(loss_tracker_path, "rb") as file:
            self.loss_tracker = pickle.load(file)
        with open(loss_tracker_validate_path, "rb") as file:
            self.loss_tracker_validate = pickle.load(file)

    def save_model_weights(self, mode: SaveMode):

        if self.previous_time_index is None:
            print("Nothing to save, nothing was done yet")
            return

        if self.previous_epoch_index is None and mode == SaveMode.NowEpoch:
            print("Nothing to save, nothing was done yet")
            return

        if self.previous_time_index is not None and self.previous_epoch_index is None:
            print("Setting previous epoch index to 0")
            self.previous_epoch_index = 0

        if self.previous_epoch_index is None:
            print(f"pei: {self.previous_epoch_index}")
            print("IMPOSSIBLE SCENARIO HAPPENED, EXITING")
            exit(42069)

        print(f"Saving Node: {mode}")
        if mode == SaveMode.NowTime:
            print(f"\nNowTime, e: {self.previous_epoch_index}, t:{self.previous_time_index}\n")
            epoch_index = self.previous_epoch_index + 1
            time_index = self.previous_time_index
            encoder_weights = self.previous_encoder_weights_time
            sdf_calculator_weights = self.previous_calculator_weights_time
            optimizer_state = self.previous_optimizer_state_time
            scheduler_state = self.previous_scheduler_state_time
            loss_tracker = self.loss_tracker
            loss_tracker_validate = self.loss_tracker_validate

        if mode == SaveMode.NowEpoch:
            print(f"\nNow Epoch, e: {self.previous_epoch_index}, t:{self.previous_time_index}\n")
            epoch_index = self.previous_epoch_index
            time_index = 0
            encoder_weights = self.previous_encoder_weights_epoch
            sdf_calculator_weights = self.previous_calculator_weights_epoch
            optimizer_state = self.previous_optimizer_state_epoch
            scheduler_state = self.previous_scheduler_state_epoch
            loss_tracker = self.loss_tracker[: epoch_index + 1]
            loss_tracker_validate = self.loss_tracker_validate[: epoch_index + 1]

        elif mode == SaveMode.NextTimeItteration:
            print(f"\nNextTime, e: {self.previous_epoch_index}, t:{self.previous_time_index}\n")
            epoch_index = self.previous_epoch_index
            time_index = self.previous_time_index + 1
            encoder_weights = self.previous_encoder_weights_time
            sdf_calculator_weights = self.previous_calculator_weights_time
            optimizer_state = self.previous_optimizer_state_time
            scheduler_state = self.previous_scheduler_state_time
            loss_tracker = self.loss_tracker
            loss_tracker_validate = self.loss_tracker_validate

        elif mode == SaveMode.NextEpochItteration or mode == SaveMode.End:
            print(f"\nNext Epoch, e: {self.previous_epoch_index}, t:{self.previous_time_index}\n")
            epoch_index = self.previous_epoch_index + 1
            time_index = 0
            encoder_weights = self.previous_encoder_weights_epoch
            sdf_calculator_weights = self.previous_calculator_weights_epoch
            optimizer_state = self.previous_optimizer_state_epoch
            scheduler_state = self.previous_scheduler_state_epoch
            loss_tracker = self.loss_tracker
            loss_tracker_validate = self.loss_tracker_validate

        print(f"Saving to Epoch Index: {epoch_index} | Time Index: {time_index}")

        encoder_weights_path = get_path("encoder", epoch_index, time_index, self.finger_index)
        calculator_weights_path = get_path("sdf_calculator", epoch_index, time_index, self.finger_index)
        optimizer_state_path = get_path("optimizer", epoch_index, time_index, self.finger_index)
        scheduler_state_path = get_path("scheduler", epoch_index, time_index, self.finger_index)
        loss_tracker_path = get_path("loss_tracker", epoch_index, time_index, self.finger_index, extension="pkl")
        loss_tracker_validate_path = get_path("loss_tracker_validate", epoch_index, time_index, self.finger_index, extension="pkl")

        torch.save(encoder_weights, encoder_weights_path)
        torch.save(sdf_calculator_weights, calculator_weights_path)
        torch.save(optimizer_state, optimizer_state_path)
        torch.save(scheduler_state, scheduler_state_path)
        save_pickle(loss_tracker_path, loss_tracker)
        save_pickle(loss_tracker_validate_path, loss_tracker_validate)

        print(f"Saved encoder weights to {encoder_weights_path}")
        print(f"Saved SDF calculator weights to {calculator_weights_path}")
        print(f"Saved optimizer state to {optimizer_state_path}")
        print(f"Saved scheduler state to {scheduler_state_path}")
        print(f"Saved loss tracker to {loss_tracker_path}")
        print(f"Saved loss tracker validate to {loss_tracker_validate_path}")

    def time_update(self, time_index):
        self.previous_encoder_weights_time = copy.deepcopy(self.mesh_encoder.state_dict())

        self.previous_calculator_weights_time = copy.deepcopy(self.sdf_calculator.state_dict())
        self.previous_optimizer_state_time = copy.deepcopy(self.optimizer.state_dict())
        self.previous_scheduler_state_time = copy.deepcopy(self.scheduler.state_dict())
        self.previous_time_index = time_index
        # print(f"time update called, {self.previous_time_index}")

    def epoch_update(self, epoch_index):
        self.previous_encoder_weights_epoch = copy.deepcopy(self.mesh_encoder.state_dict())
        self.previous_calculator_weights_epoch = copy.deepcopy(self.sdf_calculator.state_dict())
        self.previous_optimizer_state_epoch = copy.deepcopy(self.optimizer.state_dict())
        self.previous_scheduler_state_epoch = copy.deepcopy(self.scheduler.state_dict())
        self.previous_epoch_index = epoch_index


def train_model(
    training_context: TrainingContext,
    vertices_tensor,
    sdf_points,
    sdf_values,
    sdf_points_validate,
    sdf_values_validate,
    epochs=1000,
    start_epoch=0,
    start_time=0,
):
    """
    Train the mesh encoder and SDF calculator sequentially over time steps.

    Args:
        training_context (TrainingContext): The context of the training. It has the neural networks, optimizer and scheduler and previous data
        vertices_tensor (torch.Tensor): Vertices of the shapes (num_time_steps, num_vertices, vertex_dim).
        sdf_points (torch.Tensor): Points for SDF computation (num_time_steps, num_points, 3).
        sdf_values (torch.Tensor): Ground truth SDF values (num_time_steps, num_points).
        latent_dim (int): Dimensionality of the latent vector.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        start_epoch (int): Epoch to start training from.
        start_time (int): Time index to start training from.
    """
    global stop_time_signal, stop_epoch_signal
    global dL2

    # Convert inputs to PyTorch tensors
    vertices_tensor = torch.tensor(vertices_tensor, dtype=torch.float32)
    # (time_steps, num_vertices, 3)
    sdf_points = torch.tensor(sdf_points, dtype=torch.float32)
    # (time_steps, num_points, 3)
    sdf_values = torch.tensor(sdf_values, dtype=torch.float32).unsqueeze(-1)
    # (time_steps, num_points, 1)

    sdf_points_validate = torch.tensor(sdf_points_validate, dtype=torch.float32)
    sdf_values_validate = torch.tensor(sdf_values_validate, dtype=torch.float32).unsqueeze(-1)

    criterion = nn.MSELoss()

    min_validate_loss = 9001.666
    for loss_validates in training_context.loss_tracker_validate:
        loss_validates_non_zero = loss_validates[loss_validates > 0]
        if len(loss_validates_non_zero) == 0:
            continue
        min_loss = np.min(loss_validates_non_zero)  # Avoid zero values
        if min_loss < min_validate_loss:
            min_validate_loss = min_loss

    min_training_loss = 9001.666
    for loss_trainings in training_context.loss_tracker:
        loss_trainings_non_zero = loss_trainings[loss_validates > 0]
        if len(loss_trainings_non_zero) == 0:
            continue
        min_loss = np.min(loss_trainings_non_zero)  # Avoid zero values
        if min_loss < min_training_loss:
            min_training_loss = min_loss

    validate_loss_not_increase_counter = 0
    validate_loss_not_increase_save = 30

    print("\n-------Start of Training----------\n")
    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f"start of epoch {epoch}")
        total_loss: float = 0
        total_validation_loss: float = 0
        for t_index in range(start_time if epoch == start_epoch else 0, vertices_tensor.shape[0]):
            training_context.optimizer.zero_grad()

            # ------------ Get data for the current time step
            # Flatten vertices (1, num_vertices * 3)
            vertices = vertices_tensor[t_index].view(1, -1)
            # Add batch dimension (1, num_points, 3)
            points = sdf_points[t_index].unsqueeze(0)
            ground_truth_sdf = sdf_values[t_index].unsqueeze(0)  # (1, num_points, 1)

            # Encode vertices to latent vector
            latent_vector = training_context.mesh_encoder(vertices)  # (1, latent_dim)

            # Predict SDF values
            predicted_sdf = training_context.sdf_calculator(latent_vector, points)
            # (1, num_points, 1)

            # Compute training loss
            loss = criterion(predicted_sdf, ground_truth_sdf)
            loss_training = loss.item()

            # Compute validation loss
            points_validate = sdf_points_validate[t_index].unsqueeze(0)
            ground_truth_sdf_validate = sdf_values_validate[t_index].unsqueeze(0)  # (1, num_points, 1)
            predicted_sdf_validate = training_context.sdf_calculator(latent_vector, points_validate)
            loss_validate = criterion(predicted_sdf_validate, ground_truth_sdf_validate).item()

            # ----- Validation upgrade check
            if loss_validate <= min_validate_loss:
                validation_not_upgrade = True
                validate_loss_not_increase_counter += 1
                min_validate_loss = loss_validate
            else:
                validate_loss_not_increase_counter = 0
                validation_not_upgrade = False

            # ----- Training upgrade check
            if loss_training <= min_training_loss:
                training_not_upgrade = True
                min_training_loss = loss_training
            else:
                training_not_upgrade = False

            # Training/Validate upgrade string and messages. (tusm)
            tusm = "Training No Upgrade"
            tus_start = " | " if (validation_not_upgrade or training_not_upgrade) else ""
            tus_end = tusm if training_not_upgrade else " " * len(tusm)
            tus = tus_start + tus_end
            vus = " | Validation No Upgrade" if validation_not_upgrade else ""

            if epoch < SCHEDULER_SWITCH_EPOCH:
                if validate_loss_not_increase_counter >= validate_loss_not_increase_save:
                    validate_loss_not_increase_counter = 0
                    training_context.save_model_weights(SaveMode.NowEpoch)
                    # Write append to a file called validation_tacker.txt epoch and t_index -1
                    print(f"\t\t\t\tSaving previou Epoch to File")
                    with open(f"validation_tracker_{training_context.finger_index}.txt", "a") as file:
                        file.write(f"Epoch: {epoch}, Time Index: {t_index - 1}\n")

            total_loss += loss_training
            total_validation_loss += loss_validate
            if epoch < SCHEDULER_SWITCH_EPOCH:
                training_context.scheduler.step(loss_training)
            # Custom logging for the learning rate
            current_lr = training_context.scheduler.get_last_lr()
            print(
                f"\t\tTime Iteration {t_index:03d}, Training Loss: {loss_training:.15f}, Validation Loss: {loss_validate:.15f}, Learning Rate: [{current_lr[0]:.9e}] {tus} {vus}"
            )

            loss.backward()
            training_context.optimizer.step()

            training_context.loss_tracker[epoch][time_index] = loss_training
            training_context.loss_tracker_validate[epoch][time_index] = loss_validate

            # Store weights in the previous time step (We assume from this part on, the for loop has ended and the rest is atomic and "hidden", like the t_index++ part)
            training_context.time_update(t_index)

            # Handle stop time signal
            if stop_time_signal:
                print(f"Stopping after time iteration {t_index + 1}/{vertices_tensor.shape[0]}.")
                training_context.save_model_weights(SaveMode.NextTimeItteration)
                return 4  # return with code 4

        # ------------------- End of time itteration
        training_context.loss_tracker.append(np.zeros(vertices_tensor.shape[0]))
        training_context.loss_tracker_validate.append(np.zeros(vertices_tensor.shape[0]))

        if epoch == SCHEDULER_SWITCH_EPOCH:
            print(f"reached epoch {SCHEDULER_SWITCH_EPOCH}, changing the scheduler to epoch wise")
            training_context.scheduler = ReduceLROnPlateau(
                training_context.optimizer,
                mode="min",
                factor=0.8,
                patience=3,
                verbose=True,
            )
            min_validate_loss = total_loss / vertices_tensor.shape[0]
            min_training_loss = total_validation_loss / vertices_tensor.shape[0]

        if epoch >= SCHEDULER_SWITCH_EPOCH:
            training_context.scheduler.step(total_loss / vertices_tensor.shape[0])
            # ----- Validation upgrade check
            if loss_validate <= min_validate_loss:
                validation_not_upgrade = True
                validate_loss_not_increase_counter += 1
                min_validate_loss = loss_validate
            else:
                validate_loss_not_increase_counter = 0
                validation_not_upgrade = False

            # ----- Training upgrade check
            if loss_training <= min_training_loss:
                training_not_upgrade = True
                min_training_loss = loss_training
            else:
                training_not_upgrade = False

            # Training/Validate upgrade string and messages. (tusm)
            tusm = "Training No Upgrade"
            tus_start = " | " if (validation_not_upgrade or training_not_upgrade) else ""
            tus_end = tusm if training_not_upgrade else " " * len(tusm)
            tus = tus_start + tus_end
            vus = " | Validation No Upgrade" if validation_not_upgrade else ""

            avg_tl = total_loss / vertices_tensor.shape[0]
            avg_vl = total_validation_loss / vertices_tensor.shape[0]

            training_distance, validation_distance = np.sqrt(avg_tl), np.sqrt(avg_vl)

            # Step the scheduler
            print(f" End of Epoch {epoch}/{epochs -1}, AVG Training Loss: {avg_tl}, AVG Validate Loss: { avg_tl } {tus} {vus}")
            print(
                f" Training distance: {training_distance}, Validation Distance: { validation_distance }, Distance Scale: {dL2}, Ratio: {validation_distance/dL2}"
            )

            if validate_loss_not_increase_counter >= validate_loss_not_increase_save:
                validate_loss_not_increase_counter = 0
                training_context.save_model_weights(SaveMode.NowEpoch)
                # Write append to a file called validation_tacker.txt epoch and t_index -1
                print(f"\t\t\t\tSaving previou Epoch to File")
                with open(f"validation_tracker_{training_context.finger_index}.txt", "a") as file:
                    file.write(f"Epoch: {epoch}, Time Index: {t_index - 1}\n")

        else:
            avg_tl = total_loss / vertices_tensor.shape[0]
            avg_vl = total_validation_loss / vertices_tensor.shape[0]

            training_distance, validation_distance = np.sqrt(avg_tl), np.sqrt(avg_vl)
            print(f" End of Epoch {epoch}/{epochs -1}, AVG Training Loss: {avg_tl}, AVG Validate Loss: { avg_tl }")
            print(
                f" Training distance: {training_distance}, Validation Distance: { validation_distance }, Distance Scale: {dL2}, Ratio: {validation_distance/dL2}"
            )
        training_context.epoch_update(epoch)

        # Handle stop epoch signal
        if stop_epoch_signal:
            print(f"Stopping after epoch {epoch + 1}.")
            training_context.save_model_weights(SaveMode.NextEpochItteration)
            return 5  # Exit with code 5

    print("Training complete.")

    training_context.save_model_weights(SaveMode.End)

    return 0


def main(start_from_zero=True, continue_training=False, epoch_index=None, time_index=None, finger_index=DEFAULT_FINGER_INDEX):
    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_termination_time)  # Kill (no -9)
    signal.signal(signal.SIGINT, handle_termination_epoch)  # KeyboardInterrupt
    signal.signal(signal.SIGTSTP, handle_termination_time)  # Ctrl+Z
    signal.signal(signal.SIGUSR1, handle_stop_time_signal)
    signal.signal(signal.SIGUSR2, handle_stop_epoch_signal)
    signal.signal(signal.SIGRTMIN, handle_save_epoch_signal)
    signal.signal(signal.SIGRTMIN + 1, handle_save_time_signal)

    # Ensure the weights directory exists
    os.makedirs(NEURAL_WEIGHTS_DIR, exist_ok=True)

    vertices_tensor = read_pickle(LOAD_DIR, "vertices_tensor", finger_index)
    sdf_points = read_pickle(LOAD_DIR, "sdf_points", finger_index)
    sdf_values = read_pickle(LOAD_DIR, "sdf_values", finger_index)
    sdf_points_validate = read_pickle(LOAD_DIR, "sdf_points", finger_index, validate=True)
    sdf_values_validate = read_pickle(LOAD_DIR, "sdf_values", finger_index, validate=True)
    print("\n")

    b_min, b_max = compute_small_bounding_box(vertices_tensor[0])
    dx = b_max - b_min
    dL = np.linalg.norm(dx)
    global dL2
    dL2 = dL / 2

    print(f"sdf_points.shape: {sdf_points.shape}")
    print(f"sdf_values.shape: {sdf_values.shape}")

    print(f"sdf_points_validate.shape: {sdf_points_validate.shape}")
    print(f"sdf_values_validate.shape: {sdf_values_validate.shape}")

    print(f"vertices_tensor.shape: {vertices_tensor.shape}")
    number_of_shape_per_familly = sdf_points.shape[0]
    print("\n")

    # Initialize models
    input_dim = vertices_tensor.shape[1] * vertices_tensor.shape[2]
    print(f"mesh encoder input_dim = {input_dim}")
    mesh_encoder = MeshEncoder(input_dim=input_dim, latent_dim=64)
    sdf_calculator = SDFCalculator(latent_dim=64)

    global training_context
    training_context = TrainingContext(mesh_encoder, sdf_calculator, finger_index, number_of_shape_per_familly, learning_rate=1e-2)

    # Load weights if continuing training
    if continue_training:
        training_context.load_model_weights(epoch_index, time_index)

    # Train model
    ret = train_model(
        training_context,
        vertices_tensor,
        sdf_points,
        sdf_values,
        sdf_points_validate,
        sdf_values_validate,
        epochs=1000,
        start_epoch=epoch_index or 0,
        start_time=time_index or 0,
    )

    return ret


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process preprocessed data with options to start or continue.")

    # Mutually exclusive group for start or continue training
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--start_from_zero",
        action="store_true",
        help="Start processing from the beginning.",
    )
    group.add_argument(
        "--continue_training",
        action="store_true",
        help="Continue processing from the last state.",
    )
    parser.add_argument("--finger_index", type=int, help=f"Say which finger position index we takes. Default {DEFAULT_FINGER_INDEX}")

    # Arguments for epoch and time indices
    parser.add_argument(
        "--epoch_index",
        type=int,
        help="Specify the epoch index to continue processing from.",
    )
    parser.add_argument(
        "--time_index",
        type=int,
        help="Specify the time index to continue processing from.",
    )

    args = parser.parse_args()

    # Validation for argument combinations
    if args.start_from_zero and (args.epoch_index is not None or args.time_index is not None):
        parser.error("--start_from_zero cannot be used with --epoch_index or --time_index.")

    if args.time_index is not None and args.epoch_index is None:
        parser.error("--time_index can only be used if --epoch_index is specified.")

    epoch_index, time_index = None, None

    if args.continue_training:
        if args.epoch_index is not None:
            epoch_index = args.epoch_index
            time_index = args.time_index or get_latest_saved_time_for_epoch(epoch_index)
        else:
            epoch_index, time_index = get_latest_saved_indices()
    elif args.start_from_zero:
        epoch_index, time_index = 0, 0

    if args.finger_index is None:
        finger_index = DEFAULT_FINGER_INDEX
    else:
        finger_index = args.finger_index

    # Call main and exit with the returned code
    ret = main(args.start_from_zero, args.continue_training, epoch_index, time_index, finger_index)
    exit(ret)
