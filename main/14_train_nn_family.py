"""
To do: 
1. Make the cycle length different using different parameters
2. In 'learn both' epoch switch, using order, set the correct learning rate to what they were. 
3. Create an array start and end to create a list of ranges. Within those rabge, backstep using total_loss + alpha* latent encoder validation. 
4. Create an alpha of epoch equation, and set the Hyper parameters for it
5. Make it so during latent regularization epoch, the learning rate is much slower. Like 10 times smaller
"""

import pickle
import argparse
import os
import signal
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Optional

from enum import Enum

from __send_notification_sl import main as send_notification_to_my_phone

# Directory containing the pickle files
LOAD_DIR = "./training_data"

# Directory where we save and load the neural weights
NEURAL_WEIGHTS_DIR = "./neural_weights"
DEFAULT_FINGER_INDEX = 730


# ------------------ Start of Hyper Parameters
LATENT_DIM = 128
START_ENCODER_LR = 0.05
START_SDF_CALCULATOR_LR = 0.01
REACTIVE_SDF_LR_VALUE = 0.01
EPOCH_WHERE_TIME_PATIENCE_STARTS_APPLYING = 2  # When the scheduling for time starts

EPOCH_SHUFFLING_START = 6  # When we start shuffling the time index rather then doing it /\/\/\
EPOCH_SCHEDULER_CHANGE = 20  # When we start stepping the scheduler with the avg validation loss
EPOCH_WHERE_DROP_MESH_LR = [12]  # Epoch where we set the mesh encoder to the sdf encoder learning rate

# Learning rate of the different scheduling strategy
TIME_PATIENCE = 30
EPOCH_PATIENCE = 2
TIME_FACTOR = 0.92
EPOCH_FACTOR = 0.75

EPOCH_WHERE_SAVE_ALL = 50  # the epochs where we save the weights after every run


NUMBER_EPOCHS = 1000  # Total number of epochs


# ---------------------- Learning Cycle Parameters
class Param(Enum):
    MeshEncoder = 0
    SDFCalculator = 1
    Neither = 2
    Both = 3

    # Define focus lengths for each learning mode


FOCUS_LENGTHS = {
    Param.Both: 3,  # Focus length for Both
    Param.MeshEncoder: 5,  # Focus length for MeshEncoder
    Param.SDFCalculator: 6,  # Focus length for SDFCalculator
}
CYCLE_ORDER = [Param.Both, Param.MeshEncoder, Param.SDFCalculator]  # Dynamic focus cycle order


def get_previous_focus(cycle_order, current_focus):
    """
    Returns the step before the given current_focus in the cycle order.

    Args:
        cycle_order (list): The list defining the focus cycle order.
        current_focus (Param): The enum value representing the current focus.

    Returns:
        Param: The enum value before current_focus in the cycle.
    """
    if current_focus not in cycle_order:
        raise ValueError(f"{current_focus} is not in the cycle_order list.")

    current_index = cycle_order.index(current_focus)  # Find the index of current_focus
    previous_index = (current_index - 1) % len(cycle_order)  # Get the index of the previous step (cyclically)
    return cycle_order[previous_index]


def generate_focus_switch_points(cycle_order, focus_lengths, number_epochs):
    """
    Generate arrays for switch points at the start of each focus step.

    Args:
        cycle_order (list): List of Param enums defining the focus cycle order.
        focus_lengths (dict): Dictionary mapping Param enums to their focus lengths.
        number_epochs (int): Total number of epochs.

    Returns:
        dict: Dictionary with Param keys and lists of switch epochs as values.
    """
    focus_switch_points = {focus: [] for focus in cycle_order}  # Initialize lists for each focus mode
    epoch = 0
    while epoch < number_epochs:
        for focus in cycle_order:
            if epoch < number_epochs:
                focus_switch_points[focus].append(epoch)
                epoch += focus_lengths[focus]
            else:
                break
    return focus_switch_points


# Generate arrays for switch points
focus_switch_points = generate_focus_switch_points(CYCLE_ORDER, FOCUS_LENGTHS, NUMBER_EPOCHS)

# Assign arrays to specific variables for clarity
EPOCH_LEARN_BOTH = focus_switch_points[Param.Both]
EPOCH_LEARN_MESH_ENCODER = focus_switch_points[Param.MeshEncoder]
EPOCH_LEARN_SDF_CALCULATOR = focus_switch_points[Param.SDFCalculator]


# Loop through CYCLE_ORDER and print arrays in the specified order
for param in CYCLE_ORDER:
    if param == Param.Both:
        print(f"EPOCH_LEARN_BOTH = {EPOCH_LEARN_BOTH}")
    elif param == Param.MeshEncoder:
        print(f"EPOCH_LEARN_MESH_ENCODER = {EPOCH_LEARN_MESH_ENCODER}")
    elif param == Param.SDFCalculator:
        print(f"EPOCH_LEARN_SDF_CALCULATOR = {EPOCH_LEARN_SDF_CALCULATOR}")


# Example usage of get_previous_focus
current_focus = Param.SDFCalculator
previous_focus = get_previous_focus(CYCLE_ORDER, current_focus)
print(f"The step before {current_focus.name} is {previous_focus.name}.")


# ----- Latent vector regularization to force difference
# Cancelled because it didn't work

# ------------------ END of Hyper Parameters


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


def load_pickle(path: str):
    with open(path, "rb") as file:
        output = pickle.load(file)
        print(f"Loaded {type(output)} from {path}")

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

    def __init__(self, input_dim: int = 9001, latent_dim: int = LATENT_DIM):
        super(MeshEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, latent_dim)

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
        x = F.relu(self.fc3(x))
        latent_vector = self.fc4(x)  # Aggregate over vertices
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
        self.fc1 = nn.Linear(latent_dim + input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

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
        x = F.relu(self.fc3(x))
        sdf_values = self.fc4(x)
        return sdf_values


def get_path(name: str, epoch_index: int, time_index: int, finger_index: int, extension="pth"):
    return os.path.join(NEURAL_WEIGHTS_DIR, f"{name}_epoch_{epoch_index}_time_{time_index}_finger_{finger_index}.{extension}")


def load_dict_from_path(object1, path):
    if os.path.exists(path):
        object1.load_state_dict(torch.load(path))
        print(f"Loaded encoder weights from {path}.")
    else:
        raise FileNotFoundError(f"Weights/State file not found: {path} Doesn't exist")


class CustomLRScheduler:
    def __init__(self, optimizer, factor=0.5, patience=10, verbose=False):
        """
        Custom learning rate scheduler with adjustable factors and patience.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer with parameter groups for the encoder and SDF calculator.
            factor (float): Multiplicative factor for reducing the learning rate.
            patience (int): Number of steps without improvement to wait before reducing the LR.
        """
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.steps_since_improvement = 0
        self.errors_since_saving = 0
        self.best_loss = float("inf")
        self.verbose = verbose

    def set_patience_and_factor(self, patience, factor):
        """
        Set the patience and factor for learning rate adjustment.

        Args:
            patience (int): New patience value.
            factor (float): New multiplicative factor.
        """
        self.patience = patience
        self.factor = factor
        if self.verbose:
            print(f"Scheduler updated: patience = {self.patience}, factor = {self.factor}")

    def set_min_loss(self, best_loss: float, reset_counter: bool = True):
        """
        Update the best loss value and optionally reset the counters.

        Args:
            best_loss (float): New best loss value.
            reset_counter (bool): Whether to reset the counter for steps since improvement.
        """
        self.best_loss = best_loss
        if reset_counter:
            self.steps_since_improvement = 0
            self.errors_since_saving = 0

    def set_encoder_lr(self, new_lr: float):
        """Set the learning rate for the MeshEncoder."""
        self.optimizer.param_groups[Param.MeshEncoder.value]["lr"] = new_lr
        if self.verbose:
            print(f"Updated MeshEncoder LR to {new_lr}")

    def set_sdf_calculator_lr(self, new_lr: float):
        """Set the learning rate for the SDFCalculator."""
        self.optimizer.param_groups[Param.SDFCalculator.value]["lr"] = new_lr
        if self.verbose:
            print(f"Updated SDFCalculator LR to {new_lr}")

    def get_last_lr(self):
        """Get the current learning rates for all parameter groups."""
        return [param_group["lr"] for param_group in self.optimizer.param_groups]

    def step(self, validation_loss, target=Param.Both, saving_factor: float = 2.0, epoch=0):
        """
        Update learning rates based on validation loss.

        Args:
            validation_loss (float): Validation loss to track improvement.
            target (Param): Which parameter group to adjust (MeshEncoder, SDFCalculator, or Both).
            saving_factor (float): Factor to decide when to return `1` for saving.

        Returns:
            validation_not_improved, lowered_lr, save_to_file
        """
        if validation_loss is None:
            raise ValueError("Validation loss must be provided for the scheduler to operate.")

        # Check for improvement
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.steps_since_improvement = 0
            self.errors_since_saving = 0
            validation_not_improved = False
        else:
            self.steps_since_improvement += 1
            self.errors_since_saving += 1
            validation_not_improved = True

        if validation_loss >= self.best_loss * 2 and epoch > 3:
            send_notification_to_my_phone("Machine Learning", "We lost it")

        # Reduce learning rates for the specified parameter group(s)
        if self.steps_since_improvement >= self.patience:
            if target in [Param.MeshEncoder, Param.Both]:
                current_lr = self.optimizer.param_groups[Param.MeshEncoder.value]["lr"]
                self.set_encoder_lr(current_lr * self.factor)

            if target in [Param.SDFCalculator, Param.Both]:
                current_lr = self.optimizer.param_groups[Param.SDFCalculator.value]["lr"]
                self.set_sdf_calculator_lr(current_lr * self.factor)

            self.steps_since_improvement = 0  # Reset patience counter
            return validation_not_improved, True, False

        # Trigger saving if errors since saving exceed patience * (saving_factor - 1)
        if self.errors_since_saving >= self.patience * (saving_factor - 1):
            self.errors_since_saving = 0  # Reset saving counter
            return validation_not_improved, True, False  # Indicate that a saving event should occur

        return validation_not_improved, False, False  # No adjustment was made


class DummyScheduler:
    def __init__(self):
        """
        A dummy scheduler that tracks training loss and determines whether
        training has improved or not based on a minimum recorded loss.
        """
        self.min_loss = float("inf")

    def set_min_loss(self, min_loss: float):
        """
        Set a new minimum loss for the scheduler to track improvements.

        Args:
            min_loss (float): The minimum loss value to set.
        """
        self.min_loss = min_loss
        print(f"Minimum loss set to: {self.min_loss}")

    def step(self, training_loss: float) -> bool:
        """
        Check if the current training loss indicates improvement over the minimum loss.

        Args:
            training_loss (float): The current training loss.

        Returns:
            bool: True if the training loss is NOT an improvement (training_not_upgrade),
                  False if the training loss is an improvement.
        """
        if training_loss < self.min_loss:
            self.min_loss = training_loss
            # print(f"\t\t\tNew minimum loss achieved: {self.min_loss}")
            return False  # Training is improved

        # print(f"\t\t\tNo improvement in training loss. Current: {training_loss}, Min: {self.min_loss}")
        return True  # Training not improved


class TrainingContext:
    def __init__(
        self,
        encoder: MeshEncoder,
        sdf_calculator: SDFCalculator,
        finger_index: int,
        number_shape_per_familly: int,
        encoder_lr: float,
        sdf_calculator_lr: float,
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

        # Define separate parameter groups for the optimizer
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            [
                {"params": self.mesh_encoder.parameters(), "lr": encoder_lr},
                {"params": self.sdf_calculator.parameters(), "lr": sdf_calculator_lr},
            ]
        )
        self.scheduler: CustomLRScheduler = CustomLRScheduler(self.optimizer, factor=TIME_FACTOR, patience=TIME_PATIENCE)
        self.dummy_scheduler: DummyScheduler = DummyScheduler()  # only for training improvement tracking

        self.loss_tracker: list[np.ndarray] = [np.zeros(number_shape_per_familly)]
        self.loss_tracker_validate: list[np.ndarray] = [np.zeros(number_shape_per_familly)]

        self.previous_mesh_encoder_lr = encoder_lr
        self.previous_sdf_calculator_lr = sdf_calculator_lr

    def get_learning_rates(self):
        """
        Retrieve the current learning rates for the encoder and SDF calculator.

        Returns:
            tuple: (encoder_lr, sdf_calculator_lr)
        """
        return self.scheduler.get_last_lr()

    def adjust_encoder_lr(self, new_lr: float):
        """
        Adjust the learning rate for the MeshEncoder.

        Args:
            new_lr (float): The new learning rate for the MeshEncoder.
        """
        self.scheduler.set_encoder_lr(new_lr)

    def adjust_sdf_calculator_lr(self, new_lr: float):
        """
        Adjust the learning rate for the SDFCalculator.

        Args:
            new_lr (float): The new learning rate for the SDFCalculator.
        """
        self.scheduler.set_sdf_calculator_lr(new_lr)

    def load_model_weights(self, epoch_index, time_index):
        encoder_weights_path = get_path("encoder", epoch_index, time_index, self.finger_index)
        calculator_weights_path = get_path("sdf_calculator", epoch_index, time_index, self.finger_index)
        optimizer_state_path = get_path("optimizer", epoch_index, time_index, self.finger_index)

        load_dict_from_path(self.mesh_encoder, encoder_weights_path)
        load_dict_from_path(self.sdf_calculator, calculator_weights_path)
        load_dict_from_path(self.optimizer, optimizer_state_path)

        loss_tracker_path = get_path("loss_tracker", epoch_index, time_index, self.finger_index, extension="pkl")
        loss_tracker_validate_path = get_path("loss_tracker_validate", epoch_index, time_index, self.finger_index, extension="pkl")
        self.loss_tracker_validate = load_pickle(loss_tracker_validate_path)
        self.loss_tracker = load_pickle(loss_tracker_path)

        scheduler_state_path = get_path("scheduler", epoch_index, time_index, self.finger_index)
        self.scheduler = load_pickle(scheduler_state_path)

        previous_mesh_encoder_lr_path = get_path("previous_mesh_encoder_lr", epoch_index, time_index, self.finger_index)
        previous_sdf_calculator_lr_path = get_path("previous_sdf_calculator_lr", epoch_index, time_index, self.finger_index)
        self.previous_mesh_encoder_lr = load_pickle(previous_mesh_encoder_lr_path)
        self.previous_sdf_calculator_lr = load_pickle(previous_sdf_calculator_lr_path)

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

        print(f"\nSaving Node: {mode}")
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
        save_pickle(scheduler_state_path, scheduler_state)
        save_pickle(loss_tracker_path, loss_tracker)
        save_pickle(loss_tracker_validate_path, loss_tracker_validate)

        previous_mesh_encoder_lr_path = get_path("previous_mesh_encoder_lr", epoch_index, time_index, self.finger_index)
        previous_sdf_calculator_lr_path = get_path("previous_sdf_calculator_lr", epoch_index, time_index, self.finger_index)
        save_pickle(previous_mesh_encoder_lr_path, self.previous_mesh_encoder_lr)
        save_pickle(previous_sdf_calculator_lr_path, self.previous_sdf_calculator_lr)

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
        self.previous_scheduler_state_time = copy.deepcopy(self.scheduler)
        self.previous_time_index = time_index
        # print(f"time update called, {self.previous_time_index}")

    def epoch_update(self, epoch_index):
        self.previous_encoder_weights_epoch = copy.deepcopy(self.mesh_encoder.state_dict())
        self.previous_calculator_weights_epoch = copy.deepcopy(self.sdf_calculator.state_dict())
        self.previous_optimizer_state_epoch = copy.deepcopy(self.optimizer.state_dict())
        self.previous_scheduler_state_epoch = copy.deepcopy(self.scheduler)
        self.previous_epoch_index = epoch_index


def get_previous_min(training_context, start_epoch):
    min_validate_loss = float("inf")
    min_training_loss = float("inf")

    if start_epoch < EPOCH_SCHEDULER_CHANGE:
        for loss_validates in training_context.loss_tracker_validate:
            loss_validates_non_zero = loss_validates[loss_validates > 0]
            if len(loss_validates_non_zero) == 0:
                continue
            min_loss = np.min(loss_validates_non_zero)  # Avoid zero values
            if min_loss < min_validate_loss:
                min_validate_loss = min_loss

        for loss_trainings in training_context.loss_tracker:
            loss_trainings_non_zero = loss_trainings[loss_validates > 0]
            if len(loss_trainings_non_zero) == 0:
                continue
            min_loss = np.min(loss_trainings_non_zero)  # Avoid zero values
            if min_loss < min_training_loss:
                min_training_loss = min_loss

    else:
        for loss_validates in training_context.loss_tracker_validate:
            loss_validates_non_zero = loss_validates[loss_validates > 0]
            if len(loss_validates_non_zero) == 0:
                continue
            min_loss = np.mean(loss_validates_non_zero)  # Avoid zero values
            if min_loss < min_validate_loss:
                min_validate_loss = min_loss

        for loss_trainings in training_context.loss_tracker:
            loss_trainings_non_zero = loss_trainings[loss_validates > 0]
            if len(loss_trainings_non_zero) == 0:
                continue
            min_loss = np.mean(loss_trainings_non_zero)  # Avoid zero values
            if min_loss < min_training_loss:
                min_training_loss = min_loss

    return min_training_loss, min_validate_loss


def get_upgrade_message(validation_not_upgrade: bool, training_not_upgrade: bool) -> str:
    """
    Construct an upgrade message to append to training/validation log entries.

    Args:
        validation_not_upgrade (bool): Whether validation has not improved.
        training_not_upgrade (bool): Whether training has not improved.

    Returns:
        str: The formatted upgrade message.
    """
    # Fixed length for messages to ensure alignment
    training_message = "Training No Upgrade"
    validation_message = "Validation No Upgrade"

    # Construct training message
    training_status = f" | {training_message}" if training_not_upgrade else f" | {' ' * len(training_message)}"

    # Construct validation message
    validation_status = f" | {validation_message}" if validation_not_upgrade else f"{' ' * (len(validation_message) + 3)}"

    # Combine messages
    return training_status + validation_status


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

    min_training_loss, min_validate_loss = get_previous_min(training_context, start_epoch)
    training_context.scheduler.set_min_loss(min_validate_loss)
    training_context.dummy_scheduler.set_min_loss(min_training_loss)

    print(f"\nPrevious mins: {min_training_loss}, {min_validate_loss}\n")

    training_context.save_model_weights(SaveMode.NowEpoch)

    print("\n-------Start of Training----------\n")

    training_context.previous_mesh_encoder_lr, training_context.previous_sdf_calculator_lr = training_context.get_learning_rates()

    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f"\nstart of epoch {epoch}")
        total_loss: float = 0
        total_validation_loss: float = 0

        all_ts = list(range(start_time if epoch == start_epoch else 0, vertices_tensor.shape[0]))

        if epoch < EPOCH_SHUFFLING_START:
            if epoch % 2 == 0:
                all_ts_shuffled = all_ts
            else:
                all_ts_shuffled = all_ts[::-1]

        else:
            all_ts_shuffled = np.random.permutation(all_ts)

        # Space

        if epoch == EPOCH_SHUFFLING_START:
            print("________________________STARTING TO SHUFFLE THE TIME ITTERATION________________________")

        if epoch == EPOCH_WHERE_DROP_MESH_LR:
            print("________________________DROPPING MESH ENCONDER LEARNING RATE TO SDF CALCULATOR LEARNING RATE________________________")
            _, sdf_calculator_lr = training_context.get_learning_rates()
            if sdf_calculator_lr == 0:
                sdf_calculator_lr = training_context.previous_sdf_calculator_lr
                print("------USING PREVIOUS SDF LR SINCE IT WAS 0 ------------")
            training_context.adjust_encoder_lr(sdf_calculator_lr)

        if epoch in EPOCH_LEARN_MESH_ENCODER:
            print("|||||||\t\t________________________LEARNING THE MESH ENCODING________________________")
            _, training_context.previous_sdf_calculator_lr = training_context.get_learning_rates()
            print(f"training_context.previous_sdf_calculator_lr = {training_context.previous_sdf_calculator_lr}")
            print(f"training_context.previous_mesh_encoder_lr = {training_context.previous_mesh_encoder_lr}")
            if training_context.previous_sdf_calculator_lr == 0:
                return Param.SDFCalculator.value
            prev = get_previous_focus(CYCLE_ORDER, Param.MeshEncoder)
            if prev == Param.SDFCalculator:
                training_context.adjust_encoder_lr(training_context.previous_mesh_encoder_lr)
            training_context.adjust_sdf_calculator_lr(0)

        elif epoch in EPOCH_LEARN_SDF_CALCULATOR:
            print("|||||||\t\t________________________LEARNING THE SDF CALCULATOR________________________")
            training_context.previous_mesh_encoder_lr, _ = training_context.get_learning_rates()
            print(f"training_context.previous_mesh_encoder_lr = {training_context.previous_mesh_encoder_lr}")
            print(f"training_context.previous_sdf_calculator_lr = {training_context.previous_sdf_calculator_lr}")
            if training_context.previous_mesh_encoder_lr == 0:
                return Param.MeshEncoder.value
            prev = get_previous_focus(CYCLE_ORDER, Param.SDFCalculator)
            print(f"\n{prev.name}")
            if prev == Param.MeshEncoder:
                print("prev")
                training_context.adjust_sdf_calculator_lr(training_context.previous_sdf_calculator_lr)
            training_context.adjust_encoder_lr(0)

        elif epoch in EPOCH_LEARN_BOTH:
            print(f"|||||||\t\t________________________LEARNING BOTH___________________________")
            prev = get_previous_focus(CYCLE_ORDER, Param.Both)
            print(f"training_context.previous_sdf_calculator_lr = {training_context.previous_sdf_calculator_lr}")
            print(f"training_context.previous_mesh_encoder_lr = {training_context.previous_mesh_encoder_lr}")
            if prev == Param.MeshEncoder:
                print("\n\n\n\t\t-------THIS SHOULD NEVER HAPPEN______________________________-XXXXXXXX")
                training_context.adjust_sdf_calculator_lr(training_context.previous_sdf_calculator_lr)
            if prev == Param.SDFCalculator:
                training_context.adjust_encoder_lr(training_context.previous_mesh_encoder_lr)
        else:
            pe, ps = training_context.get_learning_rates()
            if pe != 0:
                training_context.previous_mesh_encoder_lr = pe
            if ps != 0:
                training_context.previous_sdf_calculator_lr = ps

        current_lr = training_context.scheduler.get_last_lr()
        print(f"current lr = {current_lr}")
        if current_lr[0] == 0 and current_lr[1] == 0:
            return Param.Both.value

        # ------------------------------------------------------------------------------------------ FOR LOOP ---------------------------------------------------------------------
        all_latent_vector = []
        for i, t_index in enumerate(all_ts_shuffled):

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

            total_loss += loss_training
            total_validation_loss += loss_validate

            if epoch < EPOCH_SCHEDULER_CHANGE:
                if epoch < EPOCH_WHERE_TIME_PATIENCE_STARTS_APPLYING:
                    validation_not_upgrade, lowered_lr, save_to_file = training_context.scheduler.step(
                        loss_validate, Param.Neither, saving_factor=1.4, epoch=epoch
                    )
                else:
                    validation_not_upgrade, lowered_lr, save_to_file = training_context.scheduler.step(
                        loss_validate, Param.Both, saving_factor=1.4, epoch=epoch
                    )
                training_not_upgrade = training_context.dummy_scheduler.step(loss_training)

                # Custom logging for the learning rate
                current_lr = training_context.scheduler.get_last_lr()

                upgrade_message = get_upgrade_message(validation_not_upgrade, training_not_upgrade)

            ps1 = f"\t{i:03d}: Time Iteration {t_index:03d}, Training Loss: {loss_training:.15f}, "
            ps2 = f"Validation Loss: {loss_validate:.15f}, Learning Rate: "
            ps3 = f"{current_lr}"
            ps4 = f"{upgrade_message}" if epoch < EPOCH_SCHEDULER_CHANGE else ""

            print(ps1 + ps2 + ps3 + ps4)

            training_context.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            training_context.optimizer.step()

            training_context.loss_tracker[epoch][i] = loss_training
            training_context.loss_tracker_validate[epoch][i] = loss_validate

            # Store weights in the previous time step (We assume from this part on, the for loop has ended and the rest is atomic and "hidden", like the t_index++ part)
            training_context.time_update(t_index)

            # Handle stop time signal
            if stop_time_signal:
                print(f"Stopping after time iteration {i + 1}/{vertices_tensor.shape[0]}.")
                training_context.save_model_weights(SaveMode.NextTimeItteration)
                return 4  # return with code 4

        # ------------------- End of time itteration
        training_context.loss_tracker.append(np.zeros(vertices_tensor.shape[0]))
        training_context.loss_tracker_validate.append(np.zeros(vertices_tensor.shape[0]))

        avg_tl = total_loss / vertices_tensor.shape[0]
        avg_vl = total_validation_loss / vertices_tensor.shape[0]

        training_distance, validation_distance = np.sqrt(avg_tl), np.sqrt(avg_vl)
        if epoch == EPOCH_SCHEDULER_CHANGE:
            print(
                f"________________________REACHED EPOCH: {EPOCH_SCHEDULER_CHANGE}, CHANGING THE SCHEDULING TO EPOCH WISE________________________"
            )
            training_context.scheduler.set_min_loss(avg_vl)
            training_context.dummy_scheduler.set_min_loss(avg_tl)

        if epoch >= EPOCH_SCHEDULER_CHANGE:
            validation_not_upgrade, lowered_lr, save_to_file = training_context.scheduler.step(
                avg_vl, Param.Both, saving_factor=2.0, epoch=epoch
            )
            training_not_upgrade = training_context.dummy_scheduler.step(avg_tl)

            upgrade_message = get_upgrade_message(validation_not_upgrade, training_not_upgrade)

            # Step the scheduler
            print(f" End of Epoch {epoch}/{epochs -1}, AVG Training Loss: {avg_tl}, AVG Validate Loss: { avg_vl }{upgrade_message}")
            ps1 = f" Training distance: {training_distance}, "
            ps2 = f"Validation Distance: {validation_distance}, Distance Scale: {dL2}, "
            ps3 = f"Val Ratio: {validation_distance/dL2}"
            print(ps1 + ps2 + ps3)

            if save_to_file:
                print(f"\nNo validation change in {np.ceil(EPOCH_PATIENCE * 2)} epochs, saving to files")
                training_context.save_model_weights(SaveMode.NowEpoch)
                with open(f"validation_tracker_{training_context.finger_index}.txt", "a") as file:
                    file.write(f"Epoch: {epoch}, Time Index: {i - 1}\n")

        else:
            print(f" End of Epoch {epoch}/{epochs -1}, AVG Training Loss: {avg_tl}, AVG Validate Loss: { avg_vl }")
            ps1 = f" Training distance: {training_distance}, "
            ps2 = f"Validation Distance: {validation_distance}, Distance Scale: {dL2}, "
            ps3 = f"Val Ratio: {validation_distance/dL2}"
            print(ps1 + ps2 + ps3)

        training_context.epoch_update(epoch)

        if epoch % 10 == 1 or epoch < EPOCH_WHERE_SAVE_ALL:
            training_context.save_model_weights(SaveMode.NextEpochItteration)

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
    # signal.signal(signal.SIGTSTP, handle_termination_time)  # Ctrl+Z
    signal.signal(signal.SIGUSR1, handle_stop_time_signal)
    signal.signal(signal.SIGUSR2, handle_stop_epoch_signal)
    signal.signal(signal.SIGRTMIN, handle_save_epoch_signal)
    signal.signal(signal.SIGRTMIN + 1, handle_save_time_signal)

    # Ensure the weights directory exists
    os.makedirs(NEURAL_WEIGHTS_DIR, exist_ok=True)

    # send_notification_to_my_phone("Machine Learning", "Starting new run")

    end = 101
    # We don't use the last data, and this also lets us cut the data for tests

    vertices_tensor = read_pickle(LOAD_DIR, "vertices_tensor", finger_index)[0:end]
    sdf_points = read_pickle(LOAD_DIR, "sdf_points", finger_index)[0:end]
    sdf_values = read_pickle(LOAD_DIR, "sdf_values", finger_index)[0:end]
    sdf_points_validate = read_pickle(LOAD_DIR, "sdf_points", finger_index, validate=True)[0:end]
    sdf_values_validate = read_pickle(LOAD_DIR, "sdf_values", finger_index, validate=True)[0:end]
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
    mesh_encoder = MeshEncoder(input_dim=input_dim, latent_dim=LATENT_DIM)
    sdf_calculator = SDFCalculator(latent_dim=LATENT_DIM)

    global training_context
    training_context = TrainingContext(
        mesh_encoder,
        sdf_calculator,
        finger_index,
        number_of_shape_per_familly,
        encoder_lr=START_ENCODER_LR,
        sdf_calculator_lr=START_SDF_CALCULATOR_LR,
    )

    # Load weights if continuing training
    if continue_training:
        training_context.load_model_weights(epoch_index, time_index)

        training_context.scheduler = CustomLRScheduler(training_context.optimizer, factor=0.8, patience=EPOCH_PATIENCE)
        pow = 0
        for i in range(3 * pow):
            training_context.scheduler.step(i)

        # This is a monkeys patch to force reducing the scheduler

    try:
        # Train model
        ret = train_model(
            training_context,
            vertices_tensor,
            sdf_points,
            sdf_values,
            sdf_points_validate,
            sdf_values_validate,
            epochs=NUMBER_EPOCHS,
            start_epoch=epoch_index or 0,
            start_time=time_index or 0,
        )
        if ret == Param.Neither.value:
            send_notification_to_my_phone("Machine Learning", "It worked, End of training, come back")
        elif ret == Param.Both.value:
            send_notification_to_my_phone("Machine Learning", "All weights zeo")
        elif ret == Param.MeshEncoder.value:
            send_notification_to_my_phone("Machine Learning", "Previous Mesh Encoder 0")
        elif ret == Param.SDFCalculator.value:
            send_notification_to_my_phone("Machine Learning", "Previous Mesh Encoder 0")
        return ret

    except Exception as e:
        import traceback

        error_message = f"Training crashed with error: {str(e)}\n{traceback.format_exc()}"
        print(error_message)  # Optional: Log to the console
        send_notification_to_my_phone("Machine Learning", "Training crashed, come back")
        return 10


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
    """ Bellow is to:  Returns 0 if succeed, 1 if mesh_encoder_lr=0, 2 if sdf_calculator_lr = 0, 3 if both is"""
    """ Case 3 should not happen because its caught one cycle element in advance by 1 and 2. 1 and 2 is if we have 0 x and we save previous y to 0, well have prev_x = prev_y = 0"""
    """ where x and y are the sdf_calculator_lr and mesh_encoder_lr"""
    exit_map = [1, 2, 0, 3]
    # Ensure ret is within range
    if ret >= 0 and ret < len(exit_map):
        exit(exit_map[ret])  # Exit using the mapped value
    else:
        exit(ret)  # Exit using the original value if out of range
