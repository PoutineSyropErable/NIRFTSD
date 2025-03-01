"""
To do:
I'll already do the basic framework of what's bellow. I just need to get the bunny head center correctly.
Generate the correct index after filtering. Generate all these physics simulations and then modify main
Look at 


    THIS IS FOR MY NEXT TODOS, IF I WANT TO ENCODE MULTIPLE PHYSICS SIMULATION INTO IT
    The idea is that v1 is a vector that points from point of force, to center of head. It tell us it which direction the head will tilt.
    Then, if we assume the head tilt at a constant velocity, then u*t will be the head tilt vector.
    We want the difference between those two.

    Currently, there is only one vector. For one physics sim. So, it's t1 - t2
    we will have to generate multiple simulations
    Then, we find using the finger index of the simulation, the finger position.
    The head center is calculated once. Maybe at the start of this file.
    We calculate u for all physics simulations.
    Then, we can have:


    # Let's say we recreate a bunch of data and simulations with finger index on the head but not ears.
    # Let's say using a modified ./3_filter_points.py
    # Example: for a specific z range, so z < top head height, z > neck height
    for finger_index in simulations_finger_index:
        vertices_tensor = read_pickle(LOAD_DIR, "vertices_tensor", finger_index)[0:end]
        sdf_points = read_pickle(LOAD_DIR, "sdf_points", finger_index)[0:end]
        sdf_values = read_pickle(LOAD_DIR, "sdf_values", finger_index)[0:end]
        sdf_points_validate = read_pickle(LOAD_DIR, "sdf_points", finger_index, validate=True)[0:end]
        sdf_values_validate = read_pickle(LOAD_DIR, "sdf_values", finger_index, validate=True)[0:end]

    We can check if the element[0] of each is a repeat. if vertices_tensor[0] is always the same, then
    remove it from all except one.
    Then, we can extend the vertices_tensor.
    Then we write a function to get from vertices_tensor index, get the t_index and simulation_index.
    simulation_index is the nth simulation in the array append/extend

    and t_index is the 0..101 saved time steps.
    With simulation index, we can do
    (We have a batch of 2 shape)
    for each shape:
    t_index, simulation_index = get_simulation_index(false_t_index)
    # false_t_index is the current t_index. Since there's just one shape. But we'll have to modify the loop to

    simulation_indices.append(simulation_index)
    time_indices.append(t_index)

"""

import re
import pickle
import argparse
import os
import signal
import time
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Optional, Tuple, Callable, Dict, List
from enum import Enum

from __send_notification_sl import main as send_notification_to_my_phone


class SignalType(Enum):
    TERMINATE_EPOCH = signal.SIGINT  # Ctrl + C
    STOP_NEXT_EPOCH = signal.SIGUSR2
    SAVE_LAST_EPOCH = signal.SIGRTMIN + 1
    SAVE_NEXT_EPOCH = signal.SIGRTMIN + 2


# Directory containing the pickle files
LOAD_DIR = "./training_data"

# Directory where we save and load the neural weights
NEURAL_WEIGHTS_DIR = "./neural_weights"
DEFAULT_FINGER_INDEX = 730


# ------------------ Start of Hyper Parameters
LATENT_DIM = 128
START_ENCODER_LR = 0.001
START_SDF_CALCULATOR_LR = 0.005
EPOCH_WHERE_TIME_PATIENCE_STARTS_APPLYING = 3  # When the scheduling for time starts

EPOCH_SHUFFLING_START = 0  # When we start shuffling the time index rather then doing it /\/\/\
EPOCH_SCHEDULER_CHANGE = 50  # When we start stepping the scheduler with the avg validation loss
EPOCH_WHERE_DROP_MESH_LR = [-1]  # Epoch where we set the mesh encoder to the sdf encoder learning rate
EPOCH_WHERE_RESET_MIN_LOSS = [2, 3, 4]

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

    # Define cycle lengths for each learning mode


INITIAL_MESH_CYCLE_LENGTH = 5


def mesh_encoder_length(n: int):
    return 1

    if n == 1:
        return INITIAL_MESH_CYCLE_LENGTH
    else:
        return 5


def both_length(n: int):
    return 50


def sdf_calculator_length(n: int):
    return 1


CYCLE_LENGTHS = {
    Param.MeshEncoder: mesh_encoder_length,  # cycle length for MeshEncoder
    Param.Both: both_length,  # cycle length for Both
    Param.SDFCalculator: sdf_calculator_length,  # cycle length for SDFCalculator
}
CYCLE_ORDER = [Param.Both, Param.MeshEncoder, Param.SDFCalculator]  # Dynamic cycle cycle order


# --------------------------------- Latent Regularisation HyperParameter functions
def alpha_latent(epoch: int) -> float:
    if epoch < 200:
        return 0.0005 / (epoch + 1) ** (1.4)
    else:
        return 0


def alpha_sdf(epoch: int) -> float:
    return 1
    if epoch <= INITIAL_MESH_CYCLE_LENGTH:
        return 0
    else:
        return 1


def get_simulation_index(false_t_index) -> Tuple[int, int]:
    """
    NEED TO MODIFY THIS FUNCTION. IT SHOULD DO SOME KIND OF false_t_index / 101, false_t_index % 101
    but modify it if we remove the center position.
    Returns:
        returns Tuple[int, int]

    return simulation_index, t_index
    """
    t_index = false_t_index
    simulation_index = 0
    # Once there's multiple shapes, modify this function
    return simulation_index, t_index


FORCE_DIRECTIONS: np.ndarray = np.array([[1.0, 0.0, 0.0]])
# Init value of correct type. In Main, it is set properly


def get_force_direction_vectors(simulation_index1: int, simulation_index2: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        returns Tuple[np.ndarray, np.ndarray]

    return v1, v2
    """
    v1 = FORCE_DIRECTIONS[simulation_index1]
    v2 = FORCE_DIRECTIONS[simulation_index2]
    return v1, v2


ELEMENT_0_CUT_IN_MAIN: bool = False


def get_time_diff_general(t1: float, t2: float, v1: Optional[np.ndarray] = None, v2: Optional[np.ndarray] = None):
    """
    t1_is_ref: If t1, u1 belongs to the first simulations. The one where we didn't crop the first frame, then it's t=0 means 0 movement
    Therefore, when t = 0, then there was 0 movement.

    However, if its from another simulations, where we removed the [0] element to stop repeating the
    reference position N times for N simulations, then the 0th element is actually not for t = 0, but for t = dt
    Hence, we must add 1.

    HOWEVER, if it is discovered that for all of them, the [0] element is for t=dt, then we can skip the t1_is_ref

    but for the other, then when t = 0, it's actually the first movemement frame of the animation, so we need to add one to t_index

    Returns:
        returns float

    return time_diff
    """

    if v1 is None or v2 is None:
        time_diff = np.abs(t1 - t2)
        return time_diff

    if ELEMENT_0_CUT_IN_MAIN:
        t1_is_ref: bool = v1 == FORCE_DIRECTIONS[0]
        t2_is_ref: bool = v2 == FORCE_DIRECTIONS[0]
        if not t1_is_ref:
            t1 = t1 + 1
        if not t2_is_ref:
            t2 = t2 + 1

    time_diff = np.linalg.norm(t2 * v2 - t1 * v1)
    # v1 and v2 should be normalised
    return time_diff


def time_diff_hyperparam_function(time_diff: int) -> float:
    return np.sqrt((time_diff) ** 2 + 5 * time_diff + 10)


def get_latent_loss(epoch, time_diff, cosine_similarity_penalty):
    f_td = time_diff_hyperparam_function(time_diff)
    latent_loss = (1 / f_td) / (torch.sqrt(1e-6 + torch.abs(1 - cosine_similarity_penalty)))
    return latent_loss


# -------------------------------------- CYCLE HELPERS. This generate the used: CYCLE_SWITCH_DICT
def get_next_cycle(current_cycle):
    """
    Returns the step before the given current_cycle in the cycle order.

    Args:
        cycle_order (list): The list defining the cycle cycle order.
        current_cycle (Param): The enum value representing the current cycle.

    Returns:
        Param: The enum value before current_cycle in the cycle.
    """
    if current_cycle not in CYCLE_ORDER:
        raise ValueError(f"{current_cycle} is not in the cycle_order list.")

    current_index = CYCLE_ORDER.index(current_cycle)  # Find the index of current_cycle
    previous_index = (current_index + 1) % len(CYCLE_ORDER)  # Get the index of the previous step (cyclically)
    return CYCLE_ORDER[previous_index]


def get_previous_cycle(current_cycle):
    """
    Returns the step before the given current_cycle in the cycle order.

    Args:
        cycle_order (list): The list defining the cycle cycle order.
        current_cycle (Param): The enum value representing the current cycle.

    Returns:
        Param: The enum value before current_cycle in the cycle.
    """
    if current_cycle not in CYCLE_ORDER:
        raise ValueError(f"{current_cycle} is not in the cycle_order list.")

    current_index = CYCLE_ORDER.index(current_cycle)  # Find the index of current_cycle
    previous_index = (current_index - 1) % len(CYCLE_ORDER)  # Get the index of the previous step (cyclically)
    return CYCLE_ORDER[previous_index]


def generate_cycles():
    current_epoch: int = 0
    current_cycle: Param = next(iter(Param))

    cycle_switch_dict: Dict[int, Param] = {}
    cycle_number_dict = {Param.MeshEncoder: 1, Param.Both: 1, Param.SDFCalculator: 1}
    while True:
        cycle_switch_dict[current_epoch] = current_cycle

        length_function: Callable[[int], int] = CYCLE_LENGTHS[current_cycle]
        current_cycle_number: int = cycle_number_dict[current_cycle]
        cycle_length: int = length_function(current_cycle_number)
        current_epoch += cycle_length

        cycle_number_dict[current_cycle] += 1

        current_cycle = get_next_cycle(current_cycle)

        if current_epoch >= NUMBER_EPOCHS:
            break

    return cycle_switch_dict


CYCLE_SWITCH_DICT: Dict[int, Param] = generate_cycles()

print(f"cycle_switch_dict = \n{CYCLE_SWITCH_DICT}\n")


# ------------------ END of Hyper Parameters


# -------------------------------------------- SIGNALS AND SAVING ------------------------------------
# Global values for signals
stop_time_signal = False
stop_epoch_signal = False
save_next_time_signal = False
save_next_epoch_signal = False


class SaveMode(Enum):
    NowTime = 1
    NowEpoch = 2
    NextTimeItteration = 3
    NextEpochItteration = 4
    End = 5


def handle_stop_epoch_signal(signum, frame):
    global stop_epoch_signal
    stop_epoch_signal = True
    print("Received stop epoch signal. Will stop after the current epoch iteration.")


def handle_termination_epoch(signum, frame):
    """
    Handle termination signals (SIGTERM, SIGINT).
    Save weights at the current epoch and time index before exiting.
    """
    print("\nhandling Epoch termination, save.NowEpoch, then exit\n")
    global training_context
    training_context.save_model_weights(SaveMode.NowEpoch)
    exit(2)


def handle_save_next_epoch_signal(signum, frame):
    global save_next_epoch_signal
    save_next_epoch_signal = True
    print("Received save next epoch signal. Will save results after the current epoch iteration.")


def handle_save_last_epoch_signal(signum, frame):
    """
    Handle termination signals (SIGTERM, SIGINT).
    Save weights at the current epoch and time index before exiting.
    """
    global training_context
    print("Received save previous epoch signal")
    training_context.save_model_weights(SaveMode.NowEpoch)


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
    return 0


# ------------------------------------------------------ CODE HELPER TO READ PRECALC
def read_pickle(directory, filename, finger_index, validate=False):
    long_file_name = f"{directory}/{filename}_{finger_index}{'_validate' if validate else ''}.pkl"

    with open(long_file_name, "rb") as file:
        output = pickle.load(file)
        print(f"Loaded {type(output)} from {long_file_name}")

    return output


def load_pickle(path: str, debug=True):
    with open(path, "rb") as file:
        output = pickle.load(file)
        if debug:
            print(f"Loaded {type(output)} from {path}")

    return output


def save_pickle(path: str, object1):
    with open(path, "wb") as f:
        pickle.dump(object1, f)


def compute_small_bounding_box(mesh_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the smallest bounding box for the vertices."""
    b_min = np.min(mesh_points, axis=0)
    b_max = np.max(mesh_points, axis=0)
    return b_min, b_max


# ------------------------------- Actual Start of Code : Class Definition ---------------
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


def load_dict_from_path(object1, path, debug=True):
    if os.path.exists(path):
        object1.load_state_dict(torch.load(path))
        if debug:
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
        self.last_execution_time = datetime.min  # Set to a very old time initially

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

        validation_ratio = np.sqrt(validation_loss) / dL2

        if epoch > EPOCH_SCHEDULER_CHANGE:
            if np.abs(validation_ratio - 0.2) <= 0.02:
                if False:
                    send_notification_to_my_phone("Machine Learning", f"Epoch = {epoch}. Val ratio={validation_ratio}")

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
            # Get the current time
            now = datetime.now()

            # Check if at least 1 minute has passed
            if now - self.last_execution_time >= timedelta(minutes=1):
                # Update last execution time
                self.last_execution_time = now
                validation_distance = np.sqrt(validation_loss)
                val_ratio = validation_distance / dL2
                if False:
                    send_notification_to_my_phone(
                        "Machine Learning", f"We lost it\nEpoch: {epoch}. Valdation loss: {validation_loss}, Validation Ratio: {val_ratio} "
                    )

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

    def step(self, training_loss: float, true_step=True) -> bool:
        """
        Check if the current training loss indicates improvement over the minimum loss.

        Args:
            training_loss (float): The current training loss.

        Returns:
            bool: True if the training loss is NOT an improvement (training_not_upgrade),
                  False if the training loss is an improvement.
        """
        if training_loss < self.min_loss:
            if true_step:
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
        self.dummy_scheduler_val: DummyScheduler = DummyScheduler()  # only for training improvement tracking
        self.dummy_scheduler: DummyScheduler = DummyScheduler()  # only for training improvement tracking

        self.loss_tracker: list[np.ndarray] = [np.zeros(number_shape_per_familly)]
        self.loss_tracker_validate: list[np.ndarray] = [np.zeros(number_shape_per_familly)]

        self.previous_mesh_encoder_lr = encoder_lr
        self.previous_sdf_calculator_lr = sdf_calculator_lr

    def reset(self, mesh_encoder_lr: float = START_ENCODER_LR, sdf_calculator_lr: float = START_SDF_CALCULATOR_LR):

        print(f"\n\n\t___---___reset\nmelr: {mesh_encoder_lr}, sclr: {sdf_calculator_lr}\n")
        # Define separate parameter groups for the optimizer
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            [
                {"params": self.mesh_encoder.parameters(), "lr": mesh_encoder_lr},
                {"params": self.sdf_calculator.parameters(), "lr": sdf_calculator_lr},
            ]
        )
        self.scheduler: CustomLRScheduler = CustomLRScheduler(self.optimizer, factor=EPOCH_FACTOR, patience=EPOCH_PATIENCE)
        self.dummy_scheduler: DummyScheduler = DummyScheduler()  # only for training improvement tracking
        self.previous_mesh_encoder_lr = mesh_encoder_lr
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

    def load_model_weights(self, epoch_index, time_index, debug=True):
        encoder_weights_path = get_path("encoder", epoch_index, time_index, self.finger_index)
        calculator_weights_path = get_path("sdf_calculator", epoch_index, time_index, self.finger_index)
        optimizer_state_path = get_path("optimizer", epoch_index, time_index, self.finger_index)

        load_dict_from_path(self.mesh_encoder, encoder_weights_path, debug)
        load_dict_from_path(self.sdf_calculator, calculator_weights_path, debug)
        load_dict_from_path(self.optimizer, optimizer_state_path, debug)

        loss_tracker_path = get_path("loss_tracker", epoch_index, time_index, self.finger_index, extension="pkl")
        loss_tracker_validate_path = get_path("loss_tracker_validate", epoch_index, time_index, self.finger_index, extension="pkl")
        self.loss_tracker_validate = load_pickle(loss_tracker_validate_path, debug)
        self.loss_tracker = load_pickle(loss_tracker_path, debug)

        scheduler_state_path = get_path("scheduler", epoch_index, time_index, self.finger_index)
        self.scheduler = load_pickle(scheduler_state_path, debug)

        previous_mesh_encoder_lr_path = get_path("previous_mesh_encoder_lr", epoch_index, time_index, self.finger_index)
        previous_sdf_calculator_lr_path = get_path("previous_sdf_calculator_lr", epoch_index, time_index, self.finger_index)
        self.previous_mesh_encoder_lr = load_pickle(previous_mesh_encoder_lr_path, debug)
        self.previous_sdf_calculator_lr = load_pickle(previous_sdf_calculator_lr_path, debug)

    def save_model_weights(self, mode: SaveMode):

        if self.previous_epoch_index is None or self.previous_epoch_index < 1:
            print("Nothing worth to save, nothing was done yet")
            return

        if self.previous_epoch_index is None and mode == SaveMode.NowEpoch:
            print("Nothing to save, nothing was done yet")
            return

        if mode == SaveMode.NowEpoch:
            print(f"\nNow Epoch, e: {self.previous_epoch_index}, t:{self.previous_time_index}\n")
            epoch_index = self.previous_epoch_index + 1
            # because its the epoch we'll be restarting from. So, if we are working on epoch 5, then epoch 4 was previous, and well restart from 5.
            time_index = 0
            encoder_weights = self.previous_encoder_weights_epoch
            sdf_calculator_weights = self.previous_calculator_weights_epoch
            optimizer_state = self.previous_optimizer_state_epoch
            scheduler_state = self.previous_scheduler_state_epoch
            loss_tracker = self.loss_tracker[: epoch_index + 1]
            loss_tracker_validate = self.loss_tracker_validate[: epoch_index + 1]

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


def get_previous_min(training_context: TrainingContext, start_epoch: int = 0, last_reset_epoch: int = 0) -> Tuple[float, float]:
    min_validate_loss = float("inf")
    min_training_loss = float("inf")

    if start_epoch < EPOCH_SCHEDULER_CHANGE:
        for loss_validates in training_context.loss_tracker_validate[last_reset_epoch:]:
            loss_validates_non_zero = loss_validates[loss_validates > 0]
            if len(loss_validates_non_zero) == 0:
                continue
            min_loss = np.min(loss_validates_non_zero)  # Avoid zero values
            if min_loss < min_validate_loss:
                min_validate_loss = min_loss

        for loss_trainings in training_context.loss_tracker[last_reset_epoch:]:
            loss_trainings_non_zero = loss_trainings[loss_trainings > 0]
            if len(loss_trainings_non_zero) == 0:
                continue
            min_loss = np.min(loss_trainings_non_zero)  # Avoid zero values
            if min_loss < min_training_loss:
                min_training_loss = min_loss

    else:
        for loss_validates in training_context.loss_tracker_validate[last_reset_epoch:]:

            loss_validates_non_zero = loss_validates[loss_validates > 0]
            if len(loss_validates_non_zero) == 0:
                continue
            min_loss = np.mean(loss_validates_non_zero)  # Avoid zero values
            if min_loss < min_validate_loss:
                min_validate_loss = min_loss

        for loss_trainings in training_context.loss_tracker[last_reset_epoch:]:

            loss_trainings_non_zero = loss_trainings[loss_trainings > 0]
            if len(loss_trainings_non_zero) == 0:
                continue
            min_loss = np.mean(loss_trainings_non_zero)  # Avoid zero values
            if min_loss < min_training_loss:
                min_training_loss = min_loss

    return min_training_loss, min_validate_loss


# ---------------- Training Logging Helper
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


# -------------------------------------- TRAINING CODE -------------------------
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
    reset: bool = False,
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
    global save_next_time_signal, save_next_epoch_signal
    global dL2
    global MESH_ONLY_LR_DIVIDE

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

    loss_validate, loss_training = float("inf"), float("inf")
    if not reset:
        min_training_loss, min_validate_loss = get_previous_min(training_context, start_epoch, 592)
        training_context.dummy_scheduler.set_min_loss(min_training_loss)
        training_context.dummy_scheduler_val.set_min_loss(min_validate_loss)

        print(f"\nPrevious mins: {min_training_loss}, {min_validate_loss}\n")
    else:
        min_training_loss, min_validate_loss = get_previous_min(training_context, start_epoch, 0)

    training_context.save_model_weights(SaveMode.NowEpoch)

    print("\n-------Start of Training----------\n")

    training_context.previous_mesh_encoder_lr, training_context.previous_sdf_calculator_lr = training_context.get_learning_rates()

    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f"\nstart of epoch {epoch}")
        total_loss: float = 0
        total_validation_loss: float = 0

        # Space

        if epoch == EPOCH_SHUFFLING_START:
            print("________________________STARTING TO SHUFFLE THE TIME ITTERATION________________________")

        if epoch in CYCLE_SWITCH_DICT:
            current_cycle = CYCLE_SWITCH_DICT[epoch]
            if current_cycle == Param.MeshEncoder:
                print("|||||||\t\t________________________LEARNING THE MESH ENCODING________________________ Taking sdf_lr/5")
                _, training_context.previous_sdf_calculator_lr = training_context.get_learning_rates()
                if training_context.previous_sdf_calculator_lr == 0:
                    return Param.SDFCalculator.value

                training_context.adjust_encoder_lr(training_context.previous_mesh_encoder_lr)
                training_context.adjust_sdf_calculator_lr(0)

            elif current_cycle == Param.SDFCalculator:
                print("|||||||\t\t________________________LEARNING THE SDF CALCULATOR________________________")
                training_context.previous_mesh_encoder_lr, _ = training_context.get_learning_rates()
                if training_context.previous_mesh_encoder_lr == 0:
                    return Param.MeshEncoder.value

                training_context.adjust_sdf_calculator_lr(training_context.previous_sdf_calculator_lr)
                training_context.adjust_encoder_lr(0)

            elif current_cycle == Param.Both:
                print(f"|||||||\t\t________________________LEARNING BOTH___________________________")
                prev = get_previous_cycle(Param.Both)
                if prev == Param.MeshEncoder:
                    training_context.adjust_sdf_calculator_lr(training_context.previous_sdf_calculator_lr)
                if prev == Param.SDFCalculator:
                    training_context.adjust_encoder_lr(training_context.previous_mesh_encoder_lr)

                print(f"training_context.previous_sdf_calculator_lr = {training_context.previous_sdf_calculator_lr}")
                print(f"training_context.previous_mesh_encoder_lr = {training_context.previous_mesh_encoder_lr}")
        else:
            pe, ps = training_context.get_learning_rates()
            if pe != 0:
                training_context.previous_mesh_encoder_lr = pe
            if ps != 0:
                training_context.previous_sdf_calculator_lr = ps

        if epoch in EPOCH_WHERE_RESET_MIN_LOSS:
            print("\t\t\t\t___-----_____----changing min loss")
            training_context.scheduler.set_min_loss(loss_validate)
            training_context.dummy_scheduler.set_min_loss(loss_training)

        current_lr = training_context.scheduler.get_last_lr()
        print(f"current lr = {current_lr}")
        if current_lr[0] == 0 and current_lr[1] == 0:
            return Param.Both.value

        all_ts = list(range(0, vertices_tensor.shape[0]))
        if len(all_ts) % 2 != 0:
            all_ts = all_ts[:-1]

        print(f"len(all_ts) = {len(all_ts)}")
        if epoch < EPOCH_SHUFFLING_START:
            if epoch % 2 == 0:
                all_ts_shuffled = all_ts
            else:
                all_ts_shuffled = all_ts[::-1]

        else:
            all_ts_shuffled = np.random.permutation(all_ts)

        # ------------------------------------------------------------------------------------------ FOR LOOP ---------------------------------------------------------------------
        for i2 in range(0, len(all_ts), 2):
            all_t_index = np.zeros(2, dtype=np.int64) - 1
            all_validation_loss = np.zeros(2, dtype=np.float64)

            all_latent_vector: List[Optional[torch.Tensor]] = [None, None]
            all_sdf_loss: List[Optional[torch.Tensor]] = [None, None]

            all_simulation_index = np.zeros(2, dtype=np.int64)
            for offset in range(2):
                i = i2 + offset
                false_t_index = all_ts_shuffled[i]
                simulation_index, t_index = get_simulation_index(false_t_index)
                all_t_index[offset] = t_index
                all_simulation_index[offset] = simulation_index

                # ------------ Get data for the current time step
                # Flatten vertices (1, num_vertices * 3)
                vertices = vertices_tensor[t_index].view(1, -1)
                # Add batch dimension (1, num_points, 3)
                points = sdf_points[t_index].unsqueeze(0)
                ground_truth_sdf = sdf_values[t_index].unsqueeze(0)  # (1, num_points, 1)

                # Encode vertices to latent vector
                latent_vector = training_context.mesh_encoder(vertices)  # (1, latent_dim)
                all_latent_vector[offset] = latent_vector

                # Predict SDF values
                predicted_sdf = training_context.sdf_calculator(latent_vector, points)
                # (1, num_points, 1)

                # Compute training loss
                sdf_loss = criterion(predicted_sdf, ground_truth_sdf)
                loss_training = sdf_loss.item()
                all_sdf_loss[offset] = sdf_loss

                # Compute validation loss
                points_validate = sdf_points_validate[t_index].unsqueeze(0)
                ground_truth_sdf_validate = sdf_values_validate[t_index].unsqueeze(0)  # (1, num_points, 1)
                predicted_sdf_validate = training_context.sdf_calculator(latent_vector, points_validate)
                loss_validate = criterion(predicted_sdf_validate, ground_truth_sdf_validate).item()

                total_loss += loss_training
                total_validation_loss += loss_validate
                all_validation_loss[offset] = loss_validate

                training_not_upgrade = training_context.dummy_scheduler.step(loss_training)
                validation_not_upgrade = training_context.dummy_scheduler_val.step(loss_validate)

                # Custom logging for the learning rate
                current_lr = training_context.scheduler.get_last_lr()

                upgrade_message = get_upgrade_message(validation_not_upgrade, training_not_upgrade)

                ps1 = f"\t\t{i:03d}: Time Iteration {t_index:03d}, Training Loss: {loss_training:.15f}, "
                ps2 = f"Validation Loss: {loss_validate:.15f}, Learning Rate: "
                ps3 = f"{current_lr}"
                ps4 = f"{upgrade_message}" if epoch < EPOCH_SCHEDULER_CHANGE else ""

                print(ps1 + ps2 + ps3 + ps4)

                training_context.loss_tracker[epoch][i] = loss_training
                training_context.loss_tracker_validate[epoch][i] = loss_validate

            # --- after 2 time itteration
            if any(value is None for value in all_latent_vector):
                print("Guard triggered: One of the latent vector is None")
                return 69
            if any(value is None for value in all_sdf_loss):
                print("Guard triggered: One of the SDF is None")
                return 666

            t_index1, t_index2 = all_t_index
            simulation_index1, simulation_index2 = all_simulation_index

            v1, v2 = get_force_direction_vectors(simulation_index1, simulation_index2)
            time_diff = get_time_diff_general(t_index1, t_index2, v1, v2)

            latent1, latent2 = all_latent_vector
            dot_product = torch.dot(latent1.flatten(), latent2.flatten())
            norm_latent1 = torch.norm(latent1)
            norm_latent2 = torch.norm(latent2)
            cosine_similarity_penalty = dot_product / (norm_latent1 * norm_latent2)

            latent_loss = get_latent_loss(epoch, time_diff, cosine_similarity_penalty)
            mean_loss = torch.mean(torch.stack(all_sdf_loss))

            alpha_s = alpha_sdf(epoch)
            alpha_l = alpha_latent(epoch)
            loss = alpha_s * mean_loss + alpha_l * latent_loss

            training_context.optimizer.zero_grad()
            loss.backward()
            training_context.optimizer.step()

            mean_val_loss = np.mean(all_validation_loss)
            valid_dist = np.sqrt(mean_val_loss)
            val_r = valid_dist / dL2

            up_msg = ""
            if epoch < EPOCH_SCHEDULER_CHANGE:
                reg_traing_no_up, _, _ = training_context.scheduler.step(loss.item(), Param.Both, saving_factor=1.4, epoch=epoch)
                if reg_traing_no_up:
                    up_msg = "Reg. Training No upgrade"

            print(
                f"\ttime_diff={time_diff:.2f}, cosine_similarity_penalty = {cosine_similarity_penalty.item()}, latent_loss = {latent_loss.item()}, ",
                end="",
            )
            print(f" mean_loss = {mean_loss.item()}, total_loss = {loss.item()}, val_r = {val_r} | {up_msg}\n")

            training_context.time_update(i)
            # ------------------- End of time itteration

        # ----------Every epoch
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
        if stop_epoch_signal or save_next_epoch_signal:
            print(f"Saving after epoch {epoch + 1}.")
            training_context.save_model_weights(SaveMode.NextEpochItteration)
            save_next_time_signal = False

        if stop_epoch_signal:
            return 5  # Exit with code 5

        print("")

    print("Training complete.")

    training_context.save_model_weights(SaveMode.End)

    return 0


# ---------------------------------------- Temporary functions, not tested


def compute_bunny_head_center(vertices, neck_height, head_top_height):
    """
    Computes the center of the bounding box formed by filtering vertices
    where neck_height <= z <= head_top_height.

    Args:
        vertices (np.ndarray): A (N, 3) NumPy array of vertices (x, y, z).
        neck_height (float): Lower bound on z-axis.
        head_top_height (float): Upper bound on z-axis.

    Returns:
        np.ndarray: The (x, y, z) center of the bounding box.
    """
    # Filter vertices within the specified z range
    filtered_vertices = vertices[(vertices[:, 2] >= neck_height) & (vertices[:, 2] <= head_top_height)]

    if filtered_vertices.size == 0:
        raise ValueError("No vertices found in the given z-range.")

    # Compute the bounding box min and max along each axis
    min_xyz = np.min(filtered_vertices, axis=0)
    max_xyz = np.max(filtered_vertices, axis=0)

    # Compute the center of the bounding box
    bbox_center = (min_xyz + max_xyz) / 2

    return bbox_center


# ------------------------------------- C like Main Function which takes sys arguments.
def main(
    start_from_zero: bool = True,
    continue_training: bool = False,
    epoch_index: Optional[int] = None,
    time_index: Optional[int] = None,
    finger_index: int = DEFAULT_FINGER_INDEX,
    reset: bool = False,
    melr: float = START_ENCODER_LR,
    sclr: float = START_SDF_CALCULATOR_LR,
):
    # Register signal handlers using SignalType Enum
    signal.signal(SignalType.TERMINATE_EPOCH.value, handle_termination_epoch)  # stop now and save previous epoch
    signal.signal(SignalType.STOP_NEXT_EPOCH.value, handle_stop_epoch_signal)  # stop next epoch
    signal.signal(SignalType.SAVE_LAST_EPOCH.value, handle_save_last_epoch_signal)
    signal.signal(SignalType.SAVE_NEXT_EPOCH.value, handle_save_next_epoch_signal)

    # Register additional signals directly (if not already in SignalType)
    signal.signal(signal.SIGTERM, handle_termination_epoch)  # kill
    # Kill -9 can't be caught
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

    """ Start of temporary thing to replace """
    global FORCE_DIRECTIONS, ELEMENT_0_CUT_IN_MAIN

    ELEMENT_0_CUT_IN_MAIN = False
    # ^^ If we end up cutting the 0th element of vertices_tensor and sdf_points...
    simulations_finger_index = np.array([730, 9001, 666, 420])
    # 730 is the actual current index of the finger_position file for a point on the head.
    # Once we generate other points and filters, we'll need to do
    # ^^ This should be obtained by reading from a file created by ./3_filter_points.py
    # We need to modify ./3_filter_points.py
    # Also, Let's create a file that shows all filtered finger force points on the bunny and when you click on it,
    # you can see the finger position (np.array) and finger_index. Let's also make the head center calculation in filter points.
    # And then we can also see the negated normal and the direction to bunny head center

    FORCE_DIRECTIONS = np.zeros((len(simulations_finger_index), 3))
    FINGER_POSITIONS_FILES = "filtered_points_of_force_on_boundary.txt"
    finger_positions = np.loadtxt(FINGER_POSITIONS_FILES, skiprows=1)
    simulation_indices = np.array(list(range(len(simulations_finger_index))))  # idk if needed

    BUNNY_CENTER_POSITION = compute_bunny_head_center(vertices_tensor[0], neck_height=0, head_top_height=1)
    # neck_height and head_top height must be found

    for sim_i, finger_i in enumerate(simulation_indices):
        finger_pos = finger_positions[finger_i]
        force_direction_also_named_v = BUNNY_CENTER_POSITION - finger_pos
        FORCE_DIRECTIONS[sim_i] = force_direction_also_named_v / np.linalg.norm(force_direction_also_named_v)
        # ^^ Maybe also take the normal direction to the surface. (negate it). Then do mean of both. Idk.
        # ./closest_points_and_normals.txt has the normals. But filtered points have different indices by definition.
        # so modifying ./3_filter_points.py so it also creates a filtered_normals_on_boundary.txt would be a good idea

    """ Aiming to do FORCE_DIRECTIONS = load from a file calculated in ./3_filter_points.py """

    """ Start of temporary thing to replace """

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

        MORE_STEPS = False
        if MORE_STEPS:
            training_context.scheduler = CustomLRScheduler(training_context.optimizer, factor=0.8, patience=EPOCH_PATIENCE)
            pow = 0
            for i in range(3 * pow):
                training_context.scheduler.step(i)

        if reset:
            training_context.reset(mesh_encoder_lr=melr, sdf_calculator_lr=sclr)

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
            reset=reset,
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


# ------------------------------------------- Wrapper to main function. No need to read, can treat as black box.
# ------ Send arguments from sys to main
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
    parser.add_argument(
        "--reset",
        action="store_true",
        help="reset the optimizer",
    )
    # Add --melr and --sclr arguments, only valid if --reset is given
    parser.add_argument(
        "--melr",
        type=float,
        help="Mesh encoder learning rate (requires --reset)",
    )

    parser.add_argument(
        "--sclr",
        type=float,
        help="SDF calculator learning rate (requires --reset)",
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

    # Validate argument dependencies
    if args.reset and not args.continue_training:
        parser.error("--reset requires --continue_training to be specified.")

    if (args.melr or args.sclr) and not args.reset:
        parser.error("--melr and --sclr require --reset to be specified.")

    melr, sclr = args.melr, args.sclr
    if args.melr == None:
        melr = START_ENCODER_LR
    if args.sclr == None:
        sclr = START_SDF_CALCULATOR_LR

    # Call main and exit with the returned code
    ret = main(args.start_from_zero, args.continue_training, epoch_index, time_index, finger_index, args.reset, melr, sclr)
    """ Bellow is to:  Returns 0 if succeed, 1 if mesh_encoder_lr=0, 2 if sdf_calculator_lr = 0, 3 if both is"""
    """ Case 3 should not happen because its caught one cycle element in advance by 1 and 2. 1 and 2 is if we have 0 x and we save previous y to 0, well have prev_x = prev_y = 0"""
    """ where x and y are the sdf_calculator_lr and mesh_encoder_lr"""
    exit_map = [1, 2, 0, 3]
    # Ensure ret is within range
    if ret >= 0 and ret < len(exit_map):
        exit(exit_map[ret])  # Exit using the mapped value
    else:
        exit(ret)  # Exit using the original value if out of range
