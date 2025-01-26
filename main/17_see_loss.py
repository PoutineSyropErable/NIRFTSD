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

import matplotlib.pyplot as plt
from enum import Enum
from __TRAINING_FILE import MeshEncoder, SDFCalculator, TrainingContext, LATENT_DIM, DEFAULT_FINGER_INDEX, NEURAL_WEIGHTS_DIR, LOAD_DIR
from __TRAINING_FILE import CustomLRScheduler


def read_pickle(directory, filename, finger_index, validate=False):
    long_file_name = f"{directory}/{filename}_{finger_index}{'_validate' if validate else ''}.pkl"

    with open(long_file_name, "rb") as file:
        output = pickle.load(file)
        print(f"Loaded {type(output)} from {long_file_name}")

    return output


def main():
    finger_index = 730
    epoch_index = 592
    time_index = 0
    vertices_tensor_np = read_pickle(LOAD_DIR, "vertices_tensor", finger_index)[:-1]
    faces = read_pickle(LOAD_DIR, "vertices_tensor", finger_index)[:-1]
    sdf_points = read_pickle(LOAD_DIR, "sdf_points", finger_index)[:-1]
    sdf_values = read_pickle(LOAD_DIR, "sdf_values", finger_index)[:-1]

    # Convert inputs to PyTorch tensors
    vertices_tensor = torch.tensor(vertices_tensor_np, dtype=torch.float32)  # (time_steps, num_vertices, 3)
    sdf_points = torch.tensor(sdf_points, dtype=torch.float32)  # (time_steps, num_points, 3)
    sdf_values = torch.tensor(sdf_values, dtype=torch.float32).unsqueeze(-1)  # (time_steps, num_points, 1)
    input_dim = vertices_tensor.shape[1] * vertices_tensor.shape[2]  # num_vertices * 3

    number_of_shape_per_familly = sdf_points.shape[0]
    mesh_encoder = MeshEncoder(input_dim=input_dim, latent_dim=LATENT_DIM)
    sdf_calculator = SDFCalculator(latent_dim=LATENT_DIM)
    training_context = TrainingContext(mesh_encoder, sdf_calculator, finger_index, number_of_shape_per_familly, 0.1, 0.01)
    training_context.load_model_weights(epoch_index, time_index)

    losses = training_context.loss_tracker_validate
    loss_mean = [np.mean(loss) for loss in losses]
    loss_max = [np.max(loss) for loss in losses]

    loss_mean = loss_mean[:-1]
    loss_max = loss_max[:-1]

    plt.figure()
    plt.title("mean and max loss over epochs")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(loss_mean[5:], label="mean")
    plt.plot(loss_max[5:], label="max")
    plt.legend()
    plt.show()

    pass


main()
