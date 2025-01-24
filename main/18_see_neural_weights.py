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


def read_pickle(directory, filename, finger_index, validate=False):
    long_file_name = f"{directory}/{filename}_{finger_index}{'_validate' if validate else ''}.pkl"

    with open(long_file_name, "rb") as file:
        output = pickle.load(file)
        print(f"Loaded {type(output)} from {long_file_name}")

    return output


def main():
    finger_index = 730
    vertices_tensor_np = read_pickle(LOAD_DIR, "vertices_tensor", finger_index)[:-1]
    sdf_points = read_pickle(LOAD_DIR, "sdf_points", finger_index)[:-1]

    # Convert inputs to PyTorch tensors
    vertices_tensor = torch.tensor(vertices_tensor_np, dtype=torch.float32)  # (time_steps, num_vertices, 3)
    input_dim = vertices_tensor.shape[1] * vertices_tensor.shape[2]  # num_vertices * 3

    number_of_shape_per_familly = sdf_points.shape[0]
    mesh_encoder = MeshEncoder(input_dim=input_dim, latent_dim=LATENT_DIM)
    sdf_calculator = SDFCalculator(latent_dim=LATENT_DIM)
    training_context = TrainingContext(mesh_encoder, sdf_calculator, finger_index, number_of_shape_per_familly, 0.1)

    epochs = list(range(2, 303, 10))
    weight_norms = []
    weight_stds = []
    for epoch_index in epochs:
        training_context.load_model_weights(epoch_index, 0)
        weights = list(training_context.mesh_encoder.parameters())
        weight_norm = sum(torch.norm(w).item() for w in weights)  # Compute the norm of all weights
        weight_std = torch.std(torch.cat([w.flatten() for w in weights])).item()
        weight_norms.append(weight_norm)
        weight_stds.append(weight_std)

    print(f"weight_norms = \n{weight_norms}\n")
    print(f"weight_stds = \n{weight_stds}\n")
    # Plot the weight norms as a function of epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, weight_norms, marker="o", label="Weight Norm")
    plt.title("Weight Norms as a Function of Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Weight Norm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, weight_stds, marker="o", label="Weight STDs")
    plt.title("Weight STDs as a Function of Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Weight STDs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return 0


main()
