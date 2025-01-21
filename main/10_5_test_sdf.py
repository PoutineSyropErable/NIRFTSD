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

import pickle

from sklearn.neighbors import KNeighborsRegressor
from functools import partial

GRID_DIM = 30


# Directory containing the pickle files
LOAD_DIR = "./training_data"

# Directory where we save and load the neural weights
NEURAL_WEIGHTS_DIR = "./neural_weights"
DEFAULT_FINGER_INDEX = 730


DEBUG_ = True
DEBUG_TIMER = False
os.chdir(sys.path[0])


def read_pickle(directory, filename, finger_index, validate=False):
    long_file_name = f"{directory}/{filename}_{finger_index}{'_validate' if validate else ''}.pkl"
    print(long_file_name, "\n")

    with open(long_file_name, "rb") as file:
        output = pickle.load(file)
        print(f"Loaded {type(output)} from {long_file_name}")

    return output


def main(finger_index=DEFAULT_FINGER_INDEX):
    vertices_tensor_np = read_pickle(LOAD_DIR, "vertices_tensor", finger_index)
    faces = read_pickle(LOAD_DIR, "vertices_tensor", finger_index)
    sdf_points = read_pickle(LOAD_DIR, "sdf_points", finger_index)
    sdf_values = read_pickle(LOAD_DIR, "sdf_values", finger_index)

    return 0


if __name__ == "__main__":
    main()
