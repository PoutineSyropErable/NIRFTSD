import subprocess
import numpy as np
import os
import argparse

# Fixed paths
FILTERED_POINTS_FILE = "filtered_points_of_force_on_boundary.txt"
SIMULATION_SCRIPT = "./4_linear_elasticity_finger_pressure_bunny_dynamic.py"
OUTPUT_DIR = "./deformed_bunny_files"

# Set up argument parsing
parser = argparse.ArgumentParser(description="Run simulations for deformation scenarios.")
parser.add_argument("--starting_index", type=int, default=270, help="Starting index for the simulation (default: 3)")
parser.add_argument("--stopping_index", type=int, default=-1, help="Stopping index for the simulation (default: 3 for all)")
args = parser.parse_args()

# Assign arguments to variables
STARTING_INDEX = args.starting_index
STOPPING_INDEX = args.stopping_index

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check if the filtered points file exists
if not os.path.exists(FILTERED_POINTS_FILE):
    raise FileNotFoundError(f"Filtered points file not found: {FILTERED_POINTS_FILE}")

# Load the filtered points
filtered_points = np.loadtxt(FILTERED_POINTS_FILE, skiprows=1)

# Adjust stopping index if -1
if STOPPING_INDEX == -1:
    STOPPING_INDEX = len(filtered_points) - 1

# Validate indices
if STARTING_INDEX < 0 or STARTING_INDEX >= len(filtered_points):
    raise ValueError(f"STARTING_INDEX ({STARTING_INDEX}) is out of range.")
if STOPPING_INDEX < 0 or STOPPING_INDEX >= len(filtered_points):
    raise ValueError(f"STOPPING_INDEX ({STOPPING_INDEX}) is out of range.")
if STOPPING_INDEX < STARTING_INDEX:
    raise ValueError(f"STOPPING_INDEX ({STOPPING_INDEX}) cannot be less than STARTING_INDEX ({STARTING_INDEX}).")

print(f"Processing range: STARTING_INDEX={STARTING_INDEX}, STOPPING_INDEX={STOPPING_INDEX}")

# Iterate over each row and run the simulation
for index, finger_position in enumerate(filtered_points[STARTING_INDEX : STOPPING_INDEX + 1], start=STARTING_INDEX):
    # Convert finger position to string for passing as arguments
    finger_position_str = " ".join(map(str, finger_position))

    print(f"Running simulation for index: {index}, finger_position: {finger_position_str}")

    # Run the simulation script with subprocess
    command = ["python", SIMULATION_SCRIPT, "--index", str(index), "--finger_position", *map(str, finger_position)]
    try:
        result = subprocess.run(command, check=True, text=False, capture_output=False)
        print(f"Simulation {index} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Simulation {index} failed with error: {e}")
        continue  # Skip to the next simulation

print("All simulations completed.")
