import os
import argparse
import numpy as np
import subprocess

# Directories
DISPLACEMENT_DIR = "./deformed_bunny_files_tunned"
ANIMATION_DIR = "./Animations_tunned"
VALID_INDICES_FILE = "filtered_not_buggy_deformed_indices.txt"
BUGGY_INDICES_FILE = "filtered_buggy_deformed_indices.txt"

# Create the animations directory if it doesn't exist
os.makedirs(ANIMATION_DIR, exist_ok=True)

# Command template for creating animations
ANIMATION_SCRIPT = "./6_create_animation.py"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Create animations for deformation scenarios.")
parser.add_argument(
    "--filter", type=str, choices=["valid", "buggy", "all"], default="all", help="Filter option: valid, buggy, or all. Default is all."
)
parser.add_argument("--replace", action="store_true", help="Replace existing animations.")
parser.add_argument("--doall", action="store_true", help="Use all files from the directory regardless of filter.")
args = parser.parse_args()

# Load indices based on the filter
valid_indices = []
buggy_indices = []

if not args.doall:
    if args.filter in ["valid", "all"] and os.path.exists(VALID_INDICES_FILE):
        valid_indices = np.loadtxt(VALID_INDICES_FILE, dtype=int, skiprows=1).tolist()
    if args.filter in ["buggy", "all"] and os.path.exists(BUGGY_INDICES_FILE):
        buggy_indices = np.loadtxt(BUGGY_INDICES_FILE, dtype=int, skiprows=1).tolist()

    if args.filter == "valid":
        indices_to_process = valid_indices
    elif args.filter == "buggy":
        indices_to_process = buggy_indices
    else:  # args.filter == "all"
        indices_to_process = sorted(set(valid_indices + buggy_indices))
else:
    # Use all files from the directory
    all_files = [f for f in os.listdir(DISPLACEMENT_DIR) if f.startswith("displacement_") and f.endswith(".h5")]
    indices_to_process = sorted(int(f.split("_")[1].split(".")[0]) for f in all_files)

# Print the final list of indices
print(f"Indices to process: {indices_to_process}")

if not indices_to_process:
    print(f"No indices to process for filter: {args.filter}.")
    exit()

# Iterate over indices and create animations
for index in indices_to_process:
    animation_file = os.path.join(ANIMATION_DIR, f"bunny_deformation_animation_{index}.mp4")

    # Skip if animation already exists and --replace is not specified
    if not args.replace and os.path.exists(animation_file):
        print(f"Skipping index {index}: Animation already exists.")
        continue

    # Command to generate the animation
    command = ["python", ANIMATION_SCRIPT, "--index", str(index), "--offscreen"]
    print(f"Creating animation for index {index}...")

    try:
        subprocess.run(command, check=True)
        print(f"Animation for index {index} created successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create animation for index {index}: {e}\n")
        continue

print("All animations processed.")
