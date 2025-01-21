import subprocess
import argparse
import time
import os
import h5py
import numpy as np

# File containing filtered points
FILTERED_POINTS_FILE = "filtered_points_of_force_on_boundary.txt"
BUGGY_INDICES_FILE = "filtered_buggy_deformed_indices.txt"
VALID_INDICES_FILE = "filtered_not_buggy_deformed_indices.txt"
OUTPUT_DIR = "./calculated_sdf"
# These are filtered using (is max deformation within a set range)

def load_displacement_data(h5_file):
    """Load displacement data from an HDF5 file with a lock check."""
    while True:
        try:
            with h5py.File(h5_file, "r") as f:
                displacements = f["displacements"][:]
                print(f"Loaded displacement data with shape: {displacements.shape}")
            return displacements
        except OSError as e:
            print(f"File {h5_file} is locked. Retrying in 10 seconds...")
            time.sleep(10)

def validate_indices(starting_index, stopping_index, indices_to_traverse):
    """Ensure starting and stopping indices are valid."""
    num_points = len(indices_to_traverse)
    if starting_index < 0 or starting_index >= num_points:
        raise ValueError(f"Starting index {starting_index} is out of range (0, {num_points - 1}).")
    if stopping_index != -1 and (stopping_index < 0 or stopping_index >= num_points):
        raise ValueError(f"Stopping index {stopping_index} is out of range (0, {num_points - 1}).")
    if stopping_index != -1 and stopping_index < starting_index:
        raise ValueError(f"Stopping index {stopping_index} cannot be less than starting index {starting_index}.")

def main():
    parser = argparse.ArgumentParser(description="Iterate SDF calculation over a range of indices.")
    parser.add_argument("--starting_index", type=int, required=True, help="Starting index (inclusive).")
    parser.add_argument("--stopping_index", type=int, required=True, help="Stopping index (inclusive). Use -1 to process till the end.")
    parser.add_argument("--sdf_only", action="store_true", help="Only save SDF values.")
    parser.add_argument(
        "--filter",
        choices=["all", "buggy", "valid"],
        default="all",
        help="Filter indices to process: 'all' for no filtering, 'buggy' for buggy indices, 'valid' for valid indices.",
    )
    parser.add_argument(
        "--doall",
        action="store_true",
        help="If set, process all indices regardless of filter selection. Overrides --filter.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing output files. If not provided, existing files will be skipped.",
    )
    args = parser.parse_args()

    # Check for incompatible arguments
    if args.doall and args.filter != "all":
        raise ValueError("--doall cannot be used with a specific --filter option. Use --filter 'all' or omit it.")

    print(f"Arguments:")
    print(f"  Starting Index: {args.starting_index}")
    print(f"  Stopping Index: {args.stopping_index}")
    print(f"  SDF Only: {'Yes' if args.sdf_only else 'No'}")
    print(f"  Filter: {args.filter}")
    print(f"  Do All: {'Yes' if args.doall else 'No'}")
    print(f"  Replace: {'Yes' if args.replace else 'No'}")
    print("\n\n")

    # Check if the filtered points file exists
    if not os.path.exists(FILTERED_POINTS_FILE):
        raise FileNotFoundError(f"Filtered points file not found: {FILTERED_POINTS_FILE}")

    # Load the filtered points
    filtered_points = np.loadtxt(FILTERED_POINTS_FILE, skiprows=1)
    print(f"Loaded {len(filtered_points)} filtered points.")

    # Determine indices to traverse
    if args.doall:
        indices_to_traverse = np.arange(len(filtered_points))
    else:
        if not (os.path.exists(BUGGY_INDICES_FILE) and os.path.exists(VALID_INDICES_FILE)):
            raise FileNotFoundError(f"One or both filter files are missing: {BUGGY_INDICES_FILE}, {VALID_INDICES_FILE}")

        buggy_indices = np.loadtxt(BUGGY_INDICES_FILE, skiprows=1, dtype=int)
        valid_indices = np.loadtxt(VALID_INDICES_FILE, skiprows=1, dtype=int)

        if args.filter == "all":
            indices_to_traverse = np.sort(np.unique(np.concatenate((buggy_indices, valid_indices))))
        elif args.filter == "buggy":
            indices_to_traverse = buggy_indices
        elif args.filter == "valid":
            indices_to_traverse = valid_indices

    if len(indices_to_traverse) == 0:
        print("No indices to traverse. Exiting.")
        return

    # Validate indices
    validate_indices(args.starting_index, args.stopping_index, indices_to_traverse)

    # Adjust stopping_index if -1
    stopping_index = len(indices_to_traverse) - 1 if args.stopping_index == -1 else args.stopping_index

    print(f"Processing indices from {args.starting_index} to {stopping_index} (inclusive).")
    print(f"SDF only mode: {'Enabled' if args.sdf_only else 'Disabled'}")

    # Iterate through the specified index range
    for current_index in indices_to_traverse[args.starting_index : stopping_index + 1]:
        # Determine output file path
        output_file = os.path.join(
            OUTPUT_DIR,
            f"sdf_points_{current_index}{'_sdf_only' if args.sdf_only else ''}.h5",
        )

        # Skip if the file exists and --replace is not set
        if os.path.exists(output_file) and not args.replace:
            print(f"Skipping index {current_index}: File {output_file} already exists.")
            continue

        print(f"Processing index: {current_index}")

        # Build the command for running `10_calculate_sdfs.py`
        command = [
            "python", "10_calculate_sdfs.py",
            "--index", str(current_index),
        ]
        if args.sdf_only:
            command.append("--sdf_only")

        try:
            # Run the command
            subprocess.run(command, check=True)
            print(f"Index {current_index} processed successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"Error processing index {current_index}: {e}\n")

    print("SDF calculation iteration completed.")

if __name__ == "__main__":
    main()

