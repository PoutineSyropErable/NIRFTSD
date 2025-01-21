import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Directories
INPUT_DIR = "./displacement_norms"
OUTPUT_DIR = "./plots"

# Argument parsing
parser = argparse.ArgumentParser(description="Generate plots for maximum deformation data.")
parser.add_argument(
    "--replace", 
    action="store_true", 
    help="Replace existing plots if they already exist."
)
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all files in the input directory
files = [f for f in os.listdir(INPUT_DIR) if f.startswith("max_displacement_array_") and f.endswith(".txt")]

if not files:
    print(f"No files found in {INPUT_DIR} with the pattern 'max_displacement_array_<index>.txt'.")
    exit()

for file in files:
    # Extract index from filename
    try:
        index = int(file.split("_")[-1].split(".")[0])
    except ValueError:
        print(f"Skipping file {file}: Unable to extract index.")
        continue

    # Output file path
    output_file = os.path.join(OUTPUT_DIR, f"maximum_deformation_plot_{index}.jpg")

    # Check if the plot already exists
    if os.path.exists(output_file) and not args.replace:
        print(f"Plot already exists and will not be replaced: {output_file}")
        continue

    # Load data
    file_path = os.path.join(INPUT_DIR, file)
    try:
        data = np.loadtxt(file_path)
    except Exception as e:
        print(f"Error reading file {file}: {e}")
        continue

    # Plot the data
    plt.figure()
    plt.plot(data, label="Maximum Deformation", color="blue")
    plt.xlabel("Time Steps")
    plt.ylabel("Maximum Deformation")
    plt.title(f"Maximum Deformation Over Time (Index {index})")
    plt.grid(True)
    plt.legend()

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved: {output_file}")

print("All plots created.")

