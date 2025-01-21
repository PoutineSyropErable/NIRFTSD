import os
import numpy as np

# Directories
INPUT_DIR = "./displacement_norms"
CUT_OFF = 0.05  # Replace with your desired cutoff value

# Get all files in the input directory
files = [f for f in os.listdir(INPUT_DIR) if f.startswith("max_displacement_array_") and f.endswith(".txt")]

if not files:
    print(f"No files found in {INPUT_DIR} with the pattern 'max_displacement_array_<index>.txt'.")
    exit()

# Initialize an array to store the maximum deformation values
max_array = np.zeros(len(files))
indices = np.zeros(len(files), dtype=int)

# Process each file
for i, file in enumerate(files):
    # Extract index from filename
    try:
        indices[i] = int(file.split("_")[-1].split(".")[0])
    except ValueError:
        print(f"Skipping file {file}: Unable to extract index.")
        continue

    # Load data
    file_path = os.path.join(INPUT_DIR, file)
    try:
        data = np.loadtxt(file_path)
    except Exception as e:
        print(f"Error reading file {file}: {e}")
        continue

    # Find maximum deformation and store it
    max_array[i] = np.max(data)

# Filter indices where maximum deformation is below the cutoff
valid_indices = indices[np.where(max_array < CUT_OFF)]
valid_indices = indices[np.where(max_array < CUT_OFF)]  # Indices where max deformation < cutoff
buggy_indices = np.setdiff1d(indices, valid_indices)  # Indices not in valid_indices

# Print the filtered indices
print("Indices with maximum deformation below the cutoff: (Valid/Not Buggy Indices)")
print(valid_indices)

# Print the filtered indices
print("Indices with maximum deformation above the cutoff: (Buggy Indices)")
print(buggy_indices)

# File names
valid_output_file = "filtered_not_buggy_deformed_indices.txt"
buggy_output_file = "filtered_buggy_deformed_indices.txt"

# Save valid indices
np.savetxt(valid_output_file, valid_indices, fmt='%d', header='Filtered Indices Below Cutoff', comments='')

# Save removed indices
np.savetxt(buggy_output_file, buggy_indices, fmt='%d', header='Removed Indices Above or Equal to Cutoff', comments='')

print(f"Filtered indices saved to {valid_output_file}.")
print(f"Removed indices saved to {buggy_output_file}.")
