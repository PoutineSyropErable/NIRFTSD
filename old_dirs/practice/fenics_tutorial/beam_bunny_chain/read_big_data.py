import h5py

def read_and_display_h5(filename):
    """
    Read an HDF5 file and display the first 5 time steps and points, along with dimensions.

    Parameters:
    - filename (str): Path to the HDF5 file to read.
    """
    with h5py.File(filename, "r") as h5file:
        # Check available datasets
        if "points" not in h5file:
            print("Dataset 'points' not found in the file.")
            return

        points = h5file["points"]

        # Get the dimensions of the dataset
        dimensions = points.shape
        print(f"Dataset dimensions (time steps, points, coordinates): {dimensions}\n")

        # Display the first 5 time steps and points
        print("First 5 time steps and first 5 points:")
        for t in range(min(5, dimensions[0])):
            print(f"Time step {t}:\n{points[t, :5]}\n")

# Specify the file to read
RANDOM = True
if RANDOM:
    filename = "random_points.h5"
else:
    filename = "incremental_points.h5"
read_and_display_h5(filename)

