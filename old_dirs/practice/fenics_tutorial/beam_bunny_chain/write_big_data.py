import h5py
import numpy as np

def generate_and_save_points(filename, num_t=1000, num_points=1000):
    """
    Generate points incrementally and save them to an HDF5 file.

    Parameters:
    - filename: Name of the HDF5 file to write.
    - num_t: Number of time steps.
    - num_points: Number of points per time step.
    """
    # Open HDF5 file in write mode
    with h5py.File(filename, "w") as f:
        # Create a dataset for points
        points_shape = (num_t, num_points, 3)  # Shape: (time, points, x/y/z)
        points_dtype = np.float32
        points_dset = f.create_dataset("points", shape=points_shape, dtype=points_dtype, compression="gzip")

        # Generate and save points incrementally
        for t in range(num_t):
            # Generate points for the current time step
            points_t = np.array([
                [i * 1 + t, i * 2 + t, i * 3 + t] for i in range(num_points)
            ])

            # Write the points to the file at the current time step
            points_dset[t, :, :] = points_t

            # (Optional) Print progress
            if t % 100 == 0:
                print(f"Generated and saved points for time step {t}/{num_t}")

# Example usage
filename = "incremental_points.h5"
generate_and_save_points(filename, num_t=1000, num_points=1000)
