import h5py
import numpy as np

def generate_random_points(filename, num_time_steps, num_points, x_range, y_range, z_range):
    """
    Generate random points for each time step and write to an HDF5 file.
    
    Parameters:
    - filename: str, name of the file to save the data.
    - num_time_steps: int, number of time steps.
    - num_points: int, number of points per time step.
    - x_range: tuple, range for x coordinates (min, max).
    - y_range: tuple, range for y coordinates (min, max).
    - z_range: tuple, range for z coordinates (min, max).
    """
    with h5py.File(filename, 'w') as h5_file:
        # Create dataset for random points
        points_dataset = h5_file.create_dataset(
            "points",
            shape=(num_time_steps, num_points, 3),
            dtype=np.float32,
            chunks=(1, num_points, 3),  # Optimize for writing one time step at a time
        )
        
        for t in range(num_time_steps):
            # Generate random x, y, z for this time step
            random_x = np.random.uniform(x_range[0], x_range[1], num_points)
            random_y = np.random.uniform(y_range[0], y_range[1], num_points)
            random_z = np.random.uniform(z_range[0], z_range[1], num_points)
            
            # Stack x, y, z into points array
            points_t = np.stack([random_x, random_y, random_z], axis=1)
            
            # Write the current time step's points to the dataset
            points_dataset[t, :, :] = points_t

# Example usage
filename = "random_points.h5"
num_time_steps = 1000  # Number of time steps
num_points = 100  # Number of points per time step
x_range = (0, 1)  # Random range for x
y_range = (0, 2)  # Random range for y
z_range = (0, 3)  # Random range for z

generate_random_points(filename, num_time_steps, num_points, x_range, y_range, z_range)
print(f"Random points written to {filename}.")

