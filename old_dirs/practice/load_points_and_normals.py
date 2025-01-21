import numpy as np
# Load the data back
loaded_data = np.loadtxt("closest_points_and_normals.txt")

# Split the data into `closest_points` and `normals`
loaded_closest_points = loaded_data[:, :3]  # First 3 columns
loaded_normals = loaded_data[:, 3:]         # Last 3 columns

print("\n\n-----Start of programs------\n\n")


print(f"Shape(normals)        = {np.shape(loaded_normals)}")
print(f"Shape(closest_points) = {np.shape(loaded_closest_points)}\n")

print("Loaded closest points:")
print(loaded_closest_points)

print("\nLoaded normals:")
print(loaded_normals)




