import numpy as np
import polyscope as ps

# Initialize Polyscope
ps.init()

# Define the vertices of the pyramid (Z-up)
vertices = np.array(
    [
        [-0.5, -0.5, 0],  # Base 1
        [0.5, -0.5, 0],  # Base 2
        [0.5, 0.5, 0],  # Base 3
        [-0.5, 0.5, 0],  # Base 4
        [0.0, 0.0, 4.0],  # Top (apex)
    ]
)

# Define the triangular faces
faces = np.array(
    [
        [0, 1, 4],  # Side 1
        [1, 2, 4],  # Side 2
        [2, 3, 4],  # Side 3
        [3, 0, 4],  # Side 4
        [0, 1, 2],  # Base triangle 1
        [2, 3, 0],  # Base triangle 2
    ]
)

# Set Z as the up direction
ps.set_up_dir("z_up")

# Register the pyramid as a surface mesh
ps.register_surface_mesh("Pyramid", vertices, faces)

# Show the visualization
ps.show()
