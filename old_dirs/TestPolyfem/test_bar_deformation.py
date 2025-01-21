import polyfempy as pf
import numpy as np
import polyscope as ps
import os

# --- Clean up Polyscope configuration files ---
config_files = [".polyscope.ini", "imgui.ini"]
for file in config_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed {file}")

# Path to the mesh file
mesh_path = "bar.mesh"

# 1. Create settings
settings = pf.Settings(
    discr_order=1,  # Linear elements
    pde=pf.PDEs.LinearElasticity  # Linear elasticity model
)

# Set material properties (Young's modulus and Poisson's ratio)
settings.set_material_params("E", 21000)
settings.set_material_params("nu", 0.3)

# 2. Define the problem
problem = pf.Problem()

# Boundary conditions
problem.set_x_symmetric(1)  # Symmetric boundary at sideset 1 in x
problem.set_y_symmetric(4)  # Symmetric boundary at sideset 4 in y
problem.set_force(3, [10000, 0,0])  # Apply force [100, 0] at sideset 3

# Link the problem with the settings
settings.problem = problem

# 3. Solve the problem
solver = pf.Solver()
solver.settings(settings)
solver.load_mesh_from_path(mesh_path, normalize_mesh=True)  # Normalize to [0,1]^2

# Solve the PDE
solver.solve()

# 4. Extract the solution
pts, tets, disp = solver.get_sampled_solution()
print("Displacement Debug Info:")
print(f"Displacement shape: {disp.shape}")
print(f"Max displacement: {np.max(disp, axis=0)}")
print(f"Min displacement: {np.min(disp, axis=0)}")

# Displace the mesh
vertices = pts + disp

# Get stresses (Mises stress)
mises, _ = solver.get_sampled_mises_avg()

# 5. Visualize with Polyscope
ps.init()  # Initialize Polyscope

# Register the original mesh and the displaced mesh
ps.register_surface_mesh("Original Mesh", pts, tets)
#ps.register_surface_mesh("Displaced Mesh", vertices, tets)

# Add stress and displacement visualization
#ps.get_volume_mesh("Displaced Mesh").add_scalar_quantity("Mises Stress", mises, defined_on="cells")
#ps.get_volume_mesh("Displaced Mesh").add_vector_quantity("Displacement", disp, defined_on="nodes")

# Show the visualization
ps.show()
