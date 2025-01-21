import polyfempy as pf
import meshio
import numpy as np

# Initialize solver
solver = pf.Solver()

# Load the input mesh
mesh_path = "bar.mesh"  # Input .mesh file
solver.load_mesh_from_path(mesh_path, normalize_mesh=True)

# Set solver settings
settings = pf.Settings()
settings.set_pde(pf.PDEs.LinearElasticity)
settings.set_material_params("E", 200.0)   # Young's modulus
settings.set_material_params("nu", 0.3)    # Poisson's ratio

# Define boundary conditions
problem = pf.Problem()
problem.add_dirichlet_value(id=1, value=[0.0, 0.0, 0.0])  # Fix one end
problem.add_neumann_value(id=2, value=[0.0, -10.0, 0.0])  # Apply force

# Solve the problem
settings.set_problem(problem)
solver.set_settings(settings)
solver.solve()

print("\n\nSolved the problem\n\n")

# Get the vertices and connectivity of the mesh
solution_vertices = solver.get_sampled_solution(boundary_only=False)[0]
mesh_connectivity = solver.get_solution()

print("\n\ngot to the solution\n\n")

# Export the mesh using Meshio
output_mesh = "bar_bent.mesh"

# Create and write a Meshio mesh
meshio.write_points_cells(
    output_mesh,
    points=solution_vertices,
    cells=[("tetra", mesh_connectivity)]
)

print(f"Exported the resulting mesh to: {output_mesh}")
