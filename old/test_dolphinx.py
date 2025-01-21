import numpy as np
import meshio
from mpi4py import MPI
from dolfinx import mesh, fem, io
import ufl
from petsc4py import PETSc
import pyvista


def get_tetra_mesh_data(file_path):
    """
    Extracts points and tetrahedral connectivity from a mesh file.

    Args:
        file_path (str): Path to the input mesh file.

    Returns:
        tuple: A tuple (points, connectivity), where:
            - points: numpy.ndarray of shape (N, 3), the mesh vertex coordinates.
            - connectivity: numpy.ndarray of shape (M, 4), the tetrahedral cell indices.
    """
    # Load the mesh file
    mesh_data = meshio.read(file_path)

    # Extract points
    points = mesh_data.points

    # Extract tetrahedral cells
    connectivity = mesh_data.cells_dict.get("tetra", None)
    if connectivity is None:
        raise ValueError("No tetrahedral cells found in the mesh file.")

    return points, connectivity


def apply_random_force_dolphinx(mesh_file, force_vector, fixed_point_index):
    """
    Apply random force at a specific point and deform the mesh.

    Args:
        mesh_file (str): Path to the input tetrahedral mesh file.
        force_vector (list): Force vector to apply (e.g., [0, 0, -1]).
        fixed_point_index (int): Index of the vertex to fix (Dirichlet BC).

    Returns:
        None. Outputs a visualization of the deformed mesh.
    """
    # --- 1. Read Mesh Data ---
    points, connectivity = get_tetra_mesh_data(mesh_file)
    domain = mesh.create_mesh(MPI.COMM_WORLD, connectivity, points, ufl_domain="tetrahedron")

    # --- 2. Define Function Space ---
    V = fem.VectorFunctionSpace(domain, ("Lagrange", 1))

    # --- 3. Boundary Conditions ---
    def fixed_boundary(x):
        return np.allclose(x.T, points[fixed_point_index], atol=1e-6)

    u_bc = fem.Constant(domain, PETSc.ScalarType((0, 0, 0)))
    bcs = [fem.dirichletbc(u_bc, fem.locate_dofs_geometrical(V, fixed_boundary), V)]

    # --- 4. Force Application ---
    ds = ufl.Measure("ds", domain=domain)
    f = fem.Constant(domain, PETSc.ScalarType(force_vector))

    # --- 5. Material Parameters ---
    E, nu = 1e4, 0.3
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u))

    # --- 6. Variational Problem ---
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(f, v) * ds

    # --- 7. Solve Problem ---
    u_sol = fem.Function(V)
    problem = fem.petsc.LinearProblem(a, L, bcs=bcs)
    u_sol = problem.solve()

    # --- 8. Visualize the Result ---
    with io.XDMFFile(MPI.COMM_WORLD, "deformed_mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(u_sol)

    # Plot results using pyvista
    pyvista.start_xvfb()
    grid = pyvista.read("deformed_mesh.xdmf")
    warped = grid.warp_by_vector("u", factor=1)
    warped.plot(show_edges=True)


# Run the function
if __name__ == "__main__":
    mesh_file = "Bunny/bunny.mesh"  # Replace with your mesh file path
    force_vector = [0, 0, -1]  # Example random force vector
    fixed_point_index = 0  # Fix the first vertex
    apply_random_force_dolphinx(mesh_file, force_vector, fixed_point_index)
