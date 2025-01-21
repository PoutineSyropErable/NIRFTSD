# Scaled variable
import pyvista as pv
from dolfinx import mesh, fem, plot, io, default_scalar_type, nls, log
from dolfinx.io.utils import XDMFFile
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem.petsc import (
    assemble_vector,
    assemble_matrix,
    create_vector,
    apply_lifting,
    set_bc,
)
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import h5py
import os, sys
import argparse
from typing import Tuple

DEBUG_ = False
# --------------------------------------------------------HELPER FUNCTIONS-----------------------------------------


def get_finger_position(index: int) -> np.ndarray:
    """
    Retrieve the finger position based on the given index.

    Parameters:
        index (int): The index for the deformation scenario.

    Returns:
        np.ndarray: The finger position as a NumPy array.
    """
    FILTERED_POINTS_FILE = "./filtered_points_of_force_on_boundary.txt"
    filtered_points = np.loadtxt(FILTERED_POINTS_FILE, skiprows=1)
    finger_position = filtered_points[index]
    return finger_position


def load_file(filename: str) -> mesh.Mesh:
    # Load the mesh from the XDMF file
    with XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        domain: mesh.Mesh = xdmf.read_mesh(name="Grid")
        print("Mesh loaded successfully!")
    return domain


def get_array_from_conn(conn) -> np.ndarray:
    """
    Convert mesh topology connectivity to a 2D numpy array.

    Parameters:
    - conn: The mesh topology connectivity (dolfinx mesh.topology.connectivity).
    type: dolfinx.cpp.graph.AdjacencyList_int32

    Returns:
    - connectivity_2d: A 2D numpy array where each row contains the vertex indices for a cell.
    """
    connectivity_array = conn.array
    offsets = conn.offsets

    # Convert the flat connectivity array into a list of arrays
    connectivity_2d = [connectivity_array[start:end] for start, end in zip(offsets[:-1], offsets[1:])]

    # Convert to numpy array with dtype=object to handle variable-length rows
    return np.array(connectivity_2d, dtype=object)


def get_mesh(filename: str) -> Tuple[mesh.Mesh, np.ndarray, np.ndarray]:
    domain = load_file(filename)
    # Extract points and cells from the mesh
    points = domain.geometry.x  # Array of vertex coordinates
    conn = domain.topology.connectivity(3, 0)
    connectivity = get_array_from_conn(conn)  # 2D numpy array of cell connectivity
    connectivity = connectivity.astype(np.int64)

    return domain, points, connectivity


# -----------------------------------------------------    CODE STARTS --------------------------------------------


def compute_estimate_dt_courant_limit(points: np.ndarray, connectivity: np.ndarray, E: float, rho: float, nu: float) -> float:
    # Compute wave speed
    c = np.sqrt(E / rho / (1 - nu**2))

    # Find the smallest edge length in the mesh
    min_edge_length = float("inf")
    for tetra in connectivity:
        vertices = points[tetra]
        for i in range(4):
            for j in range(i + 1, 4):
                edge_length = np.linalg.norm(vertices[i] - vertices[j])
                if edge_length < min_edge_length:
                    min_edge_length = edge_length

    # Compute stable time step
    print(f"type(c) = {type(c)}")
    print(f"Speed of sound in the Material    (c) = {c}")
    print(f"minimum edge length (min_edge_length) = {min_edge_length}")
    dt = min_edge_length / c
    return dt


def compute_bounding_box(points: np.ndarray, connectivity: np.ndarray) -> int:
    # Get the min and max for each column
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    pos_min = np.array([x_min, y_min, z_min])
    pos_max = np.array([x_max, y_max, z_max])

    center = (pos_min + pos_max) / 2
    bbox_size = pos_max - pos_min

    # Calculate volume
    # PyVista requires a `cells` array where the first value is the number of nodes per cell
    num_cells = connectivity.shape[0]
    print(f"\nnumber_cells = {num_cells}")
    num_nodes_per_cell = connectivity.shape[1]
    print(f"num_nodes_per_cell = {num_nodes_per_cell}\n")
    cells = np.hstack([np.full((num_cells, 1), num_nodes_per_cell), connectivity]).flatten().astype(np.int32)

    # Check the shape and contents of the cells array
    print(f"Shape of cells array: {cells.shape}")
    print(f"First 20 elements of cells array: {cells[:20]}\n")

    # Cell types: 10 corresponds to tetrahedrons in PyVista
    cell_type = np.full(num_cells, 10, dtype=np.uint8)

    # Create the PyVista UnstructuredGrid
    tetra_grid = pv.UnstructuredGrid(cells, cell_type, points)
    # Calculate the volume
    volume = tetra_grid.volume

    estimate_avg_cell_size = np.cbrt(volume / num_cells)
    print(f"estimate_avg_cell_size = {estimate_avg_cell_size}\n")

    print(f"center = {center}")
    print(f"bbox_size = {bbox_size}\n")
    print(f"Volume of tetrahedral mesh: {volume}")
    bbox_volume = bbox_size[0] * bbox_size[1] * bbox_size[2]
    print(f"Volume of Bbox: {bbox_volume}")
    print(f"ratio: {volume/bbox_volume}")

    print("\n")

    return 0


class PhysicalDeformationSimulation:
    def __init__(
        self,
        domain: mesh.Mesh,
        static_on_boundary_function,
        sigma_function,
        epsilon_function,
    ):
        self.sigma = sigma_function
        self.epsilon = epsilon_function

        self.domain = domain
        self.V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

        tdim = domain.topology.dim  # 3D
        fdim = tdim - 1  # 2D
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, static_on_boundary_function)
        boundary_dofs = fem.locate_dofs_topological(self.V, fdim, boundary_facets)
        u_D = np.array([0, 0, 0], dtype=default_scalar_type)  # no displacement on boundary
        self.bc = fem.dirichletbc(u_D, boundary_dofs, self.V)  # create a condition where u = 0 on z = 0

        self.v = ufl.TestFunction(self.V)
        self.u_t = ufl.TrialFunction(self.V)

        self.u_n = fem.Function(self.V)  # u_n

        self.u_tn = fem.Function(self.V)  # du/dt (n)

        # set u_n, a_tn to 0
        # Initialize displacement and velocity to zero
        self.u_n.interpolate(lambda x: np.zeros_like(x))
        self.u_tn.interpolate(lambda x: np.zeros_like(x))

    def set_time_param(self, t, T, dt, num_steps, dt_max):
        # Time parameters
        self.t = t
        self.T = T
        self.dt = dt
        self.num_steps = num_steps
        self.dt_max = dt_max

        if self.dt > self.dt_max:
            print("Time step too big, it probably will not converge")
            exit(1)

        # Initialize arrays to store time and maximum displacement
        self.t_array = np.linspace(0, T, num_steps)
        self.max_displacement_array = np.zeros(num_steps)

    def set_write_param(self, NUMBER_OF_STEPS_BETWEEN_WRITES: int, OUTPUT_FILE: str):
        self.NUMBER_OF_STEPS_BETWEEN_WRITES = NUMBER_OF_STEPS_BETWEEN_WRITES
        self.NUMBER_OF_WRITES = (self.num_steps + NUMBER_OF_STEPS_BETWEEN_WRITES - 1) // NUMBER_OF_STEPS_BETWEEN_WRITES
        print(f"NUMBER OF WRITES: {self.NUMBER_OF_WRITES}\n")
        self.write_index = 0

        self.OUTPUT_FILE = OUTPUT_FILE
        print(f"output file = {self.OUTPUT_FILE}")
        self.xdmf = io.XDMFFile(self.domain.comm, OUTPUT_FILE, "w")
        self.xdmf.write_mesh(self.domain)

    def calculate_constant_fem(self, rho: float):
        # Body force
        self.f = fem.Constant(self.domain, PETSc.ScalarType((0, 0, 0)))
        self.rho = rho

        # Create the solver
        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)

        # Create the matrix and vector
        a = (self.rho) * ufl.inner(self.u_t, self.v) * ufl.dx
        bilinear_form = fem.form(a)
        A = assemble_matrix(bilinear_form, bcs=[self.bc])
        A.assemble()
        self.solver.setOperators(A)

    def set_finger_position(self, finger_position: np.ndarray, R: float, pressure: float):
        self.finger_position = finger_position
        self.R = R
        p = fem.Constant(self.domain, PETSc.ScalarType(-pressure))  # Negative for compression

        # Get the point on the boundary who will be under pressure. (Near the "Finger")
        x = ufl.SpatialCoordinate(self.domain)
        distance = ufl.sqrt((x[0] - finger_position[0]) ** 2 + (x[1] - finger_position[1]) ** 2 + (x[2] - finger_position[2]) ** 2)
        is_under_pressure = ufl.conditional(ufl.lt(distance, R), 1.0, 0.0)  # If touching finger

        # Traction term
        self.traction_term = is_under_pressure * p * ufl.dot(self.v, ufl.FacetNormal(self.domain)) * ufl.ds
        self.L_CONSTANT = self.dt * ufl.inner(self.f, self.v) * ufl.dx + self.dt * self.traction_term

    def simulate_one_itteration(self, step):
        self.t += self.dt

        # Linear form
        L_changing = (
            self.rho * ufl.inner(self.u_tn, self.v) * ufl.dx - self.dt * ufl.inner(self.sigma(self.u_n), self.epsilon(self.v)) * ufl.dx
        )
        L = self.L_CONSTANT + L_changing

        linear_form = fem.form(L)
        b = create_vector(linear_form)  # L(v)

        # Assemble RHS vector
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, linear_form)

        # Solve for velocity update (u_tn1)
        self.solver.solve(b, self.u_tn.x.petsc_vec)
        self.u_tn.x.scatter_forward()

        # Update displacement
        self.u_n.x.array[:] += self.dt * self.u_tn.x.array
        # Reshape the displacement array to match the (num_points, 3) structure

        displacements = self.u_n.x.array.reshape(self.domain.geometry.x.shape)
        return displacements

    def simulate(self):
        self.xdmf.write_function(self.u_n, self.t)
        for step in range(self.num_steps):
            displacements = self.simulate_one_itteration(step)

            # Print displacement information for debugging
            displacement_norms = np.linalg.norm(displacements, axis=1)
            max_norm = np.max(displacement_norms)
            max_index = np.argmax(displacement_norms)
            point_with_max_displacement = self.domain.geometry.x[max_index]
            self.max_displacement_array[step] = max_norm

            if max_norm > 1000:
                print("The maximal norm of the displacement was way too big, exiting")
                print(f"max_norm = {max_norm}")
                return 1

            if self.write_index >= self.NUMBER_OF_WRITES:
                print(f"Warning: write_index ({self.write_index}) exceeds NUMBER_OF_WRITES ({self.NUMBER_OF_WRITES})!")
                return 2

            if step == self.num_steps - 1 or (step % self.NUMBER_OF_STEPS_BETWEEN_WRITES) == 0:
                print("-----------------------------------------------")
                print(f"Time step {step+1}/{self.num_steps}, t = {self.t}")

                self.xdmf.write_function(self.u_n, self.t)
                print(f"displacements[0:5] = \n{displacements[0:5]}\n")
                print(f"Maximum displacement norm: {max_norm}")
                print(f"Point with maximum displacement: {point_with_max_displacement}")

                ##### This doesnt work for now
                # print(f"Point of finger pressure:        {finger_position}")
                # print(f"Point of finger (inside):        {finger_inside_object}")
                # finger_displacement = get_function_value_at_point(domain, u_n, delaunay, finger_inside_object, cells, estimate_avg_cell_size)
                # print(f"Finger displacement  = {finger_displacement}")

                self.write_index += 1
                print("\n-----------------------------------------------")

        self.xdmf.close()
        return 0

    def write_max_norms(self, DISPLACEMENT_NORMS_DIR):
        print("\n\n-----End of Simulation-----\n\n")  # Close the movie file

        # Save max_displacement_array to a file
        output_file = os.path.join(DISPLACEMENT_NORMS_DIR, f"max_displacement_array_{index}.txt")
        np.savetxt(output_file, self.max_displacement_array, fmt="%.6f")
        print(f"Saved max displacement array to {output_file}")


def main(
    index: int,
    finger_position: np.ndarray,
    OUTPUT_FILE: str,
    DISPLACEMENT_NORMS_DIR: str,
) -> int:
    # Material properties
    E = 5e3  # Young's modulus for rubber in Pascals (Pa)
    nu = 0.40  # Poisson's ratio
    rho = 1  # Density

    # Finger position (center of the sphere) and radius
    R = 0.003  # Radius of the Area where the pressure is applied. Hence F ~ (pi R^2 pressure)
    pressure = 4000

    BUNNY_FILE_NAME = "bunny.xdmf"
    domain, points, connectivity = get_mesh(BUNNY_FILE_NAME)

    if DEBUG_:
        print(f"\ntype(points) = {type(points)}")
        print(f"np.shape(points) = {np.shape(points)}")
        print(f"points = \n{points}\n")

    if DEBUG_:
        print(f"\ntype(connectivity) = {type(connectivity)}")
        print(f"np.shape(connectivity) = {np.shape(connectivity)}")
        print(f"connectivity = \n{connectivity}\n")
        print("connectivity.dtype =", connectivity.dtype)

    compute_bounding_box(points, connectivity)

    # Time parameters
    t = 0.0
    T = 0.01  # Total simulation time
    dt = 1e-6
    num_steps = int(T / dt + 1)
    dt_max = compute_estimate_dt_courant_limit(points, connectivity, E, rho, nu)

    # Lamé parameters
    mu = E / (2 * (1 + nu))
    lambda_ = (E * nu) / ((1 + nu) * (1 - 2 * nu))

    print(f"Young Modulus (E)               =   {E}")
    print(f"Poisson Ration (nu)             =   {nu}")
    print(f"Material Density (rho)          =   {rho}")
    print(f"Pressure Application Radius (R) =   {R}")
    print(f"Pressure                        =   {pressure}")
    print(f"mu                              =   {mu}")
    print(f"lambda                          =   {lambda_}")

    print("")
    print(f"dt={dt}")
    print(f"dt_max = {dt_max}")
    print(f"num_steps = {num_steps}")

    # ------------------------------- FEM ------------------------------

    def epsilon(u):
        return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def sigma(u: fem.Function):
        return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

    Z_GROUND = 0

    def grounded_bunny(x):
        return x[2] <= Z_GROUND

    bunny_simulation = PhysicalDeformationSimulation(domain, grounded_bunny, epsilon, sigma)
    bunny_simulation.set_time_param(t, T, dt, num_steps, dt_max)

    NUMBER_OF_STEPS_BETWEEN_WRITES: int = 100
    bunny_simulation.set_write_param(NUMBER_OF_STEPS_BETWEEN_WRITES, OUTPUT_FILE)
    bunny_simulation.calculate_constant_fem(rho)
    bunny_simulation.set_finger_position(finger_position, R, pressure)
    main_ret = bunny_simulation.simulate()
    bunny_simulation.write_max_norms(DISPLACEMENT_NORMS_DIR)

    return main_ret


if __name__ == "__main__":

    print("\n\n----------------------Start of Program-----------------------\n\n")

    os.chdir(sys.path[0])

    # Directory for displacement norms, so we can plot max(displacement) over time, to find when it reaches equilibrium
    DISPLACEMENT_NORMS_DIR = "./displacement_norms"
    os.makedirs(DISPLACEMENT_NORMS_DIR, exist_ok=True)

    ### Argument parsing
    parser = argparse.ArgumentParser(description="Simulate and save deformation of a bunny mesh.")
    parser.add_argument("--index", type=int, required=True, help="Index of the deformation scenario.")
    parser.add_argument(
        "--finger_position",
        type=float,
        nargs=3,
        required=False,
        help="Finger position as x, y, z.",
    )
    args = parser.parse_args()

    index = args.index
    if args.finger_position is not None:
        finger_position = np.array(args.finger_position)
    else:
        finger_position = get_finger_position(index)

    ### --- Where the simulation data u(X,t) is saved. u is displacment. X is world space coordinates of undeformed
    # output_directory = "deformed_bunny_files"
    OUTPUT_DIRECTORY = "./deformed_bunny_files_tunned"
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    OUTPUT_FILE = f"{OUTPUT_DIRECTORY}/displacement_{index}.xdmf"
    OUTPUT_FILE_H5 = f"{OUTPUT_DIRECTORY}/displacement_{index}.h5"

    # File handling
    if os.path.exists(OUTPUT_FILE):
        print(f"File {OUTPUT_FILE} already exists. Removing it.")
        os.remove(OUTPUT_FILE)
    # File handling
    if os.path.exists(OUTPUT_FILE_H5):
        print(f"File {OUTPUT_FILE_H5} already exists. Removing it.")
        os.remove(OUTPUT_FILE_H5)

    ret = main(index, finger_position, OUTPUT_FILE, DISPLACEMENT_NORMS_DIR)
    exit(ret)
