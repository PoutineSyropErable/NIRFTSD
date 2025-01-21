# Scaled variable
import pyvista as pv
from dolfinx import mesh, fem, plot, io, default_scalar_type, nls, log
from dolfinx.io.utils import XDMFFile
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import h5py
import os, sys
DEBUG_ = True

os.chdir(sys.path[0])

#--------------------------------------------------------HELPER FUNCTIONS-----------------------------------------

def get_array_from_conn(conn) -> np.ndarray:
    """
    Convert mesh topology connectivity to a 2D numpy array.

    Parameters:
    - conn: The mesh topology connectivity (dolfinx mesh.topology.connectivity).

    Returns:
    - connectivity_2d: A 2D numpy array where each row contains the vertex indices for a cell.
    """
    connectivity_array = conn.array
    offsets = conn.offsets

    # Convert the flat connectivity array into a list of arrays
    connectivity_2d = [
        connectivity_array[start:end]
        for start, end in zip(offsets[:-1], offsets[1:])
    ]

    # Convert to numpy array with dtype=object to handle variable-length rows
    return np.array(connectivity_2d, dtype=object)


def get_function_value_at_point(domain, u, delaunay, query_point, cells, estimate_avg_cell_size):
    #------------------------------Need to work on it, it doesn't work
    """
    Evaluate the function `u` at a given query point.

    Parameters:
    - domain: The mesh domain.
    - u: The function to evaluate.
    - delaunay: Delaunay triangulation of the mesh points.
    - query_point: The point (x, y, z) where the function is to be evaluated.

    Returns:
    - Value of the function at the query point.
    """
    print("\n")
    # Find the simplex (cell) containing the point
    simplex = delaunay.find_simplex(query_point)
    print(f"simplex = {simplex}")
    if simplex == -1:
        raise ValueError(f"Point {query_point} is outside the mesh.")
    
    # Identify the cell containing the query point
    point = np.atleast_2d(query_point)
    print(f"point = {point}")
    containing_cell = np.array([simplex], dtype=np.int32)[0]

    print(f"containing_cell = {containing_cell}")
    points_index = connectivity[containing_cell]
    points_index = points_index.astype(np.int64)
    print(f"points_index = {points_index}")
    points_near = points[points_index]
    print(f"points_near = \n{points_near}\n")


    deltas = points_near - query_point 
    print(f"deltas = \n{deltas}\n")
    distance = np.linalg.norm(deltas, axis=1)
    print(f"distance = {distance}")
    max_distance = np.max(distance)
    print(f"max_distance = {max_distance}")
    
    # Compute the function value at the query point
    function_value = u.eval(point, containing_cell)
    return function_value

#-----------------------------------------------------    CODE STARTS --------------------------------------------



# Material properties
E = 5*10e3  # Young's modulus for rubber in Pascals (Pa)
nu = 0.40  # Poisson's ratio for rubber
rho = 1 # Some rubber that weighs as much as metal

# Finger position (center of the sphere) and radius
finger_position = np.array([0.098629, 0.004755, 0.118844])
R = 0.003  # Radius of the sphere
# Pressure
pressure = 40000


# Time parameters
t = 0.0
T = 0.001  # Total simulation time
dt = 5e-8
num_steps = int(T/dt + 1) 
print(f"dt={dt}")

# Initialize arrays to store time and maximum displacement
t_array = np.linspace(0, T, num_steps)
max_displacement_array = np.zeros(num_steps)

# Lamé parameters
mu = E / (2 * (1 + nu))
lambda_ = (E * nu) / ((1 + nu) * (1 - 2 * nu))




def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)



def load_file(filename):
    # Load the mesh from the XDMF file
    with XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        print("Mesh loaded successfully!")
    return domain

domain = load_file("bunny.xdmf")
# Extract points and cells from the mesh
points = domain.geometry.x  # Array of vertex coordinates
topology = domain.topology.connectivity(3, 0)
connectivity = get_array_from_conn(topology)  # 2D numpy array of cell connectivity
# Create a Delaunay triangulation
delaunay = Delaunay(points)

# Get the min and max for each column
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()
z_min, z_max = points[:, 2].min(), points[:, 2].max()

pos_min = np.array([x_min, y_min , z_min])
pos_max = np.array([x_max, y_max , z_max])

center = (pos_min + pos_max)/2 
bbox_size = (pos_max - pos_min)
print(f"center = {center}")
print(f"bbox_size = {bbox_size}")

# Calculate volume
# PyVista requires a `cells` array where the first value is the number of nodes per cell
num_cells = connectivity.shape[0]
print(f"number_cells = {num_cells}")
num_nodes_per_cell = connectivity.shape[1]
print(f"num_nodes_per_cell = {num_nodes_per_cell}")
cells = np.hstack([np.full((num_cells, 1), num_nodes_per_cell), connectivity]).flatten().astype(np.int32)

 # Check the shape and contents of the cells array
print(f"Shape of cells array: {cells.shape}")
print(f"First 20 elements of cells array: {cells[:20]}")

# Cell types: 10 corresponds to tetrahedrons in PyVista
cell_type = np.full(num_cells, 10, dtype=np.uint8)

# Create the PyVista UnstructuredGrid
tetra_grid = pv.UnstructuredGrid(cells, cell_type, points)
# Calculate the volume
volume = tetra_grid.volume
print(f"Volume of tetrahedral mesh: {volume}")

estimate_avg_cell_size = np.cbrt(volume/num_cells)
print(f"estimate_avg_cell_size = {estimate_avg_cell_size}")


interpol_s = 0.7
finger_inside_object = interpol_s *(center) + (1 - interpol_s)* finger_position



Z_GROUND = 0
def grounded_bunny(x):
    return x[2] <= Z_GROUND


V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

tdim = domain.topology.dim # 3D
fdim = tdim - 1 # 2D
boundary_facets = mesh.locate_entities_boundary(domain, fdim , grounded_bunny)
u_D = np.array([0, 0, 0], dtype=default_scalar_type) # no displacement on boundary
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V) # create a condition where u = 0 on z = 0


v = ufl.TestFunction(V)
u_t = ufl.TrialFunction(V)

u_n = fem.Function(V) # u_n
u_n1 = fem.Function(V) # u_n+1

u_tn = fem.Function(V) # du/dt (n)
u_tn1 = fem.Function(V) #du/dt (n+1)

# set u_n, a_tn to 0
# Initialize displacement and velocity to zero
u_n.interpolate(lambda x: np.zeros_like(x))
u_tn.interpolate(lambda x: np.zeros_like(x))
u_n1.interpolate(lambda x: np.zeros_like(x))
u_tn1.interpolate(lambda x: np.zeros_like(x))


# Body force
f = fem.Constant(domain, PETSc.ScalarType((0, 0, 0)))

p = fem.Constant(domain, PETSc.ScalarType(-pressure))  # Negative for compression

# Create the solver
solver = PETSc.KSP().create(domain.comm)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# Create the matrix and vector
a = (rho) * ufl.inner(u_t, v) * ufl.dx
bilinear_form = fem.form(a)
A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
solver.setOperators(A)


# Open HDF5 file to save displacements
h5_filename = "displacements.h5"
if os.path.exists(h5_filename):
    print(f"File {h5_filename} already exists. Removing it.")
    os.remove(h5_filename)
else:
    print(f"File {h5_filename} does not exist. Proceeding.")


num_points = domain.geometry.x.shape[0]  # Number of points in the mesh
NUMBER_OF_STEPS_BETWEEN_WRITES = 200
write_index = 0

NUMBER_OF_WRITES = (num_steps + NUMBER_OF_STEPS_BETWEEN_WRITES - 1) // NUMBER_OF_STEPS_BETWEEN_WRITES

print(f"NUMBER OF WRITES: {NUMBER_OF_WRITES}\n")

with h5py.File(h5_filename, "w") as h5_file:
    # Create a dataset for displacements
    displacements_dset = h5_file.create_dataset(
        "displacements", 
        shape=(NUMBER_OF_WRITES, num_points, 3), 
        dtype=np.float32, 
        compression="gzip"
    )

    print("")
    print(f"before loop: t = {t}")
    for step in range(num_steps):
        t += dt


        # Update H(x + u_n)
        x = ufl.SpatialCoordinate(domain)
        x_deformed = x + u_n
        distance = ufl.sqrt(
            (x_deformed[0] - finger_position[0])**2 +
            (x_deformed[1] - finger_position[1])**2 +
            (x_deformed[2] - finger_position[2])**2
        )
        H = ufl.conditional(ufl.lt(distance, R), 1.0, 0.0) # If touching finger

        # Traction term
        traction_term = H * p * ufl.dot(v, ufl.FacetNormal(domain)) * ufl.ds

        # Linear form
        L = (
            rho * ufl.inner(u_tn, v) * ufl.dx
            + dt * ufl.inner(f, v) * ufl.dx
            - dt * ufl.inner(sigma(u_n), epsilon(v)) * ufl.dx
            + dt * traction_term
        )
        linear_form = fem.form(L)
        b = create_vector(linear_form) # L(v)

        # Assemble RHS vector
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, linear_form)


        # Solve for velocity update (u_tn1)
        solver.solve(b, u_tn1.x.petsc_vec)
        u_tn1.x.scatter_forward()

        # Update displacement
        u_n.x.array[:] += dt * u_tn1.x.array
        # Reshape the displacement array to match the (num_points, 3) structure

        # Update velocity for the next time step
        u_tn.x.array[:] = u_tn1.x.array

        displacements = u_n.x.array.reshape(domain.geometry.x.shape)


        
        # Print displacement information for debugging
        displacement_norms = np.linalg.norm(displacements, axis=1)
        max_norm = np.max(displacement_norms)
        max_index = np.argmax(displacement_norms)
        point_with_max_displacement = domain.geometry.x[max_index]
        max_displacement_array[step] = max_norm


        if write_index >= NUMBER_OF_WRITES:
            print(f"Warning: write_index ({write_index}) exceeds NUMBER_OF_WRITES ({NUMBER_OF_WRITES})!")
            break


        if step == num_steps - 1 or (step % NUMBER_OF_STEPS_BETWEEN_WRITES) == 0:
            print("-----------------------------------------------")
            print(f"Time step {step+1}/{num_steps}, t = {t}")


            print(f"displacements = \n{displacements}\n")
            print(f"Maximum displacement norm: {max_norm}")
            print(f"Point with maximum displacement: {point_with_max_displacement}")

            """ This doesn't work for now"""
            # print(f"Point of finger pressure:        {finger_position}")
            # print(f"Point of finger (inside):        {finger_inside_object}")
            # finger_displacement = get_function_value_at_point(domain, u_n, delaunay, finger_inside_object, cells, estimate_avg_cell_size) 
            # print(f"Finger displacement  = {finger_displacement}")

            # Save it to a .h5 file
            displacements_dset[write_index, :, :] = displacements
            write_index += 1

            print("\n-----------------------------------------------")



print("\n\n-----End of Simulation-----\n\n")# Close the movie file





PLOT = False
if PLOT:
    max_displacement_array[step] = max_norm
    plt.figure()
    plt.title("Maximum of the norm of u over the domain as a function of t")
    plt.grid()
    plt.xlabel("t")
    plt.ylabel("max(norm(u))")
    plt.plot(t_array, max_displacement_array)
    plt.savefig("max_displacement_vs_time.png", dpi=300, bbox_inches="tight")
    plt.show()
