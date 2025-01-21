# Scaled variable
import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type, nls, log
from dolfinx.io.utils import XDMFFile
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import numpy as np
DEBUG_ = True

# Material properties
E = 10e4  # Young's modulus for rubber in Pascals (Pa)
nu = 0.40  # Poisson's ratio for rubber
rho = 1 # Some rubber that weighs as much as metal
g = 10000 # Jupiter gravity lol

# Lamé parameters
mu = E / (2 * (1 + nu))
lambda_ = (E * nu) / ((1 + nu) * (1 - 2 * nu))


# Load the mesh from the XDMF file
with XDMFFile(MPI.COMM_WORLD, "bunny.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
    print("Mesh loaded successfully!")


V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))



Z_GROUND = 0
def grounded_bunny(x):
    return x[2] <= Z_GROUND

def info(x):
    print(f"type(x) = {type(x)}\n")
    print(f"np.shape(x) = {np.shape(x)}\n")
    print(f"x = \n{x}\n")
    print(f"x.shape[1] = {x.shape[1]}\n")
    return np.ones(x.shape[1])



tdim = domain.topology.dim # 3D
fdim = tdim - 1 # 2D
boundary_facets = mesh.locate_entities_boundary(domain, fdim , grounded_bunny)
all_boundary_facets = mesh.locate_entities_boundary(domain, fdim , info)
if DEBUG_:
	print(f"boundary_facets = \n{boundary_facets}, len(boundary_facets) = {len(boundary_facets)}")
if DEBUG_:
	print(f"all_boundary_facets = \n{all_boundary_facets}, len(all_boundary_facets) = {len(all_boundary_facets)}")


u_D = np.array([0, 0, 0], dtype=default_scalar_type) # no displacement on boundary
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V) # create a condition where u = 0 on z = 0




def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


# u = ufl.TrialFunction(V)
uh = fem.Function(V)
uh.x.array[:] = 0  # Initialize displacement to zero
v = ufl.TestFunction(V)


USE_PRESSURE = True # My H(x,u) condition
if USE_PRESSURE: # Not linear due to boundary condition and H(x,u), since finger moves with boundary
    # Deformed position x + u
    x = ufl.SpatialCoordinate(domain)  # Reference position (Without deformation)
    x_deformed = x + uh   # Plus u makes it non linear


    # Finger position (center of the sphere) and radius
    finger_position = (0.098629, 0.004755, 0.118844)
    R = 0.007  # Radius of the sphere

    # Distance from the sphere center (finger position) in the deformed configuration
    distance = ufl.sqrt(
        (x_deformed[0] - finger_position[0])**2 +
        (x_deformed[1] - finger_position[1])**2 +
        (x_deformed[2] - finger_position[2])**2
    )

    H = ufl.conditional(ufl.lt(distance, R), 1.0, 0.0)

    # H(x,u) is sadly what causes a to not be Bilinear, it is not linear in u since a(u,v) = bilinear(u,v) - H(u)*v*p*(...) and H(u) is non linear
    # Hence, I'll need a time dependant, linear elastic, but non linear boundary conditions pde.

    # Pressure
    pressure = 5000.0
    p = fem.Constant(domain, default_scalar_type(-pressure))  # Negative for compression

    # Traction term with H(x, u)
    traction_term = H * p * ufl.dot(v, ufl.FacetNormal(domain)) * ufl.ds

    # No external force applied at every point
    f = fem.Constant(domain, default_scalar_type((0, 0, 0)))

    # Bilinear form
    a = ufl.inner(sigma(uh), epsilon(v)) * ufl.dx

    # (Semi) Linear form
    L = ufl.dot(f, v) * ufl.dx + traction_term

    F = a - L





problem = NonlinearProblem(F, uh, bcs=[bc])

solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 4e-3
solver.report = True
solver.relaxation_parameter = 0.075  # Reduce step size to avoid divergence
solver.line_search = "basic"


ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
print(f"\n\noptions prefix = \n{option_prefix}\n")
print(f"\n\nopts = \n{opts}\n")
opts[f"{option_prefix}ksp_type"] = "gmres"
opts[f"{option_prefix}ksp_rtol"] = solver.rtol
opts[f"{option_prefix}pc_type"] = "hypre"
opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 2
opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"
ksp.setFromOptions()


log.set_log_level(log.LogLevel.INFO)
n, converged = solver.solve(uh)
# assert (converged)
print(f"Number of interations: {n:d}")



# Create plotter and pyvista grid
p = pyvista.Plotter()
# Add title
p.add_text("Wireframe: Original | Colored Solid: Deformed", 
           position="upper_edge", 
           font_size=12, 
           color="black", 
           shadow=True)

#---------------------------------------------------- z = Z_GROUND plane
dmax = 0.15511219
plane_resolution = 10  # Number of subdivisions along each axis

# Generate a grid of points for the plane
x = np.linspace(-dmax, dmax, plane_resolution)
y = np.linspace(-dmax, dmax, plane_resolution)
x, y = np.meshgrid(x, y)
z = np.ones_like(x)*Z_GROUND  # z = 0 for the entire plan
# Create the PyVista grid for the plane
plane = pyvista.StructuredGrid(x, y, z)
p.add_mesh(plane, color="lightgray", opacity=0.5, label=f"z = {Z_GROUND} Plane")
#------------------------------------------------------ FINGER PRESSURE POINT
# Define sphere marker for the finger position
finger_marker = pyvista.Sphere(radius=R, center=finger_position)

# Add the sphere marker to the plot
p.add_mesh(finger_marker, color="red", label="Finger Position", opacity=1.0)

#------------------------------------------------------ BUNNY MESH 

topology, cell_types, geometry = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Attach vector values to grid and warp grid by vector
grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1.5)
actor_1 = p.add_mesh(warped, show_edges=True)


p.show_axes()
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure_as_array = p.screenshot("deflection.png")



print("\n\n-----End of Program-----\n\n")
