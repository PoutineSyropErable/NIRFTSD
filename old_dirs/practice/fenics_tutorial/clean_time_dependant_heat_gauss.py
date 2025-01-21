import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

# Define temporal parameters
t = 0  # Start time
T = 1.0  # Final time
num_steps = 50
dt = T / num_steps  # time step size


nx, ny = 50, 50
boundary_xy = np.array([2,2])
# [[x_min, y_min], [x_max, y_max]]
domain = mesh.create_rectangle(MPI.COMM_WORLD, [-boundary_xy, boundary_xy], [nx, ny], mesh.CellType.triangle)

xdmf = io.XDMFFile(domain.comm, "diffusion_clean.xdmf", "w")
xdmf.write_mesh(domain)


def initial_condition(x, a=5):
    return np.exp(-a * (x[0]**2 + x[1]**2))

V = fem.functionspace(domain, ("Lagrange", 1))
u_fem = fem.Function(V)
u_fem.name = "u_fem"
u_fem.interpolate(initial_condition)

u_fem_previous = fem.Function(V)
u_fem_previous.name = "u_fem_previous"
u_fem_previous.interpolate(initial_condition) # dont use a lambda expression


# Solving:
# du/dt = laplacian(u) + f 
# u = u_D on boundary_Omega x (0,T)
# u = u_0 at t = 0
# f = 0
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
u_D = PETSc.ScalarType(0)
f = PETSc.ScalarType(0)
# f = fem.Constant(domain, PETSc.ScalarType(0)) # we chose f =0, but could have whatever we wanted
a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_fem_previous + dt * f) * v * ufl.dx

bilinear_form = fem.form(a)
linear_form = fem.form(L)



fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
# Boundary facets are all the lines are the boundary
boundary_points_index = fem.locate_dofs_topological(V, fdim, boundary_facets)
# Boundary_points_index = boundary_dof
bc = fem.dirichletbc(u_D, boundary_points_index , V)





A = assemble_matrix(bilinear_form, bcs=[bc]) # a(u,v)
A.assemble()
b = create_vector(linear_form) # L(v)


solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)


# Initialize PyVista plotter and animation setup
plotter = pyvista.Plotter()
plotter.open_movie("bunny_deformation_animation.mp4", framerate=10)
plotter.add_axes()
plotter.add_text("Bunny Deformation Animation", font_size=12, color="black")

for i in range(num_steps):
    t += dt

    # Update the right hand side reusing the initial vector
    print(type(b))
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # Apply Dirichlet boundary condition to the vector
    # bilinear_form = A
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Solve linear problem
    solver.solve(b, u_fem.x.petsc_vec)
    u_fem.x.scatter_forward()

    # Update solution at previous time step (u_fem_previous)
    u_fem_previous.x.array[:] = u_fem.x.array

    # Write solution to file
    xdmf.write_function(u_fem, t)



xdmf.close()



