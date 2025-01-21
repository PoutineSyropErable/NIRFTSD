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

# Define mesh
nx, ny = 50, 50
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-2, -2]), np.array([2, 2])],
                               [nx, ny], mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))


# Create initial condition
def initial_condition(x, a=5):
    return np.exp(-a * (x[0]**2 + x[1]**2))
# u(x,y,0) = e^-(ax² + ay²)
# radius 1/sqrt(a) 3d gaussian curve 

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition) # dont use a lambda expression


# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
# Theres 200 boundary points, so we could hard code 200, or 2*(n_x+n_y)

# Mark all point as part of the boundary? No it correctly puts all the boundary elements
# It x.shape[1] implies it takes the number of points, and that its transposed of how you do it in python
# so all x and adjacent and all y are adjacent... which makes caching better for x dot x + y dot y


DEBUG_ = False
PYVISTA_ = True


if DEBUG_:
	print(f"type(boundary_facets) = {type(boundary_facets)}")
	print(f"shape(boundary_facets) = {np.shape(boundary_facets)}")
	print(f"boundary_facets = \n{boundary_facets}\n")




bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)
# set 0 on the spatial boundary of [-2,2]x[-2,2] (The 4 lines of a square)

# time dependant stuff
xdmf = io.XDMFFile(domain.comm, "diffusion.xdmf", "w")
xdmf.write_mesh(domain)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
xdmf.write_function(uh, t)

u_other = fem.Function(V)
u_other.name = "u_other"
u_other.interpolate(initial_condition)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(0)) # we chose f =0, but could have whatever we wanted
a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n + dt * f) * v * ufl.dx

bilinear_form = fem.form(a)
linear_form = fem.form(L)
# bilinear_form = a """ if we do it this way it doesnt run"""
# linear_form = L


A = assemble_matrix(bilinear_form, bcs=[bc]) # a(u,v)
A.assemble()
b = create_vector(linear_form) # L(v)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)





if PYVISTA_:
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

    plotter = pyvista.Plotter()
    plotter.open_gif("u_time.gif", fps=10)

    grid.point_data["uh"] = uh.x.array
    warped = grid.warp_by_scalar("uh", factor=1)

    viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                position_x=0.1, position_y=0.8, width=0.8, height=0.1)

    renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                                cmap=viridis, scalar_bar_args=sargs,
                                clim=[0, max(uh.x.array)])



if DEBUG_:
    # print(f"type(b.localForm()) = {type(b.localForm())}\n")
    # print(f"b.localForm() = \n{b.localForm()}\n")
    # print(f"help(b.localForm()) = \n{help(b.localForm())}\n")
    pass



for i in range(num_steps):
    t += dt
    FIRST_ = np.isclose(t,0.02)

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        # b.localForm().set(0) but with locks and unlock for multithreading (what the with x as y is for) 
        if DEBUG_ and FIRST_:
            # print(f"t = {t}\n\n")
            # print(f"type(loc_b) = {type(loc_b)}\n")
            # print(f"loc_b = \n{loc_b}\n")
            pass
 
        loc_b.set(0)
        # loc_b is a vector, this set all b_i = 0

        if DEBUG_ and FIRST_:
            # print(f"loc_b = \n{loc_b}\n")
            # print(f"help(loc_b) = \n{help(loc_b)}\n")
            pass

    # A: a(u,v) - Left Hand Side, it doesn't change during the itteration
    # A = int_\Omega {uv + \delta t grad(u) dot grad(v)} dx - it doesn't change over time, since u_n+1 = u
    # bilinear_form = fem.form(a)
    # A = assemble_matrix(bilinear_form, bcs=[bc]) # a(u,v)
    # A.assemble()

    # b: L(v) - Right Hand Side
    # b = int_\Omega {u_n + \delta t f_n+1} - it changes over time
    # linear_form = fem.form(L)
    # b = create_vector(linear_form)
    assemble_vector(b, linear_form)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    # ghost update is due to multi process
    set_bc(b, [bc])

    # Solve linear problem
    if DEBUG_ and FIRST_:
        print(f"type(uh.x.petsc_vec) = {type(uh.x.petsc_vec)}\n")
        print(f"uh.x.petsc_vec = \n{uh.x.petsc_vec}\n")
        print(f"help(uh.x.petsc_vec) = \n{help(uh.x.petsc_vec)}\n")
        pass 
    #uh.x.petsc_vec is a <class 'petsc4py.PETSc.Vec'>, same as loc_b 
    solver.solve(b, uh.x.petsc_vec)
    # Solve Au = b, where u_calculated = f(A,b), ex: u = A⁻1 b. u is the where the output is saved. Hence, it gets u from A and b
    # It already has access to A = a(u,v)
    uh.x.scatter_forward()

    # Just for funs and giggles and testing
    # solver.solve(b, u_other.x.petsc_vec)
    # u_other.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array
    # Element wise copy, rather then = which would do a pointer copy
    # So basically, u_n.x.array = uh.x.array.clone() but dont change memory address
    # So, just an in place deep copy

    # Write solution to file
    xdmf.write_function(uh, t)
    # Update plot
    if PYVISTA_:
        new_warped = grid.warp_by_scalar("uh", factor=1)
        warped.points[:, :] = new_warped.points
        warped.point_data["uh"][:] = uh.x.array
        plotter.write_frame()
if PYVISTA_:
    plotter.close()
xdmf.close()
