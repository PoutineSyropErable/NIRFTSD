print(f"\n\n\n----------START OF PROGRAM----------\n\n\n")
from mpi4py import MPI
from dolfinx import mesh

# topological_dimension
# facets_dimension


# Our domain of PDE is a mesh of 8x8 squares. It's a unit square so its in [0,1]x[0,1] 
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)
# MPI.COMM_WORLD is to share one mesh who's data is distributed on every threads.
# MPI.COMM_self is to create one mesh per threads. 
""" We can run the program with  mpirun -n 2 python3 poisson.py"""
 
# Defining the finite element function space
from dolfinx.fem import functionspace
V = functionspace(domain, ("Lagrange", 1))
# Lagrange, the basis functions are associated with nodes. They take 1 at the nodes and 0 elsewhere. 
# They are continuous accross the boundary of elements
# 1 is for the degree of the polynomial. Hence, we have   /\  lagrange v(x) function, like in the math videos
# And the result will be a linear interpolate of it

from dolfinx import fem
uD = fem.Function(V)
# uD is the functions of the boundary terms. Where we add u(x->) = u_D(x->) on boundary Omega
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
# We hard coded, the function: u(x,y) = 1 + x² + 2y²
# Here, we calculated uD, however, it is applied to the whole functions space right now, we need to apply it only to the boundary
# Since we are testing for a function we already know

import numpy
# facets -> faces. (memorisation trick). A facets is the more general construct for the boundary.
tdim = domain.topology.dim # topological_dimension : The dimension of the space, it doesn't care about if we use triangle or square building blocs
fdim = tdim - 1            # facet dimension: The dimension of the boundary = tdim -1 becauses (n) integral to n-1 on \del \Omega or \partial\Omega
# A cube is 3d, it has square face of 2d
print(f"We are in a {tdim} dimension problem, the boundary has {fdim} dimension")
# The topology is related to the connectivity, and how vertices are connected.
domain.topology.create_connectivity(fdim, tdim) # This calculate the connectivity of for each facets: here, its the two points for a line
# Create facet to cell connectivity required to determine boundary facets
# Here, 1 facets = 2 points ( a line ), Since we are in 2d, boundary is in 1d
boundary_facets = mesh.exterior_facet_indices(domain.topology) # This is just the index of the connectivity array where the line is on \del\Omega


DEBUG_ = False
if DEBUG_:
    PRINT_MESH = True
    from get_data_from_mesh import get_array_from_conn, get_data_from_fenics_mesh
    points, connectivity = get_data_from_fenics_mesh(domain, PRINT_MESH)
    print(f"points dimension: {numpy.shape(points)}, connectivity dimension: {numpy.shape(connectivity)}")

    print(f"type(boundary_facets) = {type(boundary_facets)}") # 1d nd array of indices
    print(f"boundary_facets = \n{boundary_facets}\n")

    facets_connectivity = domain.topology.connectivity(fdim, 0)
    facets_connectivity_array = get_array_from_conn(facets_connectivity)

    boundary_lines = points[facets_connectivity_array[boundary_facets]]
    print(f"Boundary Points = \n{boundary_lines}")
    print(type(boundary_lines)) # 32, 2, 3 
    # What we have is 32 = 8x4 , (pointA,pointB), 3dim?  where 8x8. Hence 8 point on the boundary line. And 4 for square edges
    # It's 3dimensional because we have [x,y,z=0], due to C++ fixed types
    print(numpy.shape(boundary_lines)) # Hence: shape = (32, 2 ,3)

"""
For the current problem, as we are using the “Lagrange” 1 function space,
the degrees of freedom are located at the vertices of each cell,
thus each facet contains two degrees of freedom.

To find the local indices of these degrees of freedom, we use dolfinx.fem.locate_dofs_topological,
which takes in the function space,
the dimension of entities in the mesh we would like to identify and
the local entities.
"""

boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
# Given a representation of the boundary lines, return a index of boundary points
bc = fem.dirichletbc(uD, boundary_dofs)
# On the boundary, set u(x) = uD(x)



if DEBUG_:
    print("\n\n")
    print(f"type(boundary_dofs) = {type(boundary_dofs)}") # a 1d np array of int
    print(f"boundary_dofs = {boundary_dofs}")
    
    boundary_points = points[boundary_dofs]
    print(f"type(boundary_points) = {type(boundary_points)}")
    print(f"shape(boundary_points) = {numpy.shape(boundary_points)}")
    print(f"boundary_points = \n{boundary_points}\n")
    # Hence, as we can see, boundary_dofs are the index of the points located on the boundary


# Defining the trial and test function
import ufl
u = ufl.TrialFunction(V) # trial
v = ufl.TestFunction(V) # test

# Defining the source term
from dolfinx import default_scalar_type
f = fem.Constant(domain, default_scalar_type(-6))

# Defining the variational problem
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx
# int_\Omega grad(u) dot grad(v) dx = int_\Omega fv dx 
# Both are over the domain, so no ds. The boundary ds integral = 0
print(f"Type(a): {type(a)}, type(L): {type(L)}")

from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
# we obtain, u_h(x,y), which is an approximation of u(x,y) with the nodes and linear interpolation


##### Checking errors
V2 = fem.functionspace(domain, ("Lagrange", 2))
# Since we have a polynomial function , an exact solution with polynomial interpolation will work
uex = fem.Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
# we already know the exact solution


L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
# Why do we need fem.form, it transform it into a form: Ie, an intermediate between python math and C++ implementation. 
# But why was it not a form in the first place? it was an ufl expression
error_local = fem.assemble_scalar(L2_error)
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
# all reduce is so that the result of L2_p1 + L2_p2 -> L2_tot 
# Where, if we have 2 process to do computation faster, p1 and p2 are partitioning of the domain. 
# Each partitioning is given one process 
# local is the result for one thread, and one partition of the domain.
# We give List[Values], operation = sum -> single value


error_max = numpy.max(numpy.abs(uD.x.array-uh.x.array))
# Only print the error on one process
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")


import pyvista
print(pyvista.global_theme.jupyter_backend)


from dolfinx import plot
# pyvista.start_xvfb()
domain.topology.create_connectivity(tdim, tdim)
# Cell to cell connectivity. Here a cell is a triangle. Hence, which neighboor every triangle has
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)


# starter mesh
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("fundamentals_mesh.png")

# deformed mesh with colors, 2d, u = color mapped
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()


# 3d deformed mesh u(x,y)
warped = u_grid.warp_by_scalar()
plotter2 = pyvista.Plotter()
plotter2.add_mesh(warped, show_edges=True, show_scalar_bar=True)
if not pyvista.OFF_SCREEN:
    plotter2.show()



# save externally

from dolfinx import io
from pathlib import Path
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
filename = results_folder / "fundamentals"
with io.VTXWriter(domain.comm, filename.with_suffix(".bp"), [uh]) as vtx:
    vtx.write(0.0)
with io.XDMFFile(domain.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)
print("\n\n-------End of program, no crash")
