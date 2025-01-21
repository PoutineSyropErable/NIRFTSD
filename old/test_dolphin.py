from dolfin import *

# Define mesh and function space
mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define boundary condition
u_D = Constant(0.0)
bc = DirichletBC(V, u_D, "on_boundary")

# Define problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Output solution to terminal
print("Test successful! Solution at center:", u(Point(0.5, 0.5)))

