import meshio
import numpy as np
from scipy.spatial import KDTree
from dolfin import *


# Function to extract mesh points and connectivity
def get_tetra_mesh_data(file_path):
    mesh = meshio.read(file_path)
    points = mesh.points
    connectivity = mesh.cells_dict.get("tetra", None)
    if connectivity is None:
        raise ValueError("No tetrahedral cells found in the mesh file.")
    return points, connectivity


# Load mesh
mesh_file = "Bunny/bunny.mesh"
points, connectivity = get_tetra_mesh_data(mesh_file)

# Convert to FEniCS mesh
mesh = Mesh()
editor = MeshEditor()
editor.open(mesh, "tetrahedron", 3, 3)
editor.init_vertices(len(points))
editor.init_cells(len(connectivity))
for i, point in enumerate(points):
    editor.add_vertex(i, point)
for i, cell in enumerate(connectivity):
    editor.add_cell(i, cell)
editor.close()

# Function space
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Find the minimum Z coordinate and set a boundary just above it
min_z = np.min(points[:, 2])  # Find the minimum Z value
z_dim = np.max(points[:, 2]) - min_z  # Z dimension (difference between min and max Z)
boundary_height = min_z + z_dim / 10  # Boundary just above the minimum Z point (glued to the floor)

print(f"Minimum Z point: {min_z}, Z dimension: {z_dim}, Setting boundary at: {boundary_height}")

# Sphere Boundary (Fix points near the center within a given radius)
mesh_center_coords = np.mean(points, axis=0)  # Compute center of the mesh
sphere_radius = 0.05  # Radius of the sphere to fix the points near the center
print(f"Mesh Center: {mesh_center_coords}, Fixing points within radius {sphere_radius} of the center.")


class SphereSubDomain(SubDomain):
    def inside(self, x, on_boundary):
        distance_to_center = np.linalg.norm(x - mesh_center_coords)
        return distance_to_center < sphere_radius and on_boundary




sphere_boundary = SphereSubDomain()

# Create Dirichlet boundary condition for sphere boundary
bc_sphere = DirichletBC(V, Constant((0.0, 0.0, 0.0)), sphere_boundary)

# Rectangular Prism Boundary (Fix points in the given Z range)
z_min = np.min(points[:, 2])
z_dim = np.max(points[:, 2]) - z_min
z_range_min = z_min + z_dim / 20
z_range_max = z_min + 3 * z_dim / 20
print(f"Fixing points in Z range between {z_range_min} and {z_range_max}")


class RectangularPrismSubDomain(SubDomain):
    def inside(self, x, on_boundary):
        return (z_range_min <= x[2] <= z_range_max) and on_boundary


rectangular_prism_boundary = RectangularPrismSubDomain()

# Create Dirichlet boundary condition for the rectangular prism boundary
bc_rectangular_prism = DirichletBC(V, Constant((0.0, 0.0, 0.0)), rectangular_prism_boundary)

# Define force point and vector
force_point = np.array([0.01413998, 0.5773221, 0.7305683])
force_vector = [3.7035865112060438, 1.1480558282122117, -0.9825552245653257]

# Find the vertex at the force point
kdtree = KDTree(points)
force_vertex_id = kdtree.query(force_point)[1]
force_vertex_coords = points[force_vertex_id]

# Material properties (jelly-like material)
E = 5000  # Young's modulus (Pa)
nu = 0.35  # Poisson's ratio
rho = 1000  # Density (kg/m^3)

mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))


def epsilon(u):
    return sym(grad(u))


def sigma(u):
    return lmbda * div(u) * Identity(3) + 2 * mu * epsilon(u)


# Time-stepping parameters
dt = 0.005
T = 0.5
time = 0.0
u_n = Function(V)  # Previous displacement
v_n = Function(V)  # Velocity
a_n = Function(V)  # Acceleration
u_new = Function(V)  # Solution at current step

# Variational problem
u = TrialFunction(V)
v = TestFunction(V)
a_form = (rho / dt**2) * inner(u, v) * dx + inner(sigma(u), epsilon(v)) * dx
L_form = (rho / dt**2) * inner(u_n + dt * v_n + 0.5 * dt**2 * a_n, v) * dx

# Apply point force (this is the same as before)
subspaces = [V.sub(i) for i in range(3)]
sources = [PointSource(subspaces[i], Point(*force_vertex_coords), force_vector[i]) for i in range(3)]

# Apply force to the correct cell (find cell index of the force point)
force_cell_index = None
for i, cell in enumerate(connectivity):
    if force_vertex_id in cell:
        force_cell_index = i
        break

if force_cell_index is None:
    print("Force is applied to cell (tetrahedron) index: None")
else:
    print(f"Force is applied to cell (tetrahedron) index: {force_cell_index}")

# Damping parameters
damping_factor = 0.2  # You can adjust this for more or less damping

# Time loop
print("\nTime-stepping simulation:")
while time < T:
    time += dt
    print(f"Time step: {time:.2f}")

    # Solve the system with the updated form that includes damping
    solve(a_form == L_form, u_new, bc_sphere)

    # Apply point force at the identified cell (same cell throughout)
    for source in sources:
        source.apply(u_new.vector())

    # Update velocity and acceleration with damping
    a_n.vector()[:] = (u_new.vector() - u_n.vector() - dt * v_n.vector()) / (0.5 * dt**2)
    v_n.vector()[:] = v_n.vector() + dt * a_n.vector()
    v_n.vector()[:] *= 1 - damping_factor  # Apply damping to velocity
    u_n.assign(u_new)

    # Print displacement at the force point
    displacement = u_new(Point(*force_vertex_coords))
    print(f"  Displacement at force point: {displacement}")

    # Compute maximum displacement
    displacement_values = u_new.vector().get_local().reshape((-1, 3))
    max_displacement = np.max(np.linalg.norm(displacement_values, axis=1))
    print(f"  Maximum displacement magnitude: {max_displacement:.5f}")

print("Simulation complete.")
