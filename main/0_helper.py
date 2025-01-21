import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dolfinx_mesh



def create_fenics_mesh_domain(points :np.ndarray, connectivity: np.ndarray) -> dolfinx_mesh.Mesh:
    points = points.astype(np.float64)
    connectivity = connectivity.astype(np.int64)

    # Define the coordinate element
    coordinate_element = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(points.shape[1],)))


    # Create the DOLFINx mesh
    mesh_domain: dolfinx_mesh.Mesh = dolfinx_mesh.create_mesh(MPI.COMM_WORLD, connectivity, points, coordinate_element)
    return mesh_domain
