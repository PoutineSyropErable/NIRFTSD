import numpy as np
from scipy.spatial import Delaunay


def barycentric_coordinates_scipy(tetra, point):
    """
    Calculate the barycentric coordinates of a point relative to a tetrahedron using scipy.

    Parameters:
        tetra (np.ndarray): 4x3 array of tetrahedron vertices.
        point (np.ndarray): 1x3 array of the point coordinates.

    Returns:
        np.ndarray: Barycentric coordinates [lambda_1, lambda_2, lambda_3, lambda_4].
    """
    # Create Delaunay triangulation
    delaunay = Delaunay(tetra)
    # Find the barycentric coordinates
    bary = delaunay.transform[0].dot(np.append(point, 1))
    return bary


# Example usage
A = np.array([0, 0, 0])
B = np.array([1, 0, 0])
C = np.array([0, 1, 0])
D = np.array([0, 0, 1])
tetra = np.array([A, B, C, D])

P = np.array([0.25, 0.25, 0.25])

bary_coords = barycentric_coordinates_scipy(tetra, P)
print("Barycentric Coordinates (Scipy):", bary_coords)
