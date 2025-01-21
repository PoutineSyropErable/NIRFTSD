import numpy as np

def signed_volume(p1, p2, p3, p4):
    """
    Compute the signed volume of a tetrahedron formed by points p1, p2, p3, and p4.
    Parameters:
    - p1, p2, p3, p4: Points in 3D space (x, y, z).
    Returns:
    - Signed volume of the tetrahedron.
    """
    mat = np.array([
        [p1[0], p1[1], p1[2], 1],
        [p2[0], p2[1], p2[2], 1],
        [p3[0], p3[1], p3[2], 1],
        [p4[0], p4[1], p4[2], 1],
    ])
    return np.linalg.det(mat) / 6.0


def point_in_tetra(point, tetra_points):
    """
    Determine if a point is inside a tetrahedron in 3D space.
    Parameters:
    - point: The query point as (x, y, z).
    - tetra_points: List of 4 vertices of the tetrahedron [(x1, y1, z1), ...].
    Returns:
    - True if the point is inside the tetrahedron, False otherwise.
    """
    p = np.array(point)
    t1, t2, t3, t4 = [np.array(tp) for tp in tetra_points]

    # Compute signed volumes
    v_tetra = signed_volume(t1, t2, t3, t4)  # Volume of the tetrahedron
    v1 = signed_volume(p, t2, t3, t4)        # Volume with point and 3 vertices
    v2 = signed_volume(t1, p, t3, t4)
    v3 = signed_volume(t1, t2, p, t4)
    v4 = signed_volume(t1, t2, t3, p)

    # Check if the point is inside
    is_inside = (np.sign(v1) == np.sign(v2) == np.sign(v3) == np.sign(v4)) and (
        np.isclose(abs(v1) + abs(v2) + abs(v3) + abs(v4), abs(v_tetra))
    )
    return is_inside


# Provided input data
simplex = 13560
point = np.array([0.08387797, 0.0014265, 0.08470987])
points_near = np.array([
    [0.10094929, -0.00463639, 0.02522489],
    [0.10239036, -0.01163404, 0.02257239],
    [0.098985, -0.01231134, 0.02429209],
    [0.09688637, -0.00551895, 0.01967707]
])

# Check if the point is inside the tetrahedron
is_point_inside = point_in_tetra(point, points_near)
print(f"Is the point {point} inside the tetrahedron? {is_point_inside}")

# Additional information
deltas = points_near - point
print(f"deltas = \n{deltas}\n")

distance = np.linalg.norm(deltas, axis=1)
print(f"distance = {distance}")
max_distance = np.max(distance)
print(f"max_distance = {max_distance}")


