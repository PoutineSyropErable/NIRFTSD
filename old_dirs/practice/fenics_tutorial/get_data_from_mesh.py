import numpy

def get_array_from_conn(conn) -> numpy.ndarray:
    """
    takes  mesh.topology.connectivity(dim,0)
    return a 2d numpy array of connectivity
    """
    connectivity_array = conn.array
    offsets = conn.offsets

    # Convert the flat connectivity array into a 2D array
    connectivity_2d = numpy.array([
        connectivity_array[start:end]
        for start, end in zip(offsets[:-1], offsets[1:])
    ])

    return connectivity_2d



def get_data_from_fenics_mesh(mesh, do_print=True) -> (numpy.ndarray, numpy.ndarray):
    """ 
    Takes a fenics mesh 
    return (points, connectivity)
    """
    tdim = mesh.topology.dim 
    conn = mesh.topology.connectivity(tdim,0)
    # print("type conn = ",type(conn))
    # print(help(type(conn)))
    # Extract the connectivity data and offsets
    connectivity_2d = get_array_from_conn(conn)

    # Print the result, checkthing the data we GOT(obtained) from dolfinx
    got_points = mesh._geometry.x
    got_connectivity = connectivity_2d

    if do_print:
        print(f"shape(points) = {numpy.shape(got_points)}")
        print(f"shape(connectivity) = {numpy.shape(got_connectivity)}\n")

        print(f"type(points) = {type(got_points)}")
        print(f"type(connectivity) = {type(got_connectivity)}\n\n")

        print(f"Mesh geometry (Points):\n{got_points}\n\n")
        print(f"Mesh Topology Connectivity (numpy.array):\n{got_connectivity}")


    return got_points, got_connectivity

