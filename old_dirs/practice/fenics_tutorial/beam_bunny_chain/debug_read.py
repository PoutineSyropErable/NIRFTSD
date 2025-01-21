from dolfinx import mesh, io
from mpi4py import MPI

def debug_mesh_and_meshtags():
    # Debugging mesh and meshtags loading
    print("Loading mesh and cell tags...")
    
    try:
        # Load the mesh and cell meshtags
        with io.XDMFFile(MPI.COMM_WORLD, "bunny.xdmf", "r") as xdmf:
            mesh_data = xdmf.read_mesh(name="Grid")
            print("Mesh loaded successfully.")
            
            # Read cell tags (if available)
            try:
                ct = xdmf.read_meshtags(mesh_data, name="Grid")
                print(f"Cell meshtags loaded. Total tags: {len(ct.values)}")
            except Exception as e:
                print("Error reading cell meshtags:", e)

        # Create connectivity for boundaries
        print("Creating boundary connectivity...")
        mesh_data.topology.create_connectivity(mesh_data.topology.dim, mesh_data.topology.dim - 1)

        # Load facet tags
        with io.XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
            try:
                ft = xdmf.read_meshtags(mesh_data, name="Grid")
                print(f"Facet meshtags loaded. Total tags: {len(ft.values)}")
            except Exception as e:
                print("Error reading facet meshtags:", e)

        # Output debug info about the mesh
        print(f"Mesh dimension: {mesh_data.topology.dim}")
        print(f"Number of vertices: {mesh_data.geometry.x.shape[0]}")
        print(f"Number of cells: {mesh_data.topology.index_map(mesh_data.topology.dim).size_local}")

        # If meshtags are loaded, print some values
        if 'ct' in locals():
            print("Sample cell tags:", ct.values[:min(len(ct.values), 10)])
        if 'ft' in locals():
            print("Sample facet tags:", ft.values[:min(len(ft.values), 10)])

    except Exception as e:
        print("Error during mesh or meshtags loading:", e)


# Run the debug function
debug_mesh_and_meshtags()

