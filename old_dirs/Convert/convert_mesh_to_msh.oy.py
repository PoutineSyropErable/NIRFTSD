import meshio
import numpy as np
import os
import sys

def convert_mesh_to_msh(input_file):
    """
    Converts a .mesh file (Medit format) to a .msh file (Gmsh format).
    
    Args:
        input_file (str): Path to the input .mesh file.
    """
    # Ensure the input file exists
    if not os.path.isfile(input_file):
        print(f"Error: File '{input_file}' does not exist.")
        return

    # Determine the output file path
    input_dir = os.path.dirname(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(input_dir, f"{base_name}.msh")

    try:
        # Read the input .mesh file
        mesh = meshio.read(input_file)

        p = mesh.points
        c = mesh.cells
        print(type(p),type(c))

        print(f"shape points: {np.shape(p)}")
        print(f"cells: {c}\n")
        print(f"cells[0]: {c[0]}\n")
        print(f"cells[1]: {c[1]}\n")
        print(f"cells[1].data: \n{c[1].data}\n")

        points = mesh.points
        connectivity = mesh.cells[1].data
        print(f"\n\npoints: \n{points}\n")
        print(f"connectivity: \n{connectivity}")


        # Write the mesh to .msh format
        meshio.write(output_file, mesh)
        print(f"Successfully converted '{input_file}' to '{output_file}'")
    except Exception as e:
        print(f"Failed to convert mesh file: {e}")

if __name__ == "__main__":
    # Check if a file argument is provided
    if len(sys.argv) < 2:
        print("Usage: python convert_mesh_to_msh.py <input_file.mesh>")
        sys.exit(1)

    # Convert the file
    input_mesh_file = sys.argv[1]
    convert_mesh_to_msh(input_mesh_file)
