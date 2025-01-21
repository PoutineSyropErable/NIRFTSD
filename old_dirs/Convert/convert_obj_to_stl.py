import trimesh
import sys
import os

def convert_obj_to_stl(input_file):
    """
    Converts an OBJ file to an STL file using Trimesh.
    
    Args:
        input_file (str): Path to the input OBJ file.
        
    Returns:
        str: Path to the output STL file.
    """
    # Check input file exists
    if not os.path.isfile(input_file):
        print(f"Error: File '{input_file}' does not exist.")
        return
    
    # Load the mesh
    mesh = trimesh.load(input_file, file_type="obj")
    
    # Check if mesh is valid
    if not mesh.is_empty:
        print("Successfully loaded the OBJ file.")
    else:
        print("Error: Loaded mesh is empty.")
        return
    
    # Repair the mesh (optional)
    mesh = mesh.process(validate=True)
    
    # Generate output STL filename
    output_file = os.path.splitext(input_file)[0] + ".stl"
    
    # Export to STL
    mesh.export(output_file)
    print(f"Converted '{input_file}' to '{output_file}'.")
    
    return output_file

def main():
    """
    Main function to convert an OBJ file to an STL file.
    Usage: python convert_obj_to_stl.py <path_to_input.obj>
    """
    if len(sys.argv) < 2:
        print("Usage: python convert_obj_to_stl.py <input_file.obj>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = convert_obj_to_stl(input_file)
    
    if output_file:
        print(f"Output STL file: {output_file}")

if __name__ == "__main__":
    main()
