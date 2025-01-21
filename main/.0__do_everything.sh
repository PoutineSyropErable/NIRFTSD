#!/bin/bash
# It's called build.sh because of my macros and keybinds to run the files build.sh thats in my current dir

#----------------------------------------------------Activate master_venv
### You'll need to input the correct conda PATH
# Define the path to Conda
CONDA_PATH="$HOME/miniconda3/bin/conda"
CONDA_VENV_NAME="master_venv"

if [ -x "$CONDA_PATH" ]; then
    # Initialize Conda for Bash
    eval "$($CONDA_PATH shell.bash hook)"
    
    # Activate the Conda environment
	conda activate "$CONDA_VENV_NAME" 
    
    # Add your commands here after activating the environment
    echo "Conda environment "$CONDA_VENV_NAME" activated."
else
    echo "Error: Conda not found at $CONDA_PATH"
    exit 1
fi

#------------------------------------------------------- Run programs

python ./1_get_tetra_mesh.py
python ./2_get_points_to_apply_force.py
python ./3_filter_points.py
# ./4_linear_elasticity_finger_pressure_bunny_dynamic.py is iteratively called by ./5_iterate_simulations.py 
python ./5_iterate_simulations.py & 
# create an animation for the 14th point where the force was applied
python ./7_create_all_plots.py &
python ./8_filter_too_deformed_meshes.py 

python ./9_create_animation.py --filter=all & 

python ./11_iterate_sdf_calculation.py --starting_index=0 --stopping_index=-1 --sdf_only --doall

python ./12_see_sdf.py 
python ./13_see_sdf_animation_and_store_data.py
python ./14_train_nn_family.py --start_from_zero

# python ./6_create_animation.py --index=14 # This will show you the animation

