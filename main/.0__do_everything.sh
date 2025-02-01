#!/bin/bash
# It's called build.sh because of my macros and keybinds to run the files build.sh thats in my current dir

set -euo pipefail


#----------------------------------------------------Activate master_venv
### You'll need to input the correct conda PATH
# Define the path to Conda
CONDA_PATH="$HOME/miniconda3/bin/conda"
CONDA_VENV_NAME="master_venv"
########### YOU NEED TO HAVE INSTALLED MASTER_VENV USING THE ENVIRONMENT FILE in project root

activate_conda_env() {
    if [ -x "$CONDA_PATH" ]; then
        # Initialize Conda for Bash
        eval "$($CONDA_PATH shell.bash hook)"

        # Activate the Conda environment
        conda activate "$CONDA_VENV_NAME"

        # Add any additional commands after activation
        echo "Conda environment '$CONDA_VENV_NAME' activated."
    else
        echo "Error: Conda not found at '$CONDA_PATH'"
        exit 1
    fi
}

# activate_conda_env

#------------------------------------------------------- Run programs

echo "-------------------------------------------GETTING TETRA MESH-------------------------------------------------"
# It's git included, running the tree bellow will recreate the data using a different seed. so index=730 will be a 
# different position. Due to linear elasticity equations, if its on ear, it will cause weird behavior in phys sim.
# Garbage in, garbage out
# python ./1_get_tetra_mesh.py
echo "-------------------------------------------GETTING POINTS TO APPLY FORCE-------------------------------------------------"
# python ./2_get_points_to_apply_force.py
echo "-------------------------------------------FILTERTING POINTS-------------------------------------------------"
# python ./3_filter_points.py
echo "-------------------------------------------DOING PHYSICS SIMULATION-------------------------------------------------"
python ./4_linear_elasticity_finger_pressure_bunny_dynamic.py --index=730
# python ./5_iterate_simulations.py & 
# create an animation for the 14th point where the force was applied
# python ./7_create_all_plots.py &
# python ./8_filter_too_deformed_meshes.py 

echo "-------------------------------------------CREATING ANIMATION-------------------------------------------------"
python ./6_create_animation.py --index=730
# python ./9_create_all_animation.py --filter=all & 

echo "-------------------------------------------CALCULATION SDF-------------------------------------------------"
python ./10_1_calculate_sdf_mini.py 
echo "-------------------------------------------SDF History-------------------------------------------------"
python ./10_6_show_sdf_hist.py 
# python ./11_iterate_sdf_calculation.py --starting_index=0 --stopping_index=-1 --sdf_only --doall

echo "-------------------------------------------SDF VISUALISATION-------------------------------------------------"
python ./12_see_sdf.py --finger_index=730 --time_index=100
# python ./13_see_sdf_animation_and_store_data.py
python ./14_train_nn_family.py --finger_index=730 --start_from_zero

python ./16_5_recreate_simple.py --epoch=482
python ./17_see_loss.py --epoch_index=482
python ./19_track_latent_vectors.py --max_epoch=1000


# python ./6_create_animation.py --index=14 # This will show you the animation


