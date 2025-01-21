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

python ./14_train_nn_family.py --start_from_zero

# python ./6_create_animation.py --index=14 # This will show you the animation
