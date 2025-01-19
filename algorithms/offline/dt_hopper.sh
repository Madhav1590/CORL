#!/bin/bash
#SBATCH --job-name=hopper
#SBATCH --time=3:00:00                           # Set a maximum time limit (3 hours)
#SBATCH --gpus=1                                  # Number of GPUs required
#SBATCH --mem-per-gpu=12G                         # Memory per GPU
#SBATCH --open-mode=append
#SBATCH --output="dt_hopper_%j.log"
env=${1}

# Load necessary environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mgoyani/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Activate the Python virtual environment
source /home/mgoyani/ENV/bin/activate

# Run the Python script
python dt_hopper.py --env_name $env
