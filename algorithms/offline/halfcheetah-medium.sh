#!/bin/bash
#SBATCH --job-name=halfcheetah_medium         # Job name
#SBATCH --output=halfcheetah_medium_%j.log    # Standard output and error log (%j creates a unique ID)
#SBATCH --time=12:00:00                           # Set a maximum time limit (48 hours)
#SBATCH --gpus=1                                  # Number of GPUs required
#SBATCH --mem-per-gpu=12G                         # Memory per GPU
#SBATCH --open-mode=append


interval=${1:-20}
# Load necessary environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mgoyani/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Activate the Python virtual environment
source /home/mgoyani/ENV/bin/activate

# Run the Python script
python sdt_hc.py --env_name halfcheetah-medium-v2 --subgoal_interval $interval
