#!/bin/bash
#SBATCH --job-name=hopper_medium_replay         # Job name
#SBATCH --output=hopper_medium_replay_%j.log    # Standard output and error log (%j creates a unique ID)
#SBATCH --time=12:00:00                           # Set a maximum time limit (48 hours)
#SBATCH --gpus=1                                  # Number of GPUs required
#SBATCH --mem-per-gpu=12G                         # Memory per GPU
#SBATCH --cpus-per-task=2                         # Number of CPU cores per task (adjust as needed)
#SBATCH --open-mode=append

interval=${1:-20}

# Load necessary environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mgoyani/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Activate the Python virtual environment
source /home/mgoyani/ENV/bin/activate

# Run the Python script
python sdt_hopper.py --env_name hopper-medium-replay-v2 --subgoal_interval $interval
