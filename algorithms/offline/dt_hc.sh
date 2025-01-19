#!/bin/bash
#SBATCH --job-name=dt_regular
#SBATCH --time=3:00:00                           # Set a maximum time limit (3 hours)
#SBATCH --gpus=1                                  # Number of GPUs required
#SBATCH --mem-per-gpu=12G                         # Memory per GPU
#SBATCH --open-mode=append

env=${1}
target_return=${2}

output_file="dt_${env}_%j.log"
exec > $output_file 2>&1

# Load necessary environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mgoyani/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Activate the Python virtual environment
source /home/mgoyani/ENV/bin/activate

# Run the Python script
python dt_hc.py --env_name $env --target_return $target_return

