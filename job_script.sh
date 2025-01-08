#!/bin/bash

# Job script for Training Model
# Description: Runs the main.py script for your project, leveraging GPU resources on the HPC.

#SBATCH -J train_model           # Job name
#SBATCH -p gpu                   # Use the GPU partition
#SBATCH --gres=gpu:1             # Request 1 GPU
# #SBATCH --nodelist=gpu04       # Uncomment to specify exact gpu to use
#SBATCH -N 1                     # Request 1 node
#SBATCH -n 1                     # Request 1 task
#SBATCH -o /users/%u/job_output_%j.out  # Standard output file
#SBATCH -e /users/%u/job_error_%j.err   # Standard error file

echo "Job started..."

# Run the Python script
python -u ~/sharedscratch/code/src/main.py
echo "Script execution completed."
