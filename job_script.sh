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

echo "Loading Anaconda module..."
module load apps/anaconda3/2023.03/bin
echo "Module loaded."

# Initialize Conda
echo "Initialising Conda..."
eval "$(conda shell.bash hook)"
echo "Conda initialised."

# Activate virtual environment
echo "Activating virtual environment..."
source ~/venv/bin/activate
echo "Environment activated."

# Run the Python script
echo "Running script..."
python -u ~/sharedscratch/src/main.py
echo "Script execution completed."

echo "Deactivating environment..."
conda deactivate

echo "Job complete."
