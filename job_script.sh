#!/bin/bash -l
#SBATCH --job-name=vae_test       # Job name
#SBATCH --output=output_%j.txt    # Standard output log file (with job ID)
#SBATCH --error=error_%j.txt      # Standard error log file (with job ID)
#SBATCH --partition=gpu           # Partition name for GPU jobs
#SBATCH --gres=gpu:1              # Request one GPU
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --mem=16G                 # Amount of memory requested
#SBATCH --time=01:00:00           # Maximum runtime (1 hour)

# Load necessary modules
module load python/3.x            # Replace with the Python version on the HPC
module load pytorch/1.x-cuda11    # Replace with the available PyTorch module

# Activate virtual environment (optional if you create one on HPC)
# source ~/venv/bin/activate

# Run the Python code
python -m code.model.train        # Adjust if the entry point is different
