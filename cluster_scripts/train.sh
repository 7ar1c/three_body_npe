#!/bin/bash
#SBATCH --mail-user=t3somani@uwaterloo.ca 
#SBATCH --mail-type=BEGIN,END,FAIL          
#SBATCH --job-name="npe_train"
#SBATCH --partition=gpu-gen          
#SBATCH --account=normal          
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2                
#SBATCH --mem=8G                          # HARD LIMIT: Normal accounts cannot exceed 8GB RAM
#SBATCH --time=12:00:00                   # Max runtime for the partition
#SBATCH --gres=gpu:1                      # Request 1 available GPU
#SBATCH --output=stdout-%x_%j.log         # Waterloo's preferred log naming convention
#SBATCH --error=stderr-%x_%j.log          # Waterloo's preferred error naming convention

echo "Job started on $(date)"
echo "Running on node: $SLURM_NODELIST"


# 2. Navigate to your project directory
cd ~/amath_445_final_project

# 3. Activate the virtual environment you already built
source .venv/bin/activate

# 4. Run the code using srun (as explicitly requested in their docs)
echo "Starting PyTorch training..."
srun python train.py

echo "Job completed on $(date)"