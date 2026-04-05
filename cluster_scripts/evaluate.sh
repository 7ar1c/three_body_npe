#!/bin/bash
#SBATCH --mail-user=t3somani@uwaterloo.ca 
#SBATCH --mail-type=BEGIN,END,FAIL          
#SBATCH --job-name="npe_eval"
#SBATCH --partition=gpu-gen          
#SBATCH --account=normal          
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2                
#SBATCH --mem=8G                          # HARD LIMIT: Normal accounts cannot >
#SBATCH --time=12:00:00                   # Max runtime for the partition
#SBATCH --gres=gpu:1                      # Request 1 available GPU
#SBATCH --output=stdout-%x_%j.log         # Waterloo's preferred log naming con>
#SBATCH --error=stderr-%x_%j.log

echo "Job started on $(date)"
echo "Running on node: $SLURM_NODELIST"


# 2. Navigate to your project directory
cd /work/t3somani/amath_445_final_project

module load anaconda3/2024.02.1
eval "$(conda shell.bash hook)"
conda activate npe_env

echo "--- DIAGNOSTICS ---"
nvidia-smi
echo "Using Python located at: $(which python)"
python -c "import torch; print(f'PyTorch Version: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
echo "-------------------"



# 4. Run the code using srun (as explicitly requested in their docs)
echo "Starting PyTorch training..."
python -u  evaluate_cuda.py

echo "Job completed on $(date)"