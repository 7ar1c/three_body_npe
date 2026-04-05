#!/bin/bash
#SBATCH --mail-user=t3somani@uwaterloo.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name="3body_100k"
#SBATCH --partition=cpu_pr3        
#SBATCH --account=normal
#SBATCH --time=12:00:00            
#SBATCH --nodes=1                  
#SBATCH --ntasks=1                 
#SBATCH --cpus-per-task=32         
#SBATCH --mem=16G                  
#SBATCH --output=%x-%j.out         
#SBATCH --error=%x-%j.err          

echo "Starting data generation job on $HOSTNAME"
echo "Requested $SLURM_CPUS_PER_TASK cores."

# 1. Navigate to your working directory
cd /work/t3somani/amath_445_final_project

# 2. Activate your virtual environment 
source .venv/bin/activate

# 3. Execute the parallel Python job
python generate_cluster.py

echo "Job completed successfully."




