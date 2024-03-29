#!/bin/bash
#SBATCH --account=rpp-bengioy            # Yoshua pays for your job
#SBATCH --cpus-per-task=4                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=3:00:00                   # The job will run for 3 hours
#SBATCH -o slurm-%j.out  # Write the log in $SCRATCH

# 1. Create your environement locally

#virtualenv --no-download $SLURM_TMPDIR/env
source /home/vermavik/virtualenv/pytorch_geo/bin/activate
module load cuda/10.0 

export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx512/Compiler/intel2018.3/cuda/10.0.130/lib64:$LD_LIBRARY_PATH


#pip install --no-index torch torchvision



# 2. Copy your dataset on the compute node
#cp $SCRATCH/<dataset> $SLURM_TMPDIR

python main.py --target 7 --logfilename ig_rl_target7_lam_e-5_all_losses_eph200 --lamda 0.00001  --epochs 200 --separate-encoder --use-unsup-loss --usup-sup-method all_layer --bs 20 # --use-unsup-loss --separate-encoder
