#!/bin/bash
#SBATCH --job-name=whisper_ft_deepspeed_full_cat
#SBATCH --output=./logs/%x_%j.log
#SBATCH --error=./logs/%x_%j.err
#SBATCH --nodes 16
#SBATCH -c 80
#SBATCH --gres=gpu:4
#SBATCH --time=1-12:00:00
#SBATCH -A bsc88
##SBATCH --qos=acc_debug
#SBATCH --exclusive
#SBATCH -q acc_bscls  




# For debugging:
#-#sbatch -q acc_debug launch.sh
#-#salloc -A bsc81 -q acc_debug -n 1 -c 80 --gres=gpu:4 -t 02:00:00

# adapting script for instruction tuning
# "caller_train_salamandra40b.sh"
# /gpfs/projects/bsc88/text/models/instruction-tuning/it-chat-v1/caller_train_salamandra40b.sh

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
#export SLURM_CPU_BIND=none # This line accelerates training x4 in mn5

chmod +x launch.sh
echo "run launch.sh"
srun --wait=100 launch.sh
