#!/bin/bash
#SBATCH --job-name=file_integrity
#SBATCH -D .
#SBATCH --output=logs/%x_%j.out
#   ------   SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=80
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4
#SBATCH --account bsc88
##SBATCH --exclusive
#SBATCH --qos acc_debug
##SBATCH -q acc_bscls          # QoS for life sciences in nodes with GPUs (acc_bscls)


######################
### Set enviroment ###
######################

date

source /gpfs/projects/bsc88/speech/research/environments/SoftPrompting/bin/activate         #Mi ambiente

export GPUS_PER_NODE=4

######################

srun python utils/check_integrity.py --input_tsv /gpfs/projects/bsc88/speech/data/raw_data/1_DATALOADERS/ALBAYZIN_2024_DATALOADER/data/dev.tsv --output_file /gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_whisper_expes/ft_whisper_acc/utils/alb_train.tsv

######################
date