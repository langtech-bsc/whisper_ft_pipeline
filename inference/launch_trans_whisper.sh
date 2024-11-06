#!/bin/bash
#SBATCH --job-name=ft_whisper
#SBATCH -D .
#SBATCH --output=logs/%x_%j.out
#   ------   SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=40
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --account bsc88
#SBATCH --exclusive
#SBATCH --qos acc_debug

######################
### Set enviroment ###
######################
date

##Comment if launching with conda
#module load intel mkl python/3.12.1
#export LD_LIBRARY_PATH=/gpfs/apps/MN5/ACC/HDF5/1.14.1-2/NVIDIA/NVHPCX/lib/:$LD_LIBRARY_PATH
##Mi ambiente
#source /gpfs/projects/bsc88/speech/ASR/environments/whisper_19_04_2024/bin/activate
#export HF_DATASETS_CACHE=CATALAN_DATA_REPO/CACHE
#export HF_HOME=HF_CACHE

#--nproc_per_node is the number of GPUs you are using
#time torchrun --nproc_per_node 1 trans_whisper_hf.py --all_arguments_of_the_script
time python trans_whisper_hf.py

date
######################
#pip cache purge

