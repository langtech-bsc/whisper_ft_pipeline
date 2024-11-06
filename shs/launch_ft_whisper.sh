#!/bin/bash
#SBATCH --job-name=whisper_ft_es_test_tiny
#SBATCH -D .
#SBATCH --output=logs/%x_%j.out
#   ------   SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=25
#SBATCH --ntasks=25
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:4
#SBATCH --account bsc88
##SBATCH --exclusive
##SBATCH --qos acc_debug
##SBATCH -q gp_bscls                    # QoS for life sciences in nodes with GPUs (acc_bscls)
#SBATCH -q acc_bscls  


######################
### Set enviroment ###
######################
date

module load intel mkl hdf5
export LD_LIBRARY_PATH=/gpfs/apps/MN5/ACC/HDF5/SRC/hdf5-hdf5-1_14_1-2/src/.libs/libhdf5.so.310:$LD_LIBRARY_PATH
export MODULEPATH=/apps/ACC/modulefiles/applications:/apps/ACC/modulefiles/compilers:/apps/ACC/modulefiles/environment:/apps/ACC/modulefiles/libs


source /gpfs/projects/bsc88/speech/ASR/environments/whisper_10_07_2024/bin/activate #Mi ambiente


export GPUS_PER_NODE=4
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK} # support told us it may improve CPU performance
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

echo $head_node_ip

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
######################




time srun accelerate launch --mixed_precision fp16 train/ft_whisper_accelerate.py

date
######################
#pip cache purge

