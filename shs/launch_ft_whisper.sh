#!/bin/bash
#SBATCH --job-name=whisper_ft_es_fullcat
#SBATCH -D .
#SBATCH --output=logs/%x_%j.out
#   ------   SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=100
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:0
#SBATCH --account bsc88
##SBATCH --exclusive
##SBATCH --qos acc_debug
#SBATCH -q gp_bscls                    # QoS for life sciences in nodes with GPUs (acc_bscls)
##SBATCH -q acc_bscls  


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

export GPUS_PER_NODE=4
export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $GPUS_PER_NODE))
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

echo "Master node: $MASTER_ADDR"
echo "World size: $WORLD_SIZE"
echo "SLURM nodes: $SLURM_NNODES"
echo "GPUs per node: $GPUS_PER_NODE"

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
######################

export NCCL_IB_TIMEOUT=150
export NCCL_SOCKET_TIMEOUT=150

srun accelerate launch train/ft_whisper_accelerate.py

date
######################
#pip cache purge

