#!/bin/bash
#SBATCH --job-name=LORAEVAL_LARGEV2_TRAIN
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --account bsc88
#SBATCH -q acc_bscls
##SBATCH --qos acc_debug
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lucas.sanchez@bsc.es
##SBATCH --exclusive


date
module load impi intel mkl hdf5 
module load EB/apps EB/install 
module load FFmpeg/6.0-GCCcore-13.2.0
#module load intel hdf5 mkl 
#export LD_LIBRARY_PATH=/gpfs/apps/MN5/ACC/HDF5/SRC/hdf5-hdf5-1_14_1-2/src/.libs/libhdf5.so.310:$LD_LIBRARY_PATH

source /gpfs/projects/bsc88/speech/research/environments/SoftPrompting/bin/activate


# Set environment variables
#export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY='805aa556027c20132de063b7c79fb877840d510d'

export GPUS_PER_NODE=4


export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export HF_HOME=/gpfs/projects/bsc88/speech/research/scripts/Lucas/cache
export HF_DATASETS_CACHE=/gpfs/projects/bsc88/speech/research/scripts/Lucas/cache
export CACHE_DIR=/gpfs/projects/bsc88/speech/research/scripts/Lucas/cache
export HF_DATASET=/gpfs/projects/bsc88/speech/research/scripts/Lucas/SoftPrompts/data_utils/loading_script_lt400.py

# Run the training script
#python evaluate_pt.py \
#    --eval_tsv /gpfs/projects/bsc88/speech/research/scripts/Lucas/SoftPrompts/data/lt400dev.tsv \
#    --model /gpfs/projects/bsc88/speech/research/models/hf_models/whisper-medium \
#    --model_name /gpfs/projects/bsc88/speech/research/scripts/Lucas/CHKs/model_asr_experiment_1_9 \
#    --prompt_length 16 \
#    --use_mlp \
#    --deep \
#    --batch_size 16
python loraeval.py 
date
    