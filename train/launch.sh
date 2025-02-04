#!/bin/bash
# =======================================
# Initialization
# =======================================

echo "Entered launch.sh"

# Activate environment
source /gpfs/projects/bsc88/speech/research/environments/DeepSpeed/bin/activate

# Define paths
export HF_HOME="/gpfs/projects/bsc88/speech/research/scripts/Lucas/DeepSpeed_Whisper/hf_cache"
export PREPROCESSED_DATA_CACHE="/gpfs/projects/bsc88/speech/data/cache/catalan_dataloader_cache/all_catalan_whisper/"
export HF_DATASETS_CACHE="/gpfs/projects/bsc88/speech/data/cache/catalan_dataloader_cache/all_catalan_whisper"
export HF_TRANSFORMERS_CACHE="/gpfs/projects/bsc88/speech/research/scripts/Lucas/DeepSpeed_Whisper/cache/model"

#export WANDB_DIR="/path/to/wandb"
mkdir -p $HF_HOME

# Setup WandB
export WANDB_PROJECT="whisper_finetune"
export WANDB_MODE="offline"

# Multinode setup
export NNODES=$SLURM_NNODES
export GPUS_PER_NODE=4
export WORLD_SIZE=$(($NNODES * $GPUS_PER_NODE))
export MASTER_ADDR=$SLURM_LAUNCH_NODE_IPADDR
export MASTER_PORT=29500

# Torchrun arguments
torchrun_args=(
  --nproc_per_node=$GPUS_PER_NODE
  --nnodes=$NNODES
  --rdzv_id=$SLURM_JOB_ID
  --rdzv_backend=c10d
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT
)

# Python script arguments
script_args=(
  --hf_dataset "/gpfs/projects/bsc88/speech/research/scripts/Lucas/SoftPrompts/loading_script.py"
  --preprocessed_data_cache ${PREPROCESSED_DATA_CACHE}
  --hf_dataset_cache ${HF_DATASETS_CACHE}
  --whisper_model "/gpfs/projects/bsc88/speech/research/models/hf_models/whisper-large-v3"
  --language "Catalan"
  --task "transcribe"
  --checkpoint_dir "checkpoints_full_cat"

  --per_device_train_batch_size 32
  --per_device_eval_batch_size 32
  --num_train_epochs 10

  --learning_rate 1e-5
  --logging_dir "/gpfs/projects/bsc88/speech/research/scripts/Lucas/DeepSpeed_Whisper/logs" 
  --report_to None #wandb 
  --deepspeed "/gpfs/projects/bsc88/speech/research/scripts/Lucas/DeepSpeed_Whisper/configs/03_ds_type3_config_autombs.json"
  #--reprocess_data
)

# For CPU offloading with Deepspeed: Make sure "CPU Adam" is compiled with GCC
export CXX=g++
export CC=g++
#export CUDA_LAUNCH_BLOCKING=1

# Launch the training script
torchrun "${torchrun_args[@]}" ft_whisper_deepspeed.py "${script_args[@]}"
