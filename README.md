# Whisper Fine-tuning

This repository contains code for fine-tuning OpenAI's Whisper model using the Accelerate library for distributed training. The implementation supports various features including Weights & Biases logging, checkpoint saving/loading, and custom data processing.

## Overview

The codebase consists of three main components:
- Argument parsing (`arg_parser.py`)
- Training script (`ft_whisper_accelerate.py`)
- Configuration settings (`settings.py`)
- Launch script (`launch_ft_whisper.sh`)

## Requirements

- PyTorch
- Transformers
- Accelerate
- Datasets
- Wandb (optional)
- jiwer
- tqdm

## Usage

### Basic Execution

The training can be launched using the provided SLURM script:

```bash
sbatch shs/launch_ft_whisper.sh
```

### Configuration
The main hyperparameters can be configured either in settings.py or passed as command-line arguments. Here are the key parameters and their purposes:
### Dataset Settings
```python
"hf_dataset": "/path/to/loading_script.py"  # HuggingFace dataset loading script
"cache_dir": "/path/to/cache"               # Cache directory for dataset
"hf_dataset_cache": "/path/to/cache"        # HuggingFace dataset cache
"hf_home": "/path/to/models"                # HuggingFace models cache
```
###Model Settings
```python
"whisper_model": "/path/to/whisper-model"   # Pre-trained Whisper model path
"language": "Catalan"                       # Target language
"task": "transcribe"                        # Task type (transcribe/translate)
"checkpoint_dir": "checkpoints"             # Directory for saving checkpoints
```
Training Hyperparameters
```python
"per_device_train_batch_size": 8            # Training batch size per GPU
"per_device_eval_batch_size": 8             # Evaluation batch size per GPU
"num_epochs": 5                             # Number of training epochs
"gradient_accumulation_steps": 1            # Steps before gradient update
"warmup_steps": 7                           # Learning rate warmup steps
"learning_rate": 1e-5                       # Initial learning rate
"weight_decay": 1e-4                        # Weight decay for regularization
"lr_scheduler_type": "linear"               # Learning rate scheduler type
```
Data Processing
```python
"max_input_length": 30.0                    # Maximum audio length in seconds
"do_lower_case": False                      # Convert text to lowercase
"do_remove_punctuation": False              # Remove punctuation from text
```
Weights & Biases Integration
```python
"use_weights_and_biases": False             # Enable W&B logging
"wandb_project": "whisper_finetuning"       # W&B project name
"wandb_entity": "bsc"                       # W&B team/organization
"wandb_run_name": "whisper_ft_es"           # W&B run name
"wandb_mode": "offline"                     # W&B mode (online/offline)
```

SLURM Configuration
The provided SLURM script (launch_ft_whisper.sh) is configured for distributed training with the following specifications:
```bash
#SBATCH --nodes=25                # Number of nodes
#SBATCH --ntasks=25              # Total number of tasks
#SBATCH --ntasks-per-node=1      # Tasks per node
#SBATCH --cpus-per-task=80       # CPUs per task
#SBATCH --gres=gpu:4             # GPUs per node
#SBATCH --time=24:00:00          # Maximum runtime
```

## Training Process

The training script:
1. Initializes the model, processor, and data loaders
2. Processes and prepares the dataset
3. Trains the model with specified hyperparameters
4. Evaluates performance using WER (Word Error Rate)
5. Saves checkpoints after each epoch
6. Logs metrics to W&B if enabled

## Monitoring

Progress can be monitored through:
- Terminal output with training loss and evaluation metrics
- Weights & Biases dashboard (if enabled)
- SLURM output logs in the `logs/` directory

## Checkpoints

Checkpoints are saved after each epoch in the specified checkpoint directory with the format:
