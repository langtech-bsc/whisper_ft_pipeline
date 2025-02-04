TRAIN_DEFAULT_SETTINGS = {

    # Model settings
    #MODEL PATH 
    "whisper_model": "/gpfs/projects/bsc88/speech/research/models/hf_models/whisper-large-v3",
    "resume_checkpoint": None,
    "load_checkpoint": False,
    "language": "Catalan",
    "task": "transcribe",
    "checkpoint_dir": "checkpoints",
    
    # Training hyperparameters
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 10,
    
    # Secondary parameters
    "preprocessing_num_workers": 16,
    "learning_rate": 1e-5,
    "weight_decay": 1e-4,
    
    # Data processing settings
    "do_lower_case": False,
    "do_remove_punctuation": False,
    
    "logging_dir": "./logs",
    
    # Wandb settings
    "use_weights_and_biases": False,

    # Deepspeed config
    "deepspeed": "/gpfs/projects/bsc88/speech/research/scripts/Lucas/DeepSpeed_Whisper/configs/03_ds_type3_config_autombs.json", 
}