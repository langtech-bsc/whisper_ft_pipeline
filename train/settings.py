TRAIN_DEFAULT_SETTINGS = {
    # Dataset settings
    #PATH TO THE LOADER FILE
    "hf_dataset": "/gpfs/projects/bsc88/speech/research/scripts/Lucas/SoftPrompts/loading_script.py",
    #"/gpfs/projects/bsc88/speech/research/scripts/Lucas/SoftPrompts/loading_script.py", 
    #PATH OF THE DATA LOADER CACHE
    "cache_dir": "/gpfs/projects/bsc88/speech/research/scripts/Lucas/Whisper_pipeline/cache",
    #HF DATA CACHE
    "hf_dataset_cache": "/gpfs/projects/bsc88/speech/research/scripts/Lucas/Whisper_pipeline/cache",
    #HF MODEL CACHE
    "hf_home": "/gpfs/projects/bsc88/speech/research/scripts/Lucas/Whisper_pipeline/cache/model_cache",
    
    
    # Model settings
    #MODEL PATH 
    "whisper_model": "/gpfs/projects/bsc88/speech/research/models/hf_models/whisper-large-v3",
    "resume_checkpoint": None,
    "load_checkpoint": False,
    "language": "Catalan",
    "task": "transcribe",
    "checkpoint_dir": "checkpoints_full_cat",
    
    # Training hyperparameters
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "num_epochs": 2,
    "gradient_accumulation_steps": 1,
    "warmup_steps": 1000,
    
    # Secondary parameters
    "preprocessing_num_workers": 16,
    "num_workers": 16,
    "pin_memory": True,
    "learning_rate": 1e-5,
    "weight_decay": 1e-4,
    "lr_scheduler_type": "linear",
    
    # Data processing settings
    "max_input_length": 30.0,
    "do_lower_case": False,
    "do_remove_punctuation": False,
    
    # Training features
    "load_best_model": True,
    "with_tracking": True,
    
    # Environment settings
    "offline_mode": True,
    
    # Wandb settings
    "use_weights_and_biases": False,
    "wandb_project": "whisper_finetuning",
    "wandb_entity": "bsc",
    "wandb_run_name": "whisper_ft_es",
    "wandb_mode": "offline",
}