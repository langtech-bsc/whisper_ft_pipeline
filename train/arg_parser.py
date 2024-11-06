import argparse
from settings import TRAIN_DEFAULT_SETTINGS

class ArgsParser:
    def __init__(self) -> None:
        self.initialize_parser()

    def initialize_parser(self):
        self.parser = argparse.ArgumentParser(
            description='Whisper fine-tuning script'
        )

    def add_parser_args(self):
        # Dataset arguments
        self.parser.add_argument(
            "--hf_dataset",
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["hf_dataset"],
            help="Path to the HuggingFace dataset loading script"
        )
        self.parser.add_argument(
            "--cache_dir",
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["cache_dir"],
            help="Cache directory for the dataset"
        )

        # Model arguments
        self.parser.add_argument(
            "--whisper_model",
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["whisper_model"],
            help="Path or name of the Whisper model to use"
        )
        self.parser.add_argument(
            "--resume_checkpoint",
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["resume_checkpoint"],
            help="Path to checkpoint to resume training from"
        )
        self.parser.add_argument(
            "--load_checkpoint",
            action="store_true",
            default=TRAIN_DEFAULT_SETTINGS["load_checkpoint"],
            help="Whether to load from checkpoint"
        )
        self.parser.add_argument(
            "--language",
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["language"],
            help="Language for the model"
        )
        self.parser.add_argument(
            "--task",
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["task"],
            help="Task for the model (transcribe or translate)"
        )
        self.parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["checkpoint_dir"],
            help="Directory to save checkpoints"
        )

        # Training hyperparameters
        self.parser.add_argument(
            "--per_device_train_batch_size",
            type=int,
            default=TRAIN_DEFAULT_SETTINGS["per_device_train_batch_size"],
            help="Training batch size per device"
        )
        self.parser.add_argument(
            "--per_device_eval_batch_size",
            type=int,
            default=TRAIN_DEFAULT_SETTINGS["per_device_eval_batch_size"],
            help="Evaluation batch size per device"
        )
        self.parser.add_argument(
            "--num_epochs",
            type=int,
            default=TRAIN_DEFAULT_SETTINGS["num_epochs"],
            help="Number of training epochs"
        )
        self.parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=TRAIN_DEFAULT_SETTINGS["gradient_accumulation_steps"],
            help="Number of gradient accumulation steps"
        )
        self.parser.add_argument(
            "--warmup_steps",
            type=int,
            default=TRAIN_DEFAULT_SETTINGS["warmup_steps"],
            help="Number of warmup steps"
        )

        # Secondary parameters
        self.parser.add_argument(
            "--preprocessing_num_workers",
            type=int,
            default=TRAIN_DEFAULT_SETTINGS["preprocessing_num_workers"],
            help="Number of preprocessing workers"
        )
        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=TRAIN_DEFAULT_SETTINGS["num_workers"],
            help="Number of data loading workers"
        )
        self.parser.add_argument(
            "--learning_rate",
            type=float,
            default=TRAIN_DEFAULT_SETTINGS["learning_rate"],
            help="Learning rate"
        )
        self.parser.add_argument(
            "--weight_decay",
            type=float,
            default=TRAIN_DEFAULT_SETTINGS["weight_decay"],
            help="Weight decay"
        )
        self.parser.add_argument(
            "--lr_scheduler_type",
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["lr_scheduler_type"],
            help="Type of learning rate scheduler"
        )

        # Data processing settings
        self.parser.add_argument(
            "--max_input_length",
            type=float,
            default=TRAIN_DEFAULT_SETTINGS["max_input_length"],
            help="Maximum input length in seconds"
        )
        self.parser.add_argument(
            "--do_lower_case",
            action="store_true",
            default=TRAIN_DEFAULT_SETTINGS["do_lower_case"],
            help="Whether to lowercase the text"
        )
        self.parser.add_argument(
            "--do_remove_punctuation",
            action="store_true",
            default=TRAIN_DEFAULT_SETTINGS["do_remove_punctuation"],
            help="Whether to remove punctuation"
        )

        # Training features
        self.parser.add_argument(
            "--load_best_model",
            action="store_true",
            default=TRAIN_DEFAULT_SETTINGS["load_best_model"],
            help="Load the best model after training"
        )
        self.parser.add_argument(
            "--with_tracking",
            action="store_true",
            default=TRAIN_DEFAULT_SETTINGS["with_tracking"],
            help="Enable training tracking"
        )

        # Wandb arguments
        self.parser.add_argument(
            "--use_weights_and_biases",
            action="store_true",
            default=TRAIN_DEFAULT_SETTINGS["use_weights_and_biases"],
            help="Whether to use Weights & Biases for logging"
        )
        self.parser.add_argument(
            "--wandb_project",
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["wandb_project"],
            help="Weights & Biases project name"
        )
        self.parser.add_argument(
            "--wandb_entity",
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["wandb_entity"],
            help="Weights & Biases entity name"
        )
        self.parser.add_argument(
            "--wandb_run_name",
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["wandb_run_name"],
            help="Weights & Biases run name"
        )
        self.parser.add_argument(
            "--offline_mode",
            action="store_true",
            default=TRAIN_DEFAULT_SETTINGS["offline_mode"],
            help="Server offline_mode"
        )
        self.parser.add_argument(
            "--wandb_mode",
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["wandb_mode"],
            choices=["online", "offline"],
            help="Weights & Biases mode (online/offline)"
        )
        self.parser.add_argument(
            "--hf_dataset_cache",
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["hf_dataset_cache"],
            help="hf_dataset_cache"
        )
        self.parser.add_argument(
            "--hf_home",
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["hf_home"],
            help="hf_home"
        )

    def main(self):
        self.add_parser_args()
        self.arguments = self.parser.parse_args()
    
    def __call__(self):
        self.main()