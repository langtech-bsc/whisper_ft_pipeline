import os
import torch
import logging
import numpy as np
import re
from datasets import load_dataset, DatasetDict, Audio, load_from_disk
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    get_scheduler,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from arg_parser import ArgsParser

from transformers import TrainerCallback


class SavePreprocessorCallback(TrainerCallback):
    def __init__(self, processor, save_dir):
        self.processor = processor
        self.save_dir = save_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        # Create epoch-specific directory
        epoch_dir = os.path.join(self.save_dir, f"checkpoint-{state.epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Save processor
        self.processor.save_pretrained(epoch_dir)


# Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# Dataset preparation wrapper
def prepare_dataset_wrapper(do_lower_case, do_remove_punctuation, processor, normalizer):
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        transcription = batch["sentence"]
        if do_lower_case:
            transcription = transcription.lower()
        if do_remove_punctuation:
            transcription = re.sub(r"[!?,.:;\'\"\(\)]", " ", transcription).strip()

        batch["labels"] = processor.tokenizer(transcription).input_ids
        return batch

    return prepare_dataset

def is_audio_in_length_range(length, max_length=30.0):
    return length < max_length


def main():

    parser = ArgsParser()
    parser()
    args = parser.arguments

    # Logging setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("WhisperFineTuning")

    # Initialize processor
    processor = WhisperProcessor.from_pretrained(
        args.whisper_model, language=args.language, task=args.task
    )
    normalizer = BasicTextNormalizer()

    # Load dataset
    processed_dataset_path = os.path.join(args.preprocessed_data_cache, "processed_dataset")
    if os.path.exists(processed_dataset_path) and not args.reprocess_data:
        logger.info("Loading preprocessed dataset...")
        vectorized_datasets = load_from_disk(processed_dataset_path)
    else:
        logger.info("Processing dataset...")
        raw_datasets = DatasetDict({
            "train": load_dataset(args.hf_dataset, split="train", cache_dir=args.hf_dataset_cache,trust_remote_code=True),
            "test": load_dataset(args.hf_dataset, split="test", cache_dir=args.hf_dataset_cache,trust_remote_code=True),
        })
        raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=16000))
        raw_datasets = raw_datasets.rename_column("normalized_text", "sentence")

        prepare_dataset = prepare_dataset_wrapper(
            args.do_lower_case, args.do_remove_punctuation, processor, normalizer
        )

        vectorized_datasets = raw_datasets.map(
            prepare_dataset, num_proc=args.preprocessing_num_workers, remove_columns=["audio_id", "absolute_path"]
        )
        
        # Filter long sequences
        vectorized_datasets["train"] = vectorized_datasets["train"].filter(
            is_audio_in_length_range,
            input_columns=["input_length"]
        )
        
        vectorized_datasets.save_to_disk(processed_dataset_path)

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Initialize callback
    save_processor_callback = SavePreprocessorCallback(
        processor=processor,
        save_dir=args.checkpoint_dir
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,        
        gradient_accumulation_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        warmup_ratio=0.1,        
        max_grad_norm=1.0,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        save_total_limit=None,
        load_best_model_at_end=True,
        fp16=True,
        deepspeed=args.deepspeed,
        report_to="wandb" if args.use_weights_and_biases else None,
    )

    # Initialize model
    model = WhisperForConditionalGeneration.from_pretrained(args.whisper_model)

    # Trainer with callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["test"],
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        callbacks=[save_processor_callback]  # Add the callback here
    )

    # Train
    trainer.train()

    # Save final model and processor
    trainer.save_model(args.checkpoint_dir)
    processor.save_pretrained(args.checkpoint_dir)

if __name__ == "__main__":
    main()
