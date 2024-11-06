import os
import torch
import evaluate
import logging
import numpy as np
import gc
import math
from tqdm import tqdm
import jiwer

from accelerate import Accelerator
from accelerate.logging import get_logger

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import transformers
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration, 
    get_scheduler
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import datasets
from datasets import load_dataset, DatasetDict, Audio
from torch.utils.data import DataLoader

from arg_parser import ArgsParser

if os.environ.get('WANDB_DISABLED', 'false').lower() != 'true':
    import wandb

########################################################################
# Data Processing Classes and Functions
########################################################################

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be different lengths and need different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def prepare_dataset_wrapper(do_lower_case, do_remove_punctuation, processor, normalizer):
    def prepare_dataset(batch):
        # load and resample audio data to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        
        # compute input length of audio sample in seconds
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

        # optional pre-processing steps
        transcription = batch["sentence"]
        
        if do_lower_case:
            transcription = transcription.lower()
        if do_remove_punctuation:
            transcription = re.sub(punctuation_to_remove_regex, " ", transcription).strip()

        # encode target text to label ids
        batch["labels"] = processor.tokenizer(transcription).input_ids
        return batch
    return prepare_dataset

def get_audio_length_processor(max_input_length):
    def is_audio_in_length_range(length):
        return length < max_input_length
    return is_audio_in_length_range

def evaluation_loop(model, eval_dataloader, processor, normalizer, forced_decoder_ids, accelerator):
    model.eval()
    predictions = []
    references = []
    normalized_predictions = []
    normalized_references = []
    
    for batch in tqdm(eval_dataloader):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                generated_tokens = (
                    accelerator.unwrap_model(model).generate(
                        input_features=batch["input_features"],
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                predictions.extend(decoded_preds)
                references.extend(decoded_labels)
                normalized_predictions.extend([normalizer(pred).strip() for pred in decoded_preds])
                normalized_references.extend([normalizer(label).strip() for label in decoded_labels])
            del generated_tokens, labels, batch
        gc.collect()
    
    
    # Calculate WER using jiwer
    wer = jiwer.wer(references, predictions)
    normalized_wer = jiwer.wer(normalized_references, normalized_predictions)
    
    eval_metrics = {
        "eval/wer": wer * 100,  # Convert to percentage
        "eval/normalized_wer": normalized_wer * 100
    }

    return eval_metrics

def main():
    # Parse arguments
    parser = ArgsParser()
    parser()
    args = parser.arguments

    # Set environment variables based on offline mode
    if args.offline_mode:
        os.environ['TRANSFORMERS_OFFLINE'] = "1"
        os.environ['HF_DATASETS_OFFLINE'] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
    os.environ['HF_DATASETS_CACHE'] = args.hf_dataset_cache
    os.environ['HF_HOME'] = args.hf_home
    

    # Initialize accelerator
    accelerator_kwargs = {"gradient_accumulation_steps": args.gradient_accumulation_steps}
    if args.with_tracking:
        accelerator_kwargs["log_with"] = "all"
        accelerator_kwargs["project_dir"] = args.checkpoint_dir
    accelerator = Accelerator(**accelerator_kwargs)

    # Setup logging
    logger = get_logger(__name__, log_level="INFO")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Set logging verbosity
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Create checkpoint directory
    if accelerator.is_main_process:
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Initialize W&B if enabled
    if args.use_weights_and_biases and accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode
        )

    # Initialize processor and normalizer
    processor = WhisperProcessor.from_pretrained(
        args.whisper_model,
        language=args.language,
        task=args.task
    )
    normalizer = BasicTextNormalizer()

    # Prepare dataset processing functions
    prepare_dataset = prepare_dataset_wrapper(
        args.do_lower_case,
        args.do_remove_punctuation,
        processor,
        normalizer
    )
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Load datasets
    raw_datasets = DatasetDict()
    raw_datasets["train"] = load_dataset(
        args.hf_dataset,
        split="train",
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )
    raw_datasets["test"] = load_dataset(
        args.hf_dataset,
        split="test",
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )

    # Process datasets
    raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=16000))
    raw_datasets = raw_datasets.rename_column("normalized_text", "sentence")
    
    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        remove_columns=["audio_id", "absolute_path"],
        num_proc=args.preprocessing_num_workers
    ).with_format("torch")

    # Filter by audio length
    is_audio_in_length_range = get_audio_length_processor(args.max_input_length)
    vectorized_datasets["train"] = vectorized_datasets["train"].filter(
        is_audio_in_length_range,
        input_columns=["input_length"],
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        vectorized_datasets["train"],
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    eval_dataloader = DataLoader(
        vectorized_datasets["test"],
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Initialize model
    if args.load_checkpoint and args.resume_checkpoint:
        logger.info(f"Loading model from checkpoint: {args.resume_checkpoint}")
        model = WhisperForConditionalGeneration.from_pretrained(args.resume_checkpoint)
        accelerator.load_state(args.resume_checkpoint)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(args.whisper_model)

    # Configure model generation settings
    model.config.forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(
        language=args.language,
        task=args.task
    )
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(
        language=args.language,
        task=args.task
    )
    model.generation_config.suppress_tokens = []

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )



    # Calculate training steps
    # Get number of devices
    num_devices = accelerator.num_processes
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / (args.gradient_accumulation_steps * num_devices))
    max_train_steps = num_update_steps_per_epoch * args.num_epochs

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Prepare for training
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Initialize metric
    wer_metric = jiwer.wer

    # Training loop
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(
        range(max_train_steps),
        disable=not accelerator.is_local_main_process
    )

    # Training state
    global_step = 0
    best_metric = None
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)

    accelerator.wait_for_everyone()
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                if args.use_weights_and_biases:
                    wandb.log({"train/loss": loss.detach().float()})
                
                if step % 100 == 0:
                    logger.info(f"Epoch {epoch}, Step {step}: Loss = {loss.item():.4f}")
                
            global_step += 1
            progress_bar.update(1)
            total_loss += loss.detach().float()

        # Wait for all processes to complete the epoch
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            logger.info(f"Completed epoch {epoch} with {step + 1} steps")

		# Save the model after each epoch
        if accelerator.is_main_process:
            accelerator.save_state(
                    os.path.join(args.checkpoint_dir, f"checkpoint-{epoch}"),
                    safe_serialization=True
                )
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                os.path.join(args.checkpoint_dir, f"checkpoint-{epoch}"),
                is_main_process=accelerator.is_main_process
            )
            processor.tokenizer.save_pretrained(
                os.path.join(args.checkpoint_dir, f"checkpoint-{epoch}")
            )

        # Wait for saving to complete before evaluation
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            # Evaluate at the end of each epoch
            eval_metrics = evaluation_loop(
                model,
                eval_dataloader,
                processor,
                normalizer,
                forced_decoder_ids,
                accelerator
            )
            logger.info(f"Epoch {epoch} evaluation: {eval_metrics}")

        accelerator.wait_for_everyone()
        
        
        if args.use_weights_and_biases:
            wandb.log(eval_metrics)

        # Wait for evaluation to complete before starting next epoch
        accelerator.wait_for_everyone()

    if args.use_weights_and_biases:
        wandb.finish()

if __name__ == "__main__":
    main()