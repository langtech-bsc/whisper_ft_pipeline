import os
import torch
import numpy as np
import wandb
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import get_peft_model, LoraConfig
import jiwer

# Set up environment and wandb
os.environ['WANDB_MODE'] = 'offline'
wandb.init(project="whisper-lora-catalan", name="offline_run")

# Model and dataset configuration
model_name_or_path = "/gpfs/projects/bsc88/speech/research/models/hf_models/whisper-tiny"
dataset_name = "/gpfs/projects/bsc88/speech/research/scripts/Lucas/SoftPrompts/loading_script_lt400.py"
language = "catalan"
task = "transcribe"

# Load and prepare dataset
data = DatasetDict()
data["train"] = load_dataset(dataset_name, name="loading_script", split="train+validation", trust_remote_code=True)
data["test"] = load_dataset(dataset_name, name="loading_script", split="test", trust_remote_code=True)

# Initialize processor
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

# Prepare dataset
data = data.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["normalized_text"]).input_ids
    return batch

data = data.map(prepare_dataset, remove_columns=data.column_names["train"], num_proc=4)

# Data collator
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



data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# class SavePeftModelCallback(TrainerCallback):
#     def on_save(
#         self,
#         args: TrainingArguments,
#         state: TrainerState,
#         control: TrainerControl,
#         **kwargs,
#     ):
#         checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

#         peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
#         kwargs["model"].save_pretrained(peft_model_path)

#         pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
#         if os.path.exists(pytorch_model_path):
#             os.remove(pytorch_model_path)
#         return control

# Load model and apply LoRA
model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

lora_config = LoraConfig(
    r=8, 
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.01
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Compute metrics
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer_metric = jiwer.wer(label_str, pred_str)

    return {"wer": wer_metric}

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-lora-catalan",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=500,
    max_steps=100,
    fp16=True,  
    logging_steps=25,
    save_steps=100,
    eval_steps=100,
    evaluation_strategy="steps",
    predict_with_generate=True,
    report_to=["wandb"],
    generation_max_length=225,
    save_total_limit=3,
    load_best_model_at_end=True,
)

# Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# Train the model
model.config.use_cache = False
trainer.train()

# Save the model
peft_model_id = "/gpfs/projects/bsc88/speech/research/scripts/Lucas/Lora/whisper-tiny"
model.save_pretrained(peft_model_id)