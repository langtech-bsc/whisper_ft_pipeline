import torch
import gc
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from peft import PeftModel, PeftConfig
import jiwer
import random
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Load the fine-tuned model
peft_model_id = "/gpfs/projects/bsc88/speech/research/scripts/Lucas/Lora/whisper-large-v2-lot9-catalan/checkpoint-750"
print(f"loading... {peft_model_id}")
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path, device_map="auto"
)
model = PeftModel.from_pretrained(model, peft_model_id)
model.config.use_cache = True

# Load the processor
processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, language="catalan", task="transcribe")

# Load and prepare the dataset
dataset_name = "/gpfs/projects/bsc88/speech/research/scripts/Lucas/SoftPrompts/loading_script.py"
data = load_dataset(dataset_name, name="loading_script", split="train", trust_remote_code=True)
data = data.cast_column("audio", Audio(sampling_rate=16000))

# Prepare dataset function
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["normalized_text"]).input_ids
    return batch

data = data.map(prepare_dataset, remove_columns=data.column_names, num_proc=4)

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

# Prepare the test dataset
eval_dataloader = DataLoader(data, batch_size=8, collate_fn=data_collator)

# Set up evaluation
forced_decoder_ids = processor.get_decoder_prompt_ids(language="catalan", task="transcribe")
normalizer = BasicTextNormalizer()

# Evaluation loop
predictions = []
references = []
normalized_predictions = []
normalized_references = []

model.eval()
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.amp.autocast("cuda"):
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].to("cuda"),
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
eval_metrics = {"eval/wer": wer, "eval/normalized_wer": normalized_wer}

print(f"WER: {wer}")
print(f"Normalized WER: {normalized_wer}")
print(f"Evaluation metrics: {eval_metrics}")

# Print sample predictions and ground truth pairs
print("\nSample Predictions and Ground Truth Pairs:")
num_samples = min(5, len(predictions))  # Print up to 5 samples
sample_indices = random.sample(range(len(predictions)), num_samples)

for idx in sample_indices:
    print(f"\nSample {idx + 1}:")
    print(f"Prediction: {predictions[idx]}")
    print(f"Ground Truth: {references[idx]}")
    print(f"Normalized Prediction: {normalized_predictions[idx]}")
    print(f"Normalized Ground Truth: {normalized_references[idx]}")
    print("-" * 50)