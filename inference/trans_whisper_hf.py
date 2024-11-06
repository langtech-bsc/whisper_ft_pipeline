import os
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from evaluate import load
import re


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

HF_DATASET="CATALAN_DATA_REPO/loading_script.py"

CHECKPOINT="/gpfs/projects/bsc88/speech/ASR/scripts/miscellaneous/ft_whisper_expes/ft_whisper_1n_ca/ok_CHECKPOINTS"

DATASET=load_dataset(HF_DATASET, split="test",cache_dir="CATALAN_DATA_REPO/CACHE",trust_remote_code=True)
#DATASET=load_dataset(HF_DATASET, split="validation",cache_dir="CATALAN_DATA_REPO/CACHE",trust_remote_code=True)

def normalize_trans(trans_in):
	#patron=r'<\|.*?\|>'
	trans_out = re.sub('<\|.*?\|>','',trans_in)
	trans_out = re.sub('\s+',' ',trans_out)
	trans_out = trans_out.strip()
	return trans_out
#ENDDEF

processor = WhisperProcessor.from_pretrained(CHECKPOINT)
model = WhisperForConditionalGeneration.from_pretrained(CHECKPOINT).to("cuda")


archivo_out = open("TRANS_OUT.trans",'w')

def map_to_trans(batch):
	audio = batch["audio"]
	clave = batch["audio_id"]
	input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
	
	with torch.no_grad():
		predicted_ids = model.generate(input_features.to("cuda"))[0]
	#ENDWITH
	
	transcription = processor.decode(predicted_ids)
	norm_trans = normalize_trans(transcription)
		
	linea_out=clave+" "+norm_trans
	archivo_out.write(linea_out+"\n")
	
	return batch
#ENDDEF

result = DATASET.map(map_to_trans)
########################################################################
archivo_out.close()