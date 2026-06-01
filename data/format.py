import csv
import os
import unicodedata
import re

def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove accents
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def get_audio_id(audio_path):
    # Extract filename without extension
    return os.path.splitext(os.path.basename(audio_path))[0]

def transform_tsv(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile, delimiter='\t')
        fieldnames = ['audio_id', 'language', 'normalized_text', 'absolute_path']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
        
        writer.writeheader()
        
        for row in reader:
            audio_path = row['audio_path']
            audio_id = get_audio_id(audio_path)
            
            new_row = {
                'audio_id': audio_id,
                'language': 'Catalan',
                'normalized_text': normalize_text(row['text']),
                'absolute_path': audio_path
            }
            writer.writerow(new_row)

# Usage
input_file = 'dev.tsv'
output_file = 'dev_out.tsv'
transform_tsv(input_file, output_file)

print("Transformation complete.")
