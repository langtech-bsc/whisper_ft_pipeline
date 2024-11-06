import argparse
import os 
import soundfile as sf
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm

def check_file(file_path):
    try:
        sf.read(file_path)
        return None  # File is not corrupted
    except Exception as e:
        return file_path  # File is corrupted

def process_files(input_tsv, output_file, n_jobs=os.cpu_count() - 1):
    print("Reading TSV file...")
    df = pd.read_csv(input_tsv, sep='\t')
    audio_files = df['absolute_path'].tolist()
    print(f"Found {len(audio_files)} files...")

    print("Checking for corrupted files...")
    with Pool(processes=n_jobs) as p:
        results = list(tqdm(p.imap(check_file, audio_files), total=len(audio_files)))

    corrupted_files = [file for file in results if file is not None]

    # Save corrupted files list to output file
    with open(output_file, 'w') as f:
        for file in corrupted_files:
            f.write(f"{file}\n")

    print(f"Done! Found {len(corrupted_files)} corrupted files.")
    print(f"List of corrupted files saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Check for corrupted audio files from a TSV input file\n\n
                       Example run:
                            python check_corrupted_files.py
                                --input_tsv /path/to/input.tsv
                                --output_file /path/to/corrupted_files.txt
                                --n_jobs 24
                    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--input_tsv",
        type=str,
        required=True,
        help="Path to the input TSV file containing audio file information",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output file to store the list of corrupted files",
    )

    parser.add_argument(
        "--n_jobs", type=int, default=None, help="Number of threads to use, by default it uses all cores"
    )

    args = parser.parse_args()

    process_files(input_tsv=args.input_tsv, output_file=args.output_file, n_jobs=args.n_jobs)