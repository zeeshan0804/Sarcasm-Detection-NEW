import os
import pandas as pd
from typing import List, Tuple
import chardet
import numpy as np

def read_file_with_fallback_encoding(file_path: str) -> str:
    """
    Attempts to read a file with multiple encoding options and error handling.
    """
    # First try to detect the encoding
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        detected = chardet.detect(raw_data)
        detected_encoding = detected['encoding']

    # List of encodings to try, starting with the detected one
    encodings = [detected_encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read().strip()
        except (UnicodeDecodeError, TypeError):
            continue
            
    # If all encodings fail, try with error handling
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            return file.read().strip()
    except Exception as e:
        print(f"Warning: Could not read file {file_path}: {str(e)}")
        return ""

def create_dataset(sarcastic_dir: str, non_sarcastic_dir: str, output_file: str, random_seed: int = 42) -> None:
    """
    Creates a randomized dataset from sarcastic and non-sarcastic text files.
    
    Args:
        sarcastic_dir: Directory containing sarcastic text files
        non_sarcastic_dir: Directory containing non-sarcastic text files
        output_file: Path to save the output CSV file
        random_seed: Seed for reproducible randomization (default: 42)
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    data: List[Tuple[str, int]] = []
    
    # Process sarcastic files
    print("Processing sarcastic files...")
    for filename in os.listdir(sarcastic_dir):
        file_path = os.path.join(sarcastic_dir, filename)
        if os.path.isfile(file_path):
            text = read_file_with_fallback_encoding(file_path)
            if text:  # Only add non-empty texts
                data.append((text, 1))  # 1 for sarcastic
    
    # Process non-sarcastic files
    print("Processing non-sarcastic files...")
    for filename in os.listdir(non_sarcastic_dir):
        file_path = os.path.join(non_sarcastic_dir, filename)
        if os.path.isfile(file_path):
            text = read_file_with_fallback_encoding(file_path)
            if text:  # Only add non-empty texts
                data.append((text, 0))  # 0 for non-sarcastic
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['text', 'is_sarcastic'])
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Save with UTF-8 encoding and BOM for Excel compatibility
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # Print dataset statistics
    total_samples = len(df)
    sarcastic_samples = df['is_sarcastic'].sum()
    non_sarcastic_samples = total_samples - sarcastic_samples
    
    print("\nDataset Statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Sarcastic samples: {sarcastic_samples} ({(sarcastic_samples/total_samples)*100:.2f}%)")
    print(f"Non-sarcastic samples: {non_sarcastic_samples} ({(non_sarcastic_samples/total_samples)*100:.2f}%)")
    print(f"\nRandomized dataset saved to {output_file}")

# Example usage
sarcastic_dir = "/content/Sarcasm-Detection-NEW/data/sarcasm_IAC_v1_corpus/IAC_V1/IAC_V1/sarc"
non_sarcastic_dir = "/content/Sarcasm-Detection-NEW/data/sarcasm_IAC_v1_corpus/IAC_V1/IAC_V1/notsarc"
output_file = "sarcasm_dataset.csv"
create_dataset(sarcastic_dir, non_sarcastic_dir, output_file)
