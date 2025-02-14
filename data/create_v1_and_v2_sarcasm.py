import os
import pandas as pd

def create_dataset(sarcastic_dir, non_sarcastic_dir, output_file):
    data = []
    
    # Process sarcastic files
    for filename in os.listdir(sarcastic_dir):
        file_path = os.path.join(sarcastic_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            data.append((text, 1))  # 1 for sarcastic
    
    # Process non-sarcastic files
    for filename in os.listdir(non_sarcastic_dir):
        file_path = os.path.join(non_sarcastic_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            data.append((text, 0))  # 0 for non-sarcastic
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data, columns=['text', 'is_sarcastic'])
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Dataset saved to {output_file}")

# Example usage
sarcastic_dir = r"C:\Users\Hassan\Desktop\Urdu_SD\data-sarc-sample\sarc"
non_sarcastic_dir = r"C:\Users\Hassan\Desktop\Urdu_SD\data-sarc-sample\notsarc"
output_file = "sarcasm_dataset_IAC_V1.csv"
create_dataset(sarcastic_dir, non_sarcastic_dir, output_file)