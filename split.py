import json
import random

# File paths
input_path = 'data/headline/Sarcasm_Headlines_Dataset.json'
input_path_2 = 'data/headline/Sarcasm_Headlines_Dataset_v2.json'
train_output_path = 'data/headline/train.txt'
test_output_path = 'data/headline/test.txt'

# Load data line by line from the JSON lines file
data = []
with open(input_path, 'r') as file:
    for line in file:
        entry = json.loads(line.strip())  # Parse each line as JSON
        data.append(entry)

with open(input_path_2, 'r') as file:
    for line in file:
        entry = json.loads(line.strip())  # Parse each line as JSON
        data.append(entry)

random.shuffle(data)
split_index = int(0.9 * len(data))
train_data = data[:split_index]
test_data = data[split_index:]

with open(train_output_path, 'w') as train_file:
    for entry in train_data:
        train_file.write(f"{entry['headline']} {entry['is_sarcastic']}\n")

with open(test_output_path, 'w') as test_file:
    for entry in test_data:
        test_file.write(f"{entry['headline']} {entry['is_sarcastic']}\n")

print("Data has been split and saved to 'train.txt' and 'test.txt'")
