import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
import numpy as np

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Encode the text using BERT tokenizer
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_bert_data(train_path, test_path, batch_size=16):
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load train data
    train_texts, train_labels = [], []
    with open(train_path, 'r') as f:
        for line in f:
            text, label = line.rsplit(' ', 1)
            train_texts.append(text)
            train_labels.append(int(label))
    
    # Load test data
    test_texts, test_labels = [], []
    with open(test_path, 'r') as f:
        for line in f:
            text, label = line.rsplit(' ', 1)
            test_texts.append(text)
            test_labels.append(int(label))
            
    print("Train data size:", len(train_texts))
    print("Test data size:", len(test_texts))
    # print("Train label distribution:", pd.Series(train_labels).value_counts())
    # print("Test label distribution:", pd.Series(test_labels).value_counts())

    # Create datasets
    train_dataset = SarcasmDataset(train_texts, train_labels, tokenizer)
    test_dataset = SarcasmDataset(test_texts, test_labels, tokenizer)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, tokenizer

# # Usage
# if __name__ == "__main__":
#     train_loader, test_loader, tokenizer = prepare_bert_data(
#         'data/riloff/train.txt',
#         'data/riloff/test.txt',
#         batch_size=16
#     )
    
#     # Example of accessing a batch
#     batch = next(iter(train_loader))
#     print("Input shape:", batch['input_ids'].shape)
#     print("Attention mask shape:", batch['attention_mask'].shape)
#     print("Labels shape:", batch['labels'].shape)