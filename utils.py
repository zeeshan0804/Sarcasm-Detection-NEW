import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import re
import argparse

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.avg_word_length = self.calculate_avg_word_length()
        self.avg_sentence_length = self.calculate_avg_sentence_length()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Preprocess the text
        text = self.preprocess_text(text)

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

    def preprocess_text(self, text):
        # Lowercase the text
        text = text.lower()
        # Remove special characters and punctuation
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def calculate_avg_word_length(self):
        total_words = 0
        total_length = 0
        for text in self.texts:
            words = text.split()
            total_words += len(words)
            total_length += sum(len(word) for word in words)
        return total_length / total_words if total_words > 0 else 0

    def calculate_avg_sentence_length(self):
        total_sentences = len(self.texts)
        total_words = sum(len(text.split()) for text in self.texts)
        return total_words / total_sentences if total_sentences > 0 else 0

def prepare_bert_data(dataset_name, batch_size=16):
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_path = f'data/{dataset_name}/train.txt'
    test_path = f'data/{dataset_name}/test.txt'
    
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

    print("Average word length in train dataset:", train_dataset.avg_word_length)
    print("Average sentence length in train dataset:", train_dataset.avg_sentence_length)
    print("Average word length in test dataset:", test_dataset.avg_word_length)
    print("Average sentence length in test dataset:", test_dataset.avg_sentence_length)

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

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Prepare BERT data')
#     parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
#     parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    
#     args = parser.parse_args()
    
#     train_loader, test_loader, tokenizer = prepare_bert_data(
#         dataset_name=args.dataset,
#         batch_size=args.batch_size
#     )
    
#     # Example of accessing a batch
#     batch = next(iter(train_loader))
#     print("Input shape:", batch['input_ids'].shape)
#     print("Attention mask shape:", batch['attention_mask'].shape)
#     print("Labels shape:", batch['labels'].shape)