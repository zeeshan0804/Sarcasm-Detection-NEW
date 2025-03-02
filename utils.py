import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
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

        # Encode the text using ModernBERT tokenizer
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

def prepare_bert_data(train_path, test_path, batch_size=16):
    # Update the function to use direct paths instead of dataset name
    tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
    
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

    # Ensure labels are within valid range
    train_labels = [label for label in train_labels if label in [0, 1]]
    test_labels = [label for label in test_labels if label in [0, 1]]

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

def evaluate(model, test_loader, criterion, device, zero_division=0):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=zero_division)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=zero_division)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=zero_division)
    
    return avg_loss, accuracy, f1, precision, recall

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
