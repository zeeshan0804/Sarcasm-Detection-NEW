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
            'labels': torch.tensor(label, dtype=torch.long),
            'raw_text': text
        }

def prepare_bert_data(train_path, test_path, batch_size=16):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load and analyze train data
    train_texts, train_labels = [], []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            text, label = line.rsplit(' ', 1)
            train_texts.append(text)
            train_labels.append(int(label))
    
    # Calculate class weights for balanced loss
    train_labels_np = np.array(train_labels)
    class_counts = np.bincount(train_labels_np)
    total_samples = len(train_labels_np)
    class_weights = torch.FloatTensor(total_samples / (len(class_counts) * class_counts))
    
    print("\nClass distribution in training:")
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {count} samples ({count/total_samples*100:.2f}%)")
    
    # Load test data
    test_texts, test_labels = [], []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            text, label = line.rsplit(' ', 1)
            test_texts.append(text)
            test_labels.append(int(label))
            
    print("Train data size:", len(train_texts))
    print("Test data size:", len(test_texts))

    train_dataset = SarcasmDataset(train_texts, train_labels, tokenizer)
    test_dataset = SarcasmDataset(test_texts, test_labels, tokenizer)
    
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
            'raw_text': [item['raw_text'] for item in batch]
        }
    
    # Calculate sample weights for balanced sampling
    sample_weights = [1/class_counts[label] for label in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use sampler instead of shuffle
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader, tokenizer, class_weights

# if __name__ == "__main__":
#     train_loader, test_loader, tokenizer = prepare_bert_data(
#         'data/riloff/train.txt',
#         'data/riloff/test.txt',
#         batch_size=16
#     )
    
#     batch = next(iter(train_loader))
#     print("Input shape:", batch['input_ids'].shape)
#     print("Attention mask shape:", batch['attention_mask'].shape)
#     print("Labels shape:", batch['labels'].shape)
#     print("Raw text length:", len(batch['raw_text']))