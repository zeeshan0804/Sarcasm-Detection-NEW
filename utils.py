import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
import numpy as np

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, glove_embeddings, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.glove_embeddings = glove_embeddings
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

        # Encode the text using GloVe embeddings
        glove_embedding = self.get_glove_embedding(text)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'glove_embedding': torch.tensor(glove_embedding, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def get_glove_embedding(self, text):
        words = text.split()
        embedding = np.zeros((self.max_length, self.glove_embeddings.shape[1]))
        for i, word in enumerate(words[:self.max_length]):
            if word in self.glove_embeddings:
                embedding[i] = self.glove_embeddings[word]
        return embedding

def load_glove_embeddings(glove_path):
    embeddings = {}
    with open(glove_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def prepare_bert_data(train_path, test_path, glove_path, batch_size=16):
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    glove_embeddings = load_glove_embeddings(glove_path)
    
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
    train_dataset = SarcasmDataset(train_texts, train_labels, tokenizer, glove_embeddings)
    test_dataset = SarcasmDataset(test_texts, test_labels, tokenizer, glove_embeddings)

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