class SarcasmDataset(Dataset):
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
            'raw_text': text  # Add this line
        }

def prepare_bert_data(train_path, test_path, batch_size=16):
    # Existing code remains same
    train_dataset = SarcasmDataset(train_texts, train_labels, tokenizer)
    test_dataset = SarcasmDataset(test_texts, test_labels, tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: {  # Add collate_fn
            'input_ids': torch.stack([item['input_ids'] for item in x]),
            'attention_mask': torch.stack([item['attention_mask'] for item in x]),
            'labels': torch.stack([item['labels'] for item in x]),
            'raw_text': [item['raw_text'] for item in x]
        }
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: {  # Add collate_fn
            'input_ids': torch.stack([item['input_ids'] for item in x]),
            'attention_mask': torch.stack([item['attention_mask'] for item in x]),
            'labels': torch.stack([item['labels'] for item in x]),
            'raw_text': [item['raw_text'] for item in x]
        }
    )
    return train_loader, test_loader, tokenizer
