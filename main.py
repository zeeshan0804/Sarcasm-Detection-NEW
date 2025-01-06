import torch
import torch.nn as nn
from transformers import BertModel
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import SarcasmDataset, prepare_bert_data
from transformers import AutoModel, AutoTokenizer
import os
import argparse

class Attention(nn.Module):
    def __init__(self, lstm_hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(lstm_hidden_size * 2, 1)

    def forward(self, lstm_out):
        # Apply attention mechanism
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        return context_vector

class SarcasmDetector(nn.Module):
    def __init__(self, dropout_rate=0.3, freeze_bert=True):
        super(SarcasmDetector, self).__init__()
        
        # ModernBERT layer with frozen parameters
        self.bert = AutoModel.from_pretrained('answerdotai/ModernBERT-base')
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.bert_dim = self.bert.config.hidden_size  # Dynamically get hidden size
        
        # Architecture parameters
        self.cnn_out_channels = 256
        self.lstm_hidden_size = 128
        self.dense_hidden_size = 64
        
        # CNN layer
        self.conv1d = nn.Conv1d(
            in_channels=self.bert_dim,
            out_channels=self.cnn_out_channels,
            kernel_size=3,
            padding=1
        )
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Attention layer
        self.attention = Attention(self.lstm_hidden_size)
        
        # Dense layers
        self.dense1 = nn.Linear(self.lstm_hidden_size * 2, self.dense_hidden_size)
        self.dense2 = nn.Linear(self.dense_hidden_size, 2)
        
        # Regularization and activation layers
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        # BERT embedding layer (frozen if specified)
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embeddings = bert_output.last_hidden_state
        
        # Debugging print statements
        print("BERT embeddings:", bert_embeddings)
        
        # CNN feature extraction
        cnn_in = bert_embeddings.permute(0, 2, 1)
        cnn_out = self.relu(self.conv1d(cnn_in))
        
        # Debugging print statements
        print("CNN output:", cnn_out)
        
        lstm_in = cnn_out.permute(0, 2, 1)
        
        # BiLSTM sequence learning
        lstm_out, _ = self.lstm(lstm_in)
        
        # Debugging print statements
        print("LSTM output:", lstm_out)
        
        # Apply attention
        context_vector = self.attention(lstm_out)
        
        # Debugging print statements
        print("Context vector:", context_vector)
        
        # Classification layers
        x = self.dense1(context_vector)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.dense2(x)
        predictions = self.softmax(logits)
        
        # Debugging print statements
        print("Logits:", logits)
        print("Predictions:", predictions)
        
        return predictions

def train_epoch(model, train_loader, optimizer, criterion, device, clip_value=1.0):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print("NaN loss encountered")
            print("Input IDs:", input_ids)
            print("Attention Mask:", attention_mask)
            print("Labels:", labels)
            print("Outputs:", outputs)
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate Sarcasm Detector')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--model_path', type=str, default='sarcasm_detector_model.pth', help='Path to save the trained model')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    train_loader, test_loader, tokenizer = prepare_bert_data(
        dataset_name=args.dataset,
        batch_size=args.batch_size
    )
    
    model_params = {
        'dropout_rate': 0.3,
        'freeze_bert': True
    }
    
    training_params = {
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size
    }
    
    model = SarcasmDetector(**model_params).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_params['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    if os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path))
        model.to(device)
        
        # Evaluate the model
        test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate(model, test_loader, criterion, device, zero_division=0)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        print(f'Test F1 Score: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}')
    
    print("Training model...")
    for epoch in range(training_params['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{training_params["num_epochs"]}, Train Loss: {train_loss:.4f}')
        
        test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate(model, test_loader, criterion, device, zero_division=0)
        print(f'Epoch {epoch+1}/{training_params["num_epochs"]}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

        scheduler.step(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.model_path)
            print(f"Model saved to {args.model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    print("Training complete!")

    model.load_state_dict(torch.load(args.model_path))
    
    test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate(model, test_loader, criterion, device, zero_division=0)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}')
