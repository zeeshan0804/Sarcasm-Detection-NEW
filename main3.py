import torch
import torch.nn as nn
from transformers import BertModel
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import SarcasmDataset, prepare_bert_data
import os

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
    def __init__(self, bert_dim=768, glove_dim=300, cnn_out_channels=256, lstm_hidden_size=128, dense_hidden_size=64, dropout_rate=0.3, freeze_bert=True):
        super(SarcasmDetector, self).__init__()
        
        # BERT layer with frozen parameters
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.bert_dim = bert_dim
        
        # GloVe embedding parameters
        self.glove_dim = glove_dim
        
        # Architecture parameters
        self.cnn_out_channels = cnn_out_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.dense_hidden_size = dense_hidden_size
        
        # CNN layer for BERT
        self.bert_conv1d = nn.Conv1d(
            in_channels=self.bert_dim,
            out_channels=self.cnn_out_channels,
            kernel_size=3,
            padding=1
        )
        
        # CNN layer for GloVe
        self.glove_conv1d = nn.Conv1d(
            in_channels=self.glove_dim,
            out_channels=self.cnn_out_channels,
            kernel_size=3,
            padding=1
        )
        
        # BiLSTM layer for BERT
        self.bert_lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # BiLSTM layer for GloVe
        self.glove_lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Attention layers
        self.bert_attention = Attention(self.lstm_hidden_size)
        self.glove_attention = Attention(self.lstm_hidden_size)
        
        # Dense layers
        self.dense1 = nn.Linear(self.lstm_hidden_size * 4, self.dense_hidden_size)
        self.dense2 = nn.Linear(self.dense_hidden_size, 2)
        
        # Regularization and activation layers
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, glove_embedding):
        # BERT embedding layer (frozen)
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embeddings = bert_output.last_hidden_state
        
        # CNN feature extraction for BERT
        bert_cnn_in = bert_embeddings.permute(0, 2, 1)
        bert_cnn_out = self.relu(self.bert_conv1d(bert_cnn_in))
        bert_lstm_in = bert_cnn_out.permute(0, 2, 1)
        
        # CNN feature extraction for GloVe
        glove_cnn_in = glove_embedding.permute(0, 2, 1)
        glove_cnn_out = self.relu(self.glove_conv1d(glove_cnn_in))
        glove_lstm_in = glove_cnn_out.permute(0, 2, 1)
        
        # BiLSTM sequence learning for BERT
        bert_lstm_out, _ = self.bert_lstm(bert_lstm_in)
        
        # BiLSTM sequence learning for GloVe
        glove_lstm_out, _ = self.glove_lstm(glove_lstm_in)
        
        # Apply attention
        bert_context_vector = self.bert_attention(bert_lstm_out)
        glove_context_vector = self.glove_attention(glove_lstm_out)
        
        # Concatenate BERT and GloVe context vectors
        combined_context_vector = torch.cat((bert_context_vector, glove_context_vector), dim=1)
        
        # Classification layers
        x = self.dense1(combined_context_vector)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.dense2(x)
        predictions = self.softmax(logits)
        
        return predictions

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        glove_embedding = batch['glove_embedding'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, glove_embedding)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            glove_embedding = batch['glove_embedding'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, glove_embedding)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1, precision, recall

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    train_loader, test_loader, tokenizer = prepare_bert_data(
        'data/Mishra/train.txt',
        'data/Mishra/test.txt',
        'data/glove.6B.300d.txt',
        batch_size=16
    )
    
    model_params = {
        'dropout_rate': 0.3,
        'freeze_bert': True
    }
    
    training_params = {
        'learning_rate': 2e-5,
        'num_epochs': 25,
        'batch_size': 16
    }
    
    model_path = 'sarcasm_detector_model_i.pth'
    
    model = SarcasmDetector(**model_params).to(device)
    
    # Use DataParallel to utilize multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_params['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        
        # Evaluate the model
        test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate(model, test_loader, criterion, device)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        print(f'Test F1 Score: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}')
    
    print("Training model...")
    for epoch in range(training_params['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{training_params["num_epochs"]}, Train Loss: {train_loss:.4f}')
        
        test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate(model, test_loader, criterion, device)
        print(f'Epoch {epoch+1}/{training_params["num_epochs"]}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

        scheduler.step(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    print("Training complete!")

    model.load_state_dict(torch.load(model_path))
    
    test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}')