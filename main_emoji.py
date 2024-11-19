import torch
import torch.nn as nn
from transformers import BertModel
from sklearn.metrics import f1_score, precision_score, recall_score
from utils_emoji import SarcasmDataset, prepare_bert_data
import os
import numpy as np
from typing import List, Optional
import gensim.models as gsm
from pathlib import Path
import time  # Add at top with other imports

class EmojiEncoder(nn.Module):
    """Encodes emoji sequences into fixed-dimensional embeddings."""
    
    def __init__(self, emoji_dim: int = 300, model_path: str = 'emoji2vec.bin'):  # Back to 300
        """
        Args:
            emoji_dim: Output dimension of emoji embeddings
            model_path: Path to pre-trained emoji2vec binary file
        """
        super().__init__()
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"emoji2vec model not found at {model_path}")
            
        self.emoji2vec = gsm.KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.projection = nn.Linear(self.emoji2vec.vector_size, emoji_dim)
        
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Convert text sequences containing emojis to fixed-length embeddings.
        
        Args:
            texts: List of strings potentially containing emojis
            
        Returns:
            Tensor of shape (batch_size, seq_len, emoji_dim) containing emoji embeddings
        """
        batch_size = len(texts)
        device = next(self.projection.parameters()).device
        
        # Pre-allocate output tensor - adding sequence length dimension
        emoji_embeddings = torch.zeros(batch_size, 128, self.emoji2vec.vector_size)  # Adding seq length dim
        
        for i, text in enumerate(texts):
            # Extract emojis present in vocabulary
            emojis = [c for c in text if c in self.emoji2vec.key_to_index]
            
            if emojis:
                # Get embeddings for all emojis in text
                emoji_vecs = [self.emoji2vec[emoji] for emoji in emojis]
                # Stack emoji vectors along sequence dimension
                emoji_seq = torch.tensor(np.stack(emoji_vecs))
                # Pad or truncate to fixed sequence length
                if len(emoji_seq) > 128:
                    emoji_embeddings[i] = emoji_seq[:128]
                else:
                    emoji_embeddings[i, :len(emoji_seq)] = emoji_seq
                
        # Move to same device as model and project each embedding in the sequence
        emoji_embeddings = emoji_embeddings.to(device)
        emoji_embeddings = self.projection(emoji_embeddings)  # Shape: [batch_size, seq_len, emoji_dim]
        return emoji_embeddings
    
class Attention(nn.Module):
    def __init__(self, lstm_hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(lstm_hidden_size * 2, 1)

    def forward(self, lstm_out):
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        return context_vector

class SarcasmDetector(nn.Module):
    def __init__(self, dropout_rate=0.3, freeze_bert=True):
        super(SarcasmDetector, self).__init__()
        
        # BERT components
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.bert_dim = 768
        
        # Emoji components
        self.emoji_encoder = EmojiEncoder()
        
        # Channels and sizes
        self.cnn_out_channels = 256
        self.lstm_hidden_size = 128
        self.dense_hidden_size = 64
        
        # BERT pathway layers
        self.bert_conv = nn.Conv1d(
            in_channels=self.bert_dim,
            out_channels=self.cnn_out_channels,
            kernel_size=3,
            padding=1
        )
        
        self.bert_lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate
        )
        
        self.bert_attention = Attention(self.lstm_hidden_size)
        
        # Emoji pathway layers
        self.emoji_conv = nn.Conv1d(
            in_channels=300,  # Original emoji dimension
            out_channels=self.cnn_out_channels,
            kernel_size=3,
            padding=1
        )
        
        self.emoji_lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate
        )
        
        self.emoji_attention = Attention(self.lstm_hidden_size)
        
        # Fusion and classification layers
        combined_features_size = (self.lstm_hidden_size * 4)  # 2 * (lstm_hidden_size * 2)
        self.fusion = nn.Linear(combined_features_size, self.dense_hidden_size)
        
        self.dense1 = nn.Linear(self.dense_hidden_size, self.dense_hidden_size)
        self.dense2 = nn.Linear(self.dense_hidden_size, 2)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def process_bert_features(self, embeddings):
        cnn_in = embeddings.permute(0, 2, 1)
        cnn_out = self.relu(self.bert_conv(cnn_in))
        lstm_in = cnn_out.permute(0, 2, 1)
        
        lstm_out, _ = self.bert_lstm(lstm_in)
        features = self.bert_attention(lstm_out)
        return features

    def process_emoji_features(self, embeddings):
        cnn_in = embeddings.permute(0, 2, 1)
        cnn_out = self.relu(self.emoji_conv(cnn_in))
        lstm_in = cnn_out.permute(0, 2, 1)
        
        lstm_out, _ = self.emoji_lstm(lstm_in)
        features = self.emoji_attention(lstm_out)
        return features

    def forward(self, input_ids, attention_mask, raw_texts):
        # Process text through BERT pathway
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embeddings = bert_output.last_hidden_state
        text_features = self.process_bert_features(bert_embeddings)
        
        # Process emoji through emoji pathway
        emoji_embeddings = self.emoji_encoder(raw_texts)
        emoji_features = self.process_emoji_features(emoji_embeddings)
        
        # Combine features
        combined_features = torch.cat([text_features, emoji_features], dim=1)
        
        # Continue through dense layers
        x = self.fusion(combined_features)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.dense2(x)
        predictions = self.softmax(logits)
        
        return predictions

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    start_time = time.time()
    total_steps = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        raw_texts = batch['raw_text']
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, raw_texts)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f'  Step [{batch_idx + 1}/{total_steps}], Loss: {loss.item():.4f}')
    
    epoch_time = time.time() - start_time
    return total_loss / len(train_loader), epoch_time

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            raw_texts = batch['raw_text']
            
            outputs = model(input_ids, attention_mask, raw_texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            
            batch_preds = preds.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
            
            if batch_idx < 3:
                print(f"\nBatch {batch_idx}:")
                for i in range(min(5, len(batch_preds))):
                    print(f"Pred: {batch_preds[i]}, True: {batch_labels[i]}")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    print("\nClass distribution:")
    print("True labels:", np.bincount(all_labels))
    print("Predictions:", np.bincount(all_preds))
    
    print(f"\nDetailed metrics:")
    print(f"Total samples: {len(all_labels)}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    return avg_loss, accuracy, f1, precision, recall

# Modify optimizer and scheduler section
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    train_loader, test_loader, tokenizer = prepare_bert_data(
        'data/iSarcasmEval/train.txt',
        'data/iSarcasmEval/test.txt',
        batch_size=16
    )
    
    model = SarcasmDetector(dropout_rate=0.3, freeze_bert=True).to(device)
    
    # Try a larger learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # increased from 2e-5
    
    # Define the criterion (loss function)
    criterion = nn.CrossEntropyLoss()
    
    # Make scheduler more aggressive
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=2,  # reduced from 3
        factor=0.1,  # more aggressive reduction from 0.5
        min_lr=1e-6
    )
    
    model_path = 'sarcasm_detector_model_i.pth'
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate(model, test_loader, criterion, device)
    
    print("Training model...")
    total_train_time = 0
    
    for epoch in range(25):
        print(f'\nEpoch {epoch+1}/25')
        train_loss, epoch_time = train_epoch(model, train_loader, optimizer, criterion, device)
        total_train_time += epoch_time
        
        # Evaluate on training set
        print("\nEvaluating on training set:")
        train_loss, train_accuracy, train_f1, train_precision, train_recall = evaluate(model, train_loader, criterion, device)
        
        # Evaluate on test set
        print("\nEvaluating on test set:")
        test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate(model, test_loader, criterion, device)
        
        print(f'\nEpoch Summary:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        print(f'Epoch Time: {epoch_time:.2f}s')
        print(f'Total Training Time: {total_train_time/60:.2f}m')

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
    print(f'Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    print(f'F1 Score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}')