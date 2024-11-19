import torch
import torch.nn as nn
from transformers import BertModel
from emoji2vec import Emoji2Vec
import numpy as np

class EmojiEncoder(nn.Module):
    def __init__(self, emoji_dim=300):
        super(EmojiEncoder, self).__init__()
        self.emoji2vec = Emoji2Vec()
        self.emoji_dim = emoji_dim
        self.projection = nn.Linear(emoji2vec.vector_size, emoji_dim)
        
    def forward(self, text_batch):
        emoji_embeddings = []
        for text in text_batch:
            # Extract emojis from text
            emojis = [c for c in text if c in self.emoji2vec.emoji_to_vec]
            if emojis:
                # Get emoji vectors
                emoji_vecs = [self.emoji2vec[emoji] for emoji in emojis]
                emoji_vec = torch.tensor(np.mean(emoji_vecs, axis=0))
            else:
                emoji_vec = torch.zeros(self.emoji2vec.vector_size)
            emoji_embeddings.append(emoji_vec)
            
        emoji_tensor = torch.stack(emoji_embeddings).to(next(self.projection.parameters()).device)
        return self.projection(emoji_tensor)

class SarcasmDetector(nn.Module):
    def __init__(self, dropout_rate=0.3, freeze_bert=True):
        super(SarcasmDetector, self).__init__()
        
        # BERT and original layers
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.bert_dim = 768
        
        # Emoji encoding
        self.emoji_encoder = EmojiEncoder(emoji_dim=256)
        
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
        
        # Fusion layer for combining text and emoji features
        self.fusion = nn.Linear(self.lstm_hidden_size * 2 + 256, self.dense_hidden_size)
        
        # Dense layers
        self.dense1 = nn.Linear(self.dense_hidden_size, self.dense_hidden_size)
        self.dense2 = nn.Linear(self.dense_hidden_size, 2)
        
        # Regularization and activation layers
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, raw_texts):
        # BERT embedding layer
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embeddings = bert_output.last_hidden_state
        
        # CNN feature extraction
        cnn_in = bert_embeddings.permute(0, 2, 1)
        cnn_out = self.relu(self.conv1d(cnn_in))
        lstm_in = cnn_out.permute(0, 2, 1)
        
        # BiLSTM sequence learning
        lstm_out, _ = self.lstm(lstm_in)
        
        # Apply attention
        text_features = self.attention(lstm_out)
        
        # Get emoji features
        emoji_features = self.emoji_encoder(raw_texts)
        
        # Combine text and emoji features
        combined_features = torch.cat([text_features, emoji_features], dim=1)
        fused_features = self.fusion(combined_features)
        
        # Classification layers
        x = self.dense1(fused_features)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.dense2(x)
        predictions = self.softmax(logits)
        
        return predictions

# Modified data preparation to include raw text
def prepare_batch(batch, device):
    return {
        'input_ids': batch['input_ids'].to(device),
        'attention_mask': batch['attention_mask'].to(device),
        'labels': batch['labels'].to(device),
        'raw_texts': batch['raw_text']  # Keep raw text for emoji processing
    }
