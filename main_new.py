import torch
import torch.nn as nn
from transformers import BertModel
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import SarcasmDataset, prepare_bert_data
import os
import argparse
from tqdm import tqdm
import time

class SarcasmDetector(nn.Module):
    def __init__(self, dropout_rate=0.3, freeze_bert=False):  # Note: changed default freeze_bert to False
        super(SarcasmDetector, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, 2)  # 768 is BERT's hidden size
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        predictions = self.softmax(logits)
        return predictions

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    # Create progress bar
    progress_bar = tqdm(
        train_loader,
        desc=f'Epoch {epoch}/{num_epochs}',
        total=len(train_loader),
        unit='batch'
    )
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(train_loader)
    
    print(f'\nEpoch Statistics:')
    print(f'Time taken: {epoch_time:.2f} seconds')
    print(f'Average loss: {avg_loss:.4f}')
    
    return avg_loss

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    
    # Add progress bar for evaluation
    progress_bar = tqdm(test_loader, desc='Evaluating', unit='batch')
    
    with torch.no_grad():
        for batch in progress_bar:
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
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
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
    print("Train data:", train_loader)
    
    model_params = {
        'dropout_rate': 0.3,
        'freeze_bert': False  # We want to fine-tune BERT
    }
    
    training_params = {
        'learning_rate': 2e-5,  # Typical BERT fine-tuning learning rate
        'num_epochs': 10,       # Reduced epochs as BERT fine-tuning usually requires fewer
        'batch_size': 16
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
        test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate(model, test_loader, criterion, device)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        print(f'Test F1 Score: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}')
    
    print("Training model...")
    total_start_time = time.time()
    
    for epoch in range(training_params['num_epochs']):
        num_epochs = training_params['num_epochs']
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        train_loss = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            criterion, 
            device,
            epoch + 1,
            training_params['num_epochs']
        )
        print(f'Epoch {epoch+1}/{training_params["num_epochs"]}, Train Loss: {train_loss:.4f}')
        
        test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate(
            model, test_loader, criterion, device
        )
        
        print(f'Epoch {epoch+1}/{training_params["num_epochs"]}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        print(f'Test Metrics:')
        print(f'Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')
        print(f'F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}')
        
        scheduler.step(test_loss)
        
        # Save model after each epoch
        epoch_model_path = f"{args.model_path.split('.')[0]}_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Model saved to {epoch_model_path}")
        
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), args.model_path)
            print(f"Best model saved to {args.model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    total_time = time.time() - total_start_time
    print(f"\nTraining complete! Total time: {total_time/60:.2f} minutes")

    model.load_state_dict(torch.load(args.model_path))
    
    test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}')
    
    # Print embedding length and embedding matrix shape
    embedding_matrix = model.bert.embeddings.word_embeddings.weight
    print(f'Embedding length: {embedding_matrix.shape[1]}')
    print(f'Embedding matrix shape: {embedding_matrix.shape}')
