import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.models.cnn import SignCNN
from ml.models.lstm import SignLSTM
from ml.datasets.sign_dataset import SignSequenceDataset
from ml.preprocessing.preprocessing import get_base_transform
from ml.utils.config import RAW_DATA_DIR, MODELS_DIR, RESULTS_DIR, LSTM_BATCH_SIZE, LSTM_LEARNING_RATE, LSTM_EPOCHS, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT
from ml.utils.vocabulary import get_num_classes, VOCABULARY

def train_lstm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_classes = get_num_classes()
    feature_dim = 256 # As defined in CNN
    
    # Load CNN
    cnn = SignCNN(num_classes=num_classes, feature_dim=feature_dim).to(device)
    cnn_path = MODELS_DIR / "best_cnn.pth"
    if cnn_path.exists():
        cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
        print("Loaded pre-trained CNN features.")
    else:
        print("Warning: Pre-trained CNN not found. LSTM will be trained on random CNN features.")
    
    cnn.eval() # Freeze CNN during LSTM training
    
    # Load dataset
    print("Loading sequence dataset...")
    full_dataset = SignSequenceDataset(RAW_DATA_DIR, transform=get_base_transform())
    if len(full_dataset) == 0:
        print("Sequence dataset is empty. Run collect_data.py first.")
        return
        
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0)
    
    lstm = SignLSTM(feature_dim=feature_dim, hidden_size=LSTM_HIDDEN_SIZE, num_layers=LSTM_NUM_LAYERS, num_classes=num_classes, dropout=LSTM_DROPOUT).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm.parameters(), lr=LSTM_LEARNING_RATE)
    
    best_val_loss = float('inf')
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print("Starting LSTM training...")
    for epoch in range(LSTM_EPOCHS):
        lstm.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for sequences, labels in train_loader:
            # sequences shape: (B, Seq_Len, C, H, W)
            sequences, labels = sequences.to(device), labels.to(device)
            B, S, C, H, W = sequences.size()
            
            # Extract features using CNN
            with torch.no_grad():
                flat_seq = sequences.view(B * S, C, H, W)
                features = cnn.extract_features(flat_seq)
                features = features.view(B, S, -1)
                
            optimizer.zero_grad()
            outputs = lstm(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * B
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        
        # Validation
        lstm.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                B, S, C, H, W = sequences.size()
                
                flat_seq = sequences.view(B * S, C, H, W)
                features = cnn.extract_features(flat_seq)
                features = features.view(B, S, -1)
                
                outputs = lstm(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * B
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{LSTM_EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(lstm.state_dict(), MODELS_DIR / "best_lstm.pth")
            
    # Evaluation
    print("Evaluating best LSTM model...")
    lstm.load_state_dict(torch.load(MODELS_DIR / "best_lstm.pth", map_location=device, weights_only=True))
    lstm.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            B, S, C, H, W = sequences.size()
            flat_seq = sequences.view(B * S, C, H, W)
            features = cnn.extract_features(flat_seq)
            features = features.view(B, S, -1)
            
            outputs = lstm(features)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    if len(all_labels) > 0:
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        print(f"Final Validation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        present_labels = sorted(list(set(all_labels)))
        target_names = [VOCABULARY[i] for i in present_labels]
        
        report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
        with open(RESULTS_DIR / "lstm_classification_report.txt", "w") as f:
            f.write(report)
            
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('LSTM Training Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.title('LSTM Training Accuracy')
        plt.legend()
        plt.savefig(RESULTS_DIR / "lstm_training_curves.png")
        plt.close()
        
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.title('LSTM Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(RESULTS_DIR / "lstm_confusion_matrix.png")
        plt.close()
        
    print("LSTM Training complete. Results saved in 'results/' and model in 'models/'.")

if __name__ == "__main__":
    train_lstm()
