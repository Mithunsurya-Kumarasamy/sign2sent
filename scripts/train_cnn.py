import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.models.cnn import SignCNN
from ml.datasets.cnn_dataset import SignFrameDataset
from ml.preprocessing.preprocessing import get_train_transform, get_base_transform
from ml.utils.config import RAW_DATA_DIR, MODELS_DIR, RESULTS_DIR, CNN_BATCH_SIZE, CNN_LEARNING_RATE, CNN_EPOCHS
from ml.utils.vocabulary import get_num_classes, VOCABULARY

def train_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup dataset
    print("Loading dataset...")
    full_dataset = SignFrameDataset(RAW_DATA_DIR, transform=get_train_transform())
    if len(full_dataset) == 0:
        print("Dataset is empty. Run collect_data.py first.")
        return
        
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Overwrite transform for validation to not use augmentation
    val_dataset.dataset.transform = get_base_transform()
    
    train_loader = DataLoader(train_dataset, batch_size=CNN_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CNN_BATCH_SIZE, shuffle=False, num_workers=0)
    
    num_classes = get_num_classes()
    model = SignCNN(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CNN_LEARNING_RATE)
    
    best_val_loss = float('inf')
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print("Starting training...")
    for epoch in range(CNN_EPOCHS):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{CNN_EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODELS_DIR / "best_cnn.pth")
            
    # Evaluation on best model
    print("Evaluating best model...")
    model.load_state_dict(torch.load(MODELS_DIR / "best_cnn.pth"))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"Final Validation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Save Classification Report
    report = classification_report(all_labels, all_preds, target_names=[VOCABULARY[i] for i in sorted(list(set(all_labels)))], zero_division=0)
    with open(RESULTS_DIR / "cnn_classification_report.txt", "w") as f:
        f.write(report)
        
    # Plot Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('CNN Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('CNN Training Accuracy')
    plt.legend()
    plt.savefig(RESULTS_DIR / "cnn_training_curves.png")
    plt.close()
    
    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[VOCABULARY[i] for i in sorted(list(set(all_labels)))], yticklabels=[VOCABULARY[i] for i in sorted(list(set(all_labels)))])
    plt.title('CNN Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(RESULTS_DIR / "cnn_confusion_matrix.png")
    plt.close()
    
    print("Training complete. Results saved in 'results/' and model in 'models/'.")

if __name__ == "__main__":
    train_cnn()
