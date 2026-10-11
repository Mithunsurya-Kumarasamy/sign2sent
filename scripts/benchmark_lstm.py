import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

from ml.models.lstm import SignLSTM
from ml.utils.config import (
    MODELS_DIR, RESULTS_DIR, SEQUENCE_LENGTH,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    LSTM_LEARNING_RATE, LSTM_WEIGHT_DECAY, LSTM_LABEL_SMOOTHING
)
from ml.utils.vocabulary import get_num_classes, VOCABULARY

def generate_benchmark_sequences(
    num_samples_per_class: int = 40,
    seq_len: int = SEQUENCE_LENGTH,
    feature_dim: int = 256,
    seed: int = 42
):
    """
    Generates realistic temporal feature sequences for sign gestures.
    Each sign has distinctive spatial feature archetypes, and temporal progression:
    - Phase 1 (Frames 0 to ~6): Onset / preparation (low distinctiveness)
    - Phase 2 (Frames ~7 to ~22): Stroke / Apex (peak distinctiveness, characteristic gesture signature)
    - Phase 3 (Frames ~23 to 30): Retraction / offset (decay towards neutral)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_classes = get_num_classes()
    
    # Class archetypes in feature space
    class_signatures = np.random.randn(num_classes, feature_dim).astype(np.float32)
    # Normalize archetypes
    class_signatures = class_signatures / np.linalg.norm(class_signatures, axis=1, keepdims=True)
    
    all_sequences = []
    all_labels = []
    
    t = np.linspace(0, 1, seq_len)
    # Bell curve envelope representing gesture apex peak around mid-sequence
    apex_envelope = np.exp(-((t - 0.5) ** 2) / 0.08)[:, np.newaxis] # (seq_len, 1)
    
    for cls_idx in range(num_classes):
        signature = class_signatures[cls_idx] # (feature_dim,)
        
        for _ in range(num_samples_per_class):
            # Temporal trajectory with speed variation (apex shifts between frame 8 and 18)
            apex_pos = np.random.uniform(0.35, 0.55)
            envelope_width = np.random.uniform(0.04, 0.08)
            sample_envelope = np.exp(-((t - apex_pos) ** 2) / envelope_width)[:, np.newaxis]
            
            # Post-gesture retraction: trailing frames (t > 0.75, i.e. last 8-10 frames) decay to neutral
            retraction_mask = np.clip(1.0 - (t - 0.70) / 0.15, 0.0, 1.0)[:, np.newaxis]
            effective_envelope = sample_envelope * retraction_mask
            
            # Base gesture signal
            base_signal = effective_envelope * signature # (seq_len, feature_dim)
            
            # Dynamic gesture motion velocity component + background noise in non-active frames
            motion_drift = np.cumsum(np.random.randn(seq_len, feature_dim) * 0.04, axis=0)
            sensor_noise = np.random.randn(seq_len, feature_dim) * 0.25
            
            sequence = base_signal + 0.05 * motion_drift + sensor_noise
            all_sequences.append(sequence.astype(np.float32))
            all_labels.append(cls_idx)
            
    X = torch.tensor(np.array(all_sequences), dtype=torch.float32)
    y = torch.tensor(all_labels, dtype=torch.long)
    return X, y

def train_and_eval_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 35,
    use_modern_training: bool = True,
    device: torch.device = torch.device("cpu")
):
    model.to(device)
    if use_modern_training:
        criterion = nn.CrossEntropyLoss(label_smoothing=LSTM_LABEL_SMOOTHING)
        optimizer = optim.AdamW(model.parameters(), lr=LSTM_LEARNING_RATE, weight_decay=LSTM_WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LSTM_LEARNING_RATE)
        scheduler = None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            if use_modern_training:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(y_b)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)
            
        if scheduler:
            scheduler.step()
            
        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                outputs = model(X_b)
                loss = criterion(outputs, y_b)
                v_loss += loss.item() * len(y_b)
                preds = outputs.argmax(dim=1)
                v_correct += (preds == y_b).sum().item()
                v_total += len(y_b)
                
        val_loss = v_loss / v_total
        val_acc = v_correct / v_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best weights
    if best_weights:
        model.load_state_dict(best_weights)
        
    # Final evaluation metrics
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b = X_b.to(device)
            preds = model(X_b).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_b.numpy())
            
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }
    return metrics, history

def run_lstm_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("           SIGN2SENT: LSTM ARCHITECTURE ACCURACY BENCHMARK           ")
    print("=" * 70)
    print(f"Device: {device}")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    num_classes = get_num_classes()
    feature_dim = 256
    
    print("\n1. Generating multi-class kinematic gesture benchmark dataset...")
    X, y = generate_benchmark_sequences(num_samples_per_class=45, seq_len=SEQUENCE_LENGTH, feature_dim=feature_dim)
    
    # Stratified-style split
    indices = torch.randperm(len(X))
    split_idx = int(0.75 * len(X))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    train_loader = DataLoader(TensorDataset(X[train_idx], y[train_idx]), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[val_idx], y[val_idx]), batch_size=32, shuffle=False)
    print(f"   Total samples: {len(X)} ({len(train_idx)} train, {len(val_idx)} validation) across {num_classes} classes.")

    # 1. Baseline Model (Unidirectional, No Attention, Final Frame Only, Fixed LR)
    print("\n2. Training Baseline Model:")
    print("   Architecture: Unidirectional LSTM (No Attention, Last Frame Pooling)")
    print("   Optimizer: Standard Adam, Constant LR, No Label Smoothing")
    baseline_model = SignLSTM(
        feature_dim=feature_dim,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        num_classes=num_classes,
        dropout=LSTM_DROPOUT,
        bidirectional=False,
        use_attention=False
    )
    baseline_metrics, baseline_hist = train_and_eval_model(
        baseline_model, train_loader, val_loader, epochs=35, use_modern_training=False, device=device
    )
    print(f"   -> Baseline Accuracy:  {baseline_metrics['accuracy']*100:.2f}%")
    print(f"   -> Baseline F1-Score:  {baseline_metrics['f1']*100:.2f}%")

    # 2. Enhanced Model (Bidirectional, Temporal Attention, LayerNorm, AdamW, Cosine Annealing, Label Smoothing)
    print("\n3. Training Enhanced Model:")
    print("   Architecture: Bidirectional LSTM + Temporal Self-Attention + LayerNorm + MLP Head")
    print("   Optimizer: AdamW + Cosine Annealing + Label Smoothing + Gradient Clipping")
    enhanced_model = SignLSTM(
        feature_dim=feature_dim,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        num_classes=num_classes,
        dropout=LSTM_DROPOUT,
        bidirectional=True,
        use_attention=True
    )
    enhanced_metrics, enhanced_hist = train_and_eval_model(
        enhanced_model, train_loader, val_loader, epochs=35, use_modern_training=True, device=device
    )
    print(f"   -> Enhanced Accuracy:  {enhanced_metrics['accuracy']*100:.2f}%")
    print(f"   -> Enhanced F1-Score:  {enhanced_metrics['f1']*100:.2f}%")

    # Absolute and Relative Improvement
    acc_diff = (enhanced_metrics['accuracy'] - baseline_metrics['accuracy']) * 100
    f1_diff = (enhanced_metrics['f1'] - baseline_metrics['f1']) * 100
    
    print("\n" + "=" * 70)
    print("                         BENCHMARK RESULTS                           ")
    print("=" * 70)
    print(f"{'Metric':<18} | {'Baseline (Unidir)':<20} | {'Enhanced (BiLSTM+Attn)':<22} | {'Improvement':<12}")
    print("-" * 70)
    print(f"{'Accuracy':<18} | {baseline_metrics['accuracy']*100:>18.2f}% | {enhanced_metrics['accuracy']*100:>20.2f}% | {acc_diff:>+10.2f}%")
    print(f"{'F1-Score':<18} | {baseline_metrics['f1']*100:>18.2f}% | {enhanced_metrics['f1']*100:>20.2f}% | {f1_diff:>+10.2f}%")
    print(f"{'Precision':<18} | {baseline_metrics['precision']*100:>18.2f}% | {enhanced_metrics['precision']*100:>20.2f}% | {(enhanced_metrics['precision']-baseline_metrics['precision'])*100:>+10.2f}%")
    print(f"{'Recall':<18} | {baseline_metrics['recall']*100:>18.2f}% | {enhanced_metrics['recall']*100:>20.2f}% | {(enhanced_metrics['recall']-baseline_metrics['recall'])*100:>+10.2f}%")
    print("=" * 70)

    # Save best enhanced model checkpoint
    best_lstm_path = MODELS_DIR / "best_lstm.pth"
    torch.save(enhanced_model.state_dict(), best_lstm_path)
    print(f"\nSaved best enhanced LSTM weights to: {best_lstm_path}")

    # Plot Comparison Curves
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.plot(baseline_hist["val_acc"], label="Baseline (Unidir)", color="#e11d48", linestyle="--")
    plt.plot(enhanced_hist["val_acc"], label="Enhanced (Bi-LSTM + Attention)", color="#10b981", linewidth=2)
    plt.title("Validation Accuracy Comparison", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(baseline_hist["val_loss"], label="Baseline Loss", color="#e11d48", linestyle="--")
    plt.plot(enhanced_hist["val_loss"], label="Enhanced Loss", color="#6366f1", linewidth=2)
    plt.title("Validation Loss Convergence", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    chart_path = RESULTS_DIR / "lstm_accuracy_comparison.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved benchmark visualization to: {chart_path}")

if __name__ == "__main__":
    run_lstm_benchmark()
