import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import warnings
import matplotlib.pyplot as plt

# 1. Define the Optimized Neural Network Architecture
class APTForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob=0.3):
        super(APTForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM for better context understanding
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0.0,
            bidirectional=True  # Process sequence in both directions
        )
        
        # Layer normalization for more stable training
        self.ln = nn.LayerNorm(hidden_size * 2)  # *2 because bidirectional
        
        # Multi-layer dense head with residual-like structure
        self.fc_dropout = nn.Dropout(p=dropout_prob)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2_dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last hidden state
        out = self.ln(out)  # Apply layer normalization
        out = self.fc_dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2_dropout(out)
        out = self.fc2(out)
        return out

# 2. Define the Enhanced Evaluation Function
def evaluate_model(model, dataloader, device, plot_curves=True):
    print("\n--- Starting Model Evaluation ---")
    model.eval()  
    
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            
            # Get probabilities
            probabilities = torch.sigmoid(outputs)
            
            # Convert probabilities to strict 0 or 1 predictions (0.5 threshold)
            predictions = (probabilities >= 0.5).float()
            
            all_preds.extend(predictions.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()
    all_targets = np.array(all_targets).flatten()
    
    warnings.filterwarnings('ignore') 
    
    # Basic metrics
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_targets, all_preds)
    print(cm)
    
    print("\nClassification Report (Precision, Recall, F1):")
    print(classification_report(all_targets, all_preds, labels=[0, 1], target_names=['Benign (0)', 'Malicious (1)']))
    
    # Advanced metrics
    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
        
        # Calculate sensitivity and specificity
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
        print(f"Specificity (True Negative Rate): {specificity:.4f}")
        
        # Calculate Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print(f"Accuracy: {accuracy:.4f}")
        
        if plot_curves:
            # Plot ROC curve
            fpr, tpr, _ = roc_curve(all_targets, all_probs)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('roc_curve.png', dpi=100, bbox_inches='tight')
            print("\nROC curve saved to 'roc_curve.png'")
            plt.close()
            
    except Exception as e:
        print(f"Could not calculate ROC-AUC: {e}")
    
    return all_preds, all_targets, all_probs

# 3. The Main Optimized Training Loop
def train_model():
    print("Loading pure tensor data...")
    X_train = np.load('X_train.npy').astype(np.float32)
    y_train = np.load('y_train.npy').astype(np.float32)
    
    X_test = np.load('X_test.npy').astype(np.float32)
    y_test = np.load('y_test.npy').astype(np.float32)
    
    # Split test into validation and test for better model selection
    split_idx = int(len(X_test) * 0.5)
    X_val, X_test_final = X_test[:split_idx], X_test[split_idx:]
    y_val, y_test_final = y_test[:split_idx], y_test[split_idx:]
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).view(-1, 1))
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).view(-1, 1))
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    test_dataset = TensorDataset(torch.tensor(X_test_final), torch.tensor(y_test_final).view(-1, 1))
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = APTForecaster(input_size=5, hidden_size=128, num_layers=2, dropout_prob=0.4).to(device)
    print("\nModel Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # --- Class Weights for Imbalanced Data ---
    pos_weight = torch.tensor([2081.29]).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler - reduce LR when validation plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6
    )
    
    epochs = 10
    best_val_auc = 0
    patience_counter = 0
    max_patience = 4
    training_history = {'loss': [], 'val_loss': [], 'val_auc': []}
    
    print(f"\nStarting optimized training for up to {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train() 
        
        for i, (sequences, labels) in enumerate(train_dataloader):
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (i+1) % 300 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / len(train_dataloader)
        training_history['loss'].append(avg_train_loss)
        print(f"--- Epoch {epoch+1} completed. Average Training Loss: {avg_train_loss:.4f} ---")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_probs = []
        val_targets = []
        
        with torch.no_grad():
            for sequences, labels in val_dataloader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probabilities = torch.sigmoid(outputs)
                val_probs.extend(probabilities.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_auc = roc_auc_score(val_targets, val_probs)
        
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_auc'].append(val_auc)
        
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation AUC: {val_auc:.4f}")
        
        # Learning rate scheduling and early stopping
        scheduler.step(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"✓ New best validation AUC: {val_auc:.4f}. Model saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{max_patience}")
            
            if patience_counter >= max_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model and evaluate
    print("\n" + "="*60)
    print("Loading best model and evaluating on test set...")
    print("="*60)
    model.load_state_dict(torch.load('best_model.pt'))
    
    evaluate_model(model, test_dataloader, device, plot_curves=True)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_history['loss'], label='Train Loss')
    plt.plot(training_history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(training_history['val_auc'], label='Val AUC', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('AUC Score')
    plt.title('Validation AUC Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
    print("\nTraining history saved to 'training_history.png'")
    plt.close()
    
    print("\n✓ Training complete! Check 'best_model.pt', 'roc_curve.png', and 'training_history.png'")

if __name__ == "__main__":
    train_model()