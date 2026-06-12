import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import warnings

# 1. Define multiple model architectures for ensemble
class APTForecaster_Small(nn.Module):
    def __init__(self, input_size):
        super(APTForecaster_Small, self).__init__()
        self.lstm = nn.LSTM(input_size, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class APTForecaster_Medium(nn.Module):
    def __init__(self, input_size):
        super(APTForecaster_Medium, self).__init__()
        self.lstm = nn.LSTM(input_size, 96, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.ln = nn.LayerNorm(192)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(96, 1)
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.ln(out)
        out = self.fc(out)
        return out

class APTForecaster_Large(nn.Module):
    def __init__(self, input_size):
        super(APTForecaster_Large, self).__init__()
        self.lstm = nn.LSTM(input_size, 128, num_layers=2, batch_first=True, dropout=0.4, bidirectional=True)
        self.ln = nn.LayerNorm(256)
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.ln(out)
        out = self.fc(out)
        return out

class APTEnsemble:
    def __init__(self, device):
        self.device = device
        self.models = [
            APTForecaster_Small(5).to(device),
            APTForecaster_Medium(5).to(device),
            APTForecaster_Large(5).to(device)
        ]
        self.model_names = ['Small (64 units)', 'Medium (96 units)', 'Large (128 units)']
    
    def train_ensemble(self, train_loader, val_loader, epochs=8, lr=0.001):
        print("\n" + "="*70)
        print("TRAINING ENSEMBLE (3 MODELS)")
        print("="*70)
        
        for idx, model in enumerate(self.models):
            print(f"\n[{idx+1}/3] Training {self.model_names[idx]}...")
            
            pos_weight = torch.tensor([2081.29]).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6
            )
            
            best_val_auc = 0
            patience_counter = 0
            
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                
                for sequences, labels in train_loader:
                    sequences, labels = sequences.to(self.device), labels.to(self.device)
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                
                # Validation
                model.eval()
                val_probs = []
                val_targets = []
                
                with torch.no_grad():
                    for sequences, labels in val_loader:
                        sequences, labels = sequences.to(self.device), labels.to(self.device)
                        outputs = model(sequences)
                        probs = torch.sigmoid(outputs)
                        val_probs.extend(probs.cpu().numpy())
                        val_targets.extend(labels.cpu().numpy())
                
                val_auc = roc_auc_score(val_targets, val_probs)
                scheduler.step(val_auc)
                
                if epoch % 2 == 0:
                    print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}")
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    torch.save(model.state_dict(), f'ensemble_model_{idx}.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= 3:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            
            print(f"  [OK] Best Val AUC: {best_val_auc:.4f}")
            model.load_state_dict(torch.load(f'ensemble_model_{idx}.pt'))
    
    def predict_ensemble(self, dataloader, threshold=0.5):
        """Get predictions from all models"""
        all_ensemble_probs = []
        all_targets = []
        
        for model_idx, model in enumerate(self.models):
            model.eval()
            model_probs = []
            
            with torch.no_grad():
                for sequences, labels in dataloader:
                    sequences, labels = sequences.to(self.device), labels.to(self.device)
                    outputs = model(sequences)
                    probs = torch.sigmoid(outputs)
                    model_probs.extend(probs.cpu().numpy())
                    
                    if model_idx == 0:  # Only collect targets once
                        all_targets.extend(labels.cpu().numpy())
            
            all_ensemble_probs.append(np.array(model_probs).flatten())
        
        all_targets = np.array(all_targets).flatten()
        all_ensemble_probs = np.array(all_ensemble_probs)  # Convert to numpy array
        return all_ensemble_probs, all_targets
    
    def evaluate_ensemble(self, test_loader):
        print("\n" + "="*70)
        print("ENSEMBLE EVALUATION")
        print("="*70)
        
        ensemble_probs, all_targets = self.predict_ensemble(test_loader)
        
        print("\n[INDIVIDUAL MODEL PERFORMANCE]")
        individual_metrics = []
        
        for idx, (probs, name) in enumerate(zip(ensemble_probs, self.model_names)):
            preds = (probs >= 0.5).astype(int)
            auc = roc_auc_score(all_targets, probs)
            
            cm = confusion_matrix(all_targets, preds)
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = cm[0,0], 0, cm[0,1], 0
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            print(f"\nModel {idx+1} - {name}:")
            print(f"  AUC: {auc:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            
            individual_metrics.append({'name': name, 'auc': auc, 'precision': precision, 'recall': recall})
        
        # Ensemble methods
        print("\n[ENSEMBLE METHODS]")
        
        # Method 1: Average Voting
        ensemble_avg = np.mean(ensemble_probs, axis=0)
        ensemble_avg_preds = (ensemble_avg >= 0.5).astype(int)
        avg_auc = roc_auc_score(all_targets, ensemble_avg)
        
        cm_avg = confusion_matrix(all_targets, ensemble_avg_preds)
        if cm_avg.size == 4:
            tn, fp, fn, tp = cm_avg.ravel()
        else:
            tn, fp, fn, tp = cm_avg[0,0], 0, cm_avg[0,1], 0
        
        avg_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        avg_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        avg_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        print(f"\n1. AVERAGE VOTING ENSEMBLE:")
        print(f"   AUC: {avg_auc:.4f}")
        print(f"   Accuracy: {avg_accuracy:.4f}")
        print(f"   Precision: {avg_precision:.4f}")
        print(f"   Recall: {avg_recall:.4f}")
        print(f"   TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        
        # Method 2: Majority Voting (hard voting)
        hard_votes = np.sum(ensemble_probs >= 0.5, axis=0) >= 2
        hard_preds = hard_votes.astype(int)
        hard_auc = roc_auc_score(all_targets, hard_preds)
        
        cm_hard = confusion_matrix(all_targets, hard_preds)
        if cm_hard.size == 4:
            tn_h, fp_h, fn_h, tp_h = cm_hard.ravel()
        else:
            tn_h, fp_h, fn_h, tp_h = cm_hard[0,0], 0, cm_hard[0,1], 0
        
        hard_recall = tp_h / (tp_h + fn_h) if (tp_h + fn_h) > 0 else 0
        hard_precision = tp_h / (tp_h + fp_h) if (tp_h + fp_h) > 0 else 0
        hard_accuracy = (tp_h + tn_h) / (tp_h + tn_h + fp_h + fn_h) if (tp_h + tn_h + fp_h + fn_h) > 0 else 0
        
        print(f"\n2. MAJORITY VOTING (hard voting):")
        print(f"   Accuracy: {hard_accuracy:.4f}")
        print(f"   Precision: {hard_precision:.4f}")
        print(f"   Recall: {hard_recall:.4f}")
        print(f"   TP: {tp_h}, FP: {fp_h}, FN: {fn_h}, TN: {tn_h}")
        
        # Method 3: Weighted Average
        weights = np.array([m['auc'] for m in individual_metrics])
        weights = weights / np.sum(weights)
        ensemble_weighted = np.average(ensemble_probs, axis=0, weights=weights)
        ensemble_weighted_preds = (ensemble_weighted >= 0.5).astype(int)
        weighted_auc = roc_auc_score(all_targets, ensemble_weighted)
        
        cm_weighted = confusion_matrix(all_targets, ensemble_weighted_preds)
        if cm_weighted.size == 4:
            tn_w, fp_w, fn_w, tp_w = cm_weighted.ravel()
        else:
            tn_w, fp_w, fn_w, tp_w = cm_weighted[0,0], 0, cm_weighted[0,1], 0
        
        weighted_recall = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0
        weighted_precision = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else 0
        weighted_accuracy = (tp_w + tn_w) / (tp_w + tn_w + fp_w + fn_w) if (tp_w + tn_w + fp_w + fn_w) > 0 else 0
        
        print(f"\n3. WEIGHTED VOTING (by AUC):")
        print(f"   Weights: {weights}")
        print(f"   AUC: {weighted_auc:.4f}")
        print(f"   Accuracy: {weighted_accuracy:.4f}")
        print(f"   Precision: {weighted_precision:.4f}")
        print(f"   Recall: {weighted_recall:.4f}")
        print(f"   TP: {tp_w}, FP: {fp_w}, FN: {fn_w}, TN: {tn_w}")
        
        # Visualizations
        print("\n[5] Generating ensemble visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Individual Model ROCs
        for idx, probs in enumerate(ensemble_probs):
            auc = roc_auc_score(all_targets, probs)
            fpr, tpr, _ = roc_curve(all_targets, probs)
            axes[0, 0].plot(fpr, tpr, label=f'{self.model_names[idx]} (AUC={auc:.4f})', linewidth=2)
        
        axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('Individual Model ROC Curves')
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Ensemble Methods ROCs
        ensemble_methods = [
            ('Average Voting', ensemble_avg, avg_auc),
            ('Majority Voting', hard_preds, hard_auc),
            ('Weighted Voting', ensemble_weighted, weighted_auc)
        ]
        
        for method_name, method_probs, method_auc in ensemble_methods:
            if method_name != 'Majority Voting':
                fpr, tpr, _ = roc_curve(all_targets, method_probs)
            else:
                fpr, tpr, _ = roc_curve(all_targets, method_probs)
            axes[0, 1].plot(fpr, tpr, label=f'{method_name} (AUC={method_auc:.4f})', linewidth=2)
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('Ensemble Methods ROC Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Performance Comparison
        methods = ['Model 1\n(Small)', 'Model 2\n(Medium)', 'Model 3\n(Large)', 
                   'Avg\nVoting', 'Majority\nVoting', 'Weighted\nVoting']
        aucs = [m['auc'] for m in individual_metrics] + [avg_auc, hard_auc, weighted_auc]
        recalls = [m['recall'] for m in individual_metrics] + [avg_recall, hard_recall, weighted_recall]
        
        x = np.arange(len(methods))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, aucs, width, label='AUC', alpha=0.8)
        axes[1, 0].bar(x + width/2, recalls, width, label='Recall', alpha=0.8)
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('AUC vs Recall Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(methods, fontsize=9)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].set_ylim([0, 1.1])
        
        # Plot 4: Confusion Matrices
        cm_avg_normalized = cm_avg.astype('float') / cm_avg.sum(axis=1)[:, np.newaxis]
        im = axes[1, 1].imshow(cm_avg_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1, 1].figure.colorbar(im, ax=axes[1, 1])
        axes[1, 1].set(xticks=np.arange(cm_avg.shape[1]),
                       yticks=np.arange(cm_avg.shape[0]),
                       yticklabels=['Benign', 'Malicious'],
                       xticklabels=['Benign', 'Malicious'])
        axes[1, 1].set_ylabel('True Label')
        axes[1, 1].set_xlabel('Predicted Label')
        axes[1, 1].set_title('Ensemble Confusion Matrix (Avg Voting)')
        
        for i in range(cm_avg.shape[0]):
            for j in range(cm_avg.shape[1]):
                axes[1, 1].text(j, i, f'{cm_avg[i, j]}',
                               ha="center", va="center",
                               color="white" if cm_avg_normalized[i, j] > 0.5 else "black")
        
        plt.tight_layout()
        plt.savefig('ensemble_results.png', dpi=150, bbox_inches='tight')
        print("[DONE] Ensemble visualizations saved to 'ensemble_results.png'")
        plt.close()
        
        print("\n" + "="*70)
        print("[SUCCESS] ENSEMBLE TRAINING & TESTING COMPLETE!")
        print("="*70)

def main():
    # Load data
    print("Loading data...")
    X_train = np.load('X_train.npy').astype(np.float32)
    y_train = np.load('y_train.npy').astype(np.float32)
    
    X_test = np.load('X_test.npy').astype(np.float32)
    y_test = np.load('y_test.npy').astype(np.float32)
    
    # Split test set
    split_idx = int(len(X_test) * 0.5)
    X_val, X_test_final = X_test[:split_idx], X_test[split_idx:]
    y_val, y_test_final = y_test[:split_idx], y_test[split_idx:]
    
    # Create dataloaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).view(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).view(-1, 1))
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    test_dataset = TensorDataset(torch.tensor(X_test_final), torch.tensor(y_test_final).view(-1, 1))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create and train ensemble
    ensemble = APTEnsemble(device)
    ensemble.train_ensemble(train_loader, val_loader, epochs=8)
    
    # Evaluate ensemble
    ensemble.evaluate_ensemble(test_loader)

if __name__ == "__main__":
    main()
