import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import warnings

# 1. Define the same model architecture
class APTForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob=0.3):
        super(APTForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        self.ln = nn.LayerNorm(hidden_size * 2)
        self.fc_dropout = nn.Dropout(p=dropout_prob)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2_dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.ln(out)
        out = self.fc_dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2_dropout(out)
        out = self.fc2(out)
        return out

def test_model():
    print("="*70)
    print("ADVANCED MODEL TESTING & EVALUATION")
    print("="*70)
    
    # Load data
    print("\n[1] Loading test data...")
    X_test = np.load('X_test.npy').astype(np.float32)
    y_test = np.load('y_test.npy').astype(np.float32)
    
    # Use the second half as test set (same as training)
    split_idx = int(len(X_test) * 0.5)
    X_test_final = X_test[split_idx:]
    y_test_final = y_test[split_idx:]
    
    test_dataset = TensorDataset(torch.tensor(X_test_final), torch.tensor(y_test_final).view(-1, 1))
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\n[2] Loading trained model...")
    model = APTForecaster(input_size=5, hidden_size=128, num_layers=2, dropout_prob=0.4).to(device)
    model.load_state_dict(torch.load('best_model.pt'))
    print("✓ Model loaded successfully")
    
    # Run inference
    print("\n[3] Running inference on test set...")
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    all_logits = []
    
    with torch.no_grad():
        for sequences, labels in test_dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            logits = model(sequences)
            probs = torch.sigmoid(logits)
            
            all_logits.extend(logits.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs >= 0.5).float().cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    all_logits = np.array(all_logits).flatten()
    all_probs = np.array(all_probs).flatten()
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Detailed metrics
    print("\n" + "="*70)
    print("TEST SET PERFORMANCE METRICS")
    print("="*70)
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n[CONFUSION MATRIX]")
    print(f"True Negatives (TN):  {tn:>10}")
    print(f"False Positives (FP): {fp:>10}")
    print(f"False Negatives (FN): {fn:>10}")
    print(f"True Positives (TP):  {tp:>10}")
    
    # Basic metrics
    print("\n[BASIC METRICS]")
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    
    # Advanced metrics
    print("\n[ADVANCED METRICS]")
    print(f"Sensitivity (TPR):    {sensitivity:.4f}")
    print(f"Specificity (TNR):    {specificity:.4f}")
    
    # False Positive & False Negative Rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    print(f"False Positive Rate:  {fpr:.4f}")
    print(f"False Negative Rate:  {fnr:.4f}")
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
        print(f"ROC-AUC Score:       {roc_auc:.4f}")
    except:
        roc_auc = None
    
    # Matthews Correlation Coefficient
    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0
    print(f"Matthews CC:         {mcc:.4f}")
    
    # Classification Report
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(all_targets, all_preds, labels=[0, 1], 
                                target_names=['Benign (0)', 'Malicious (1)']))
    
    # Threshold optimization
    print("\n[THRESHOLD ANALYSIS]")
    print("Finding optimal threshold...")
    
    precision_vals, recall_vals, thresholds = precision_recall_curve(all_targets, all_probs)
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    print(f"Optimal Threshold:   {best_threshold:.4f}")
    print(f"F1-Score at optimum: {best_f1:.4f}")
    
    # Test with optimal threshold
    if best_threshold != 0.5:
        print(f"\nRe-evaluating with optimal threshold ({best_threshold:.4f})...")
        opt_preds = (all_probs >= best_threshold).astype(int)
        opt_f1 = f1_score(all_targets, opt_preds)
        opt_precision = np.sum((opt_preds == 1) & (all_targets == 1)) / np.sum(opt_preds == 1) if np.sum(opt_preds == 1) > 0 else 0
        opt_recall = np.sum((opt_preds == 1) & (all_targets == 1)) / np.sum(all_targets == 1) if np.sum(all_targets == 1) > 0 else 0
        opt_accuracy = np.mean(opt_preds == all_targets)
        
        print(f"Optimal Accuracy:    {opt_accuracy:.4f}")
        print(f"Optimal Precision:   {opt_precision:.4f}")
        print(f"Optimal Recall:      {opt_recall:.4f}")
        print(f"Optimal F1-Score:    {opt_f1:.4f}")
    
    # Prediction distribution
    print("\n[PREDICTION DISTRIBUTION]")
    print(f"Predicted as Benign:    {np.sum(all_preds == 0):>10} ({np.mean(all_preds == 0)*100:.2f}%)")
    print(f"Predicted as Malicious: {np.sum(all_preds == 1):>10} ({np.mean(all_preds == 1)*100:.2f}%)")
    
    print(f"\nActual Benign:    {np.sum(all_targets == 0):>10} ({np.mean(all_targets == 0)*100:.2f}%)")
    print(f"Actual Malicious: {np.sum(all_targets == 1):>10} ({np.mean(all_targets == 1)*100:.2f}%)")
    
    # Probability statistics
    print("\n[PROBABILITY STATISTICS]")
    print(f"Mean Probability (Benign):    {np.mean(all_probs[all_targets == 0]):.4f}")
    print(f"Mean Probability (Malicious): {np.mean(all_probs[all_targets == 1]):.4f}")
    print(f"Min Probability: {np.min(all_probs):.4f}")
    print(f"Max Probability: {np.max(all_probs):.4f}")
    
    # Visualizations
    print("\n[4] Generating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Probability Distribution
    axes[0, 0].hist(all_probs[all_targets == 0], bins=50, alpha=0.6, label='Benign', color='blue')
    axes[0, 0].hist(all_probs[all_targets == 1], bins=50, alpha=0.6, label='Malicious', color='red')
    axes[0, 0].axvline(0.5, color='black', linestyle='--', label='Default Threshold')
    if best_threshold != 0.5:
        axes[0, 0].axvline(best_threshold, color='green', linestyle='--', label='Optimal Threshold')
    axes[0, 0].set_xlabel('Prediction Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Probability Distribution by Class')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ROC Curve
    if roc_auc:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={roc_auc:.4f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        axes[0, 1].fill_between(fpr, tpr, alpha=0.2)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Confusion Matrix Heatmap
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], 
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'])
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    axes[1, 0].set_title('Confusion Matrix')
    
    # 4. Precision-Recall Curve
    axes[1, 1].plot(recall_vals, precision_vals, linewidth=2, label='PR Curve')
    axes[1, 1].scatter(opt_recall if best_threshold != 0.5 else recall, 
                       opt_precision if best_threshold != 0.5 else precision,
                       color='red', s=100, label=f'Optimal Point', zorder=5)
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_title('Precision-Recall Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_test_results.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to 'model_test_results.png'")
    plt.close()
    
    # Sample predictions
    print("\n[5] Sample Predictions (first 20 test samples):")
    print("-" * 70)
    print(f"{'Index':<8} {'Actual':<12} {'Predicted':<12} {'Probability':<15} {'Correct':<10}")
    print("-" * 70)
    
    for i in range(min(20, len(all_targets))):
        actual = 'Malicious' if all_targets[i] == 1 else 'Benign'
        predicted = 'Malicious' if all_preds[i] == 1 else 'Benign'
        prob = all_probs[i]
        correct = '✓' if all_targets[i] == all_preds[i] else '✗'
        print(f"{i:<8} {actual:<12} {predicted:<12} {prob:<15.4f} {correct:<10}")
    
    print("\n" + "="*70)
    print("✓ TESTING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - model_test_results.png (comprehensive test visualizations)")
    print("  - best_model.pt (trained model)")

if __name__ == "__main__":
    test_model()
