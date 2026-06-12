import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import warnings

# 1. Define the Upgraded Neural Network Architecture
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
            dropout=dropout_prob if num_layers > 1 else 0.0
        )
        
        self.fc_dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size, 1)
        # REMOVED Sigmoid layer - BCEWithLogitsLoss handles this natively for better stability
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.fc_dropout(out)
        out = self.fc(out) 
        # Output is now raw "logits", not probabilities
        return out

# 2. Define the Evaluation Function
def evaluate_model(model, dataloader):
    print("\n--- Starting Model Evaluation ---")
    model.eval()  
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            outputs = model(sequences)
            
            # Since we removed Sigmoid from the model, we apply it manually here for evaluation
            probabilities = torch.sigmoid(outputs)
            
            # Convert probabilities to strict 0 or 1 predictions (0.5 threshold)
            predictions = (probabilities >= 0.5).float()
            
            all_preds.extend(predictions.numpy())
            all_targets.extend(labels.numpy())
            
    warnings.filterwarnings('ignore') 
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    
    print("\nClassification Report (Precision, Recall, F1):")
    print(classification_report(all_targets, all_preds, labels=[0, 1], target_names=['Benign (0)', 'Malicious (1)']))

# 3. The Main Training Loop
def train_model():
    print("Loading pure tensor data...")
    X_train = np.load('X_train.npy').astype(np.float32)
    y_train = np.load('y_train.npy').astype(np.float32)
    
    X_test = np.load('X_test.npy').astype(np.float32)
    y_test = np.load('y_test.npy').astype(np.float32)
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).view(-1, 1))
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).view(-1, 1))
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    model = APTForecaster(input_size=5, hidden_size=64, num_layers=2, dropout_prob=0.3)
    print("\nModel Architecture:")
    print(model)
    
    # --- THE CRITICAL FIX: Class Weights ---
    # Weight = Negative Samples (399,608) / Positive Samples (192)
    pos_weight = torch.tensor([2081.29])
    
    # BCEWithLogitsLoss combines Sigmoid and BCELoss in one mathematically stable class
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 3
    print(f"\nStarting training on pure data for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train() 
        
        for i, (sequences, labels) in enumerate(train_dataloader):
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (i+1) % 500 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
                
        print(f"--- Epoch {epoch+1} completed. Average Loss: {epoch_loss/len(train_dataloader):.4f} ---")
    
    evaluate_model(model, test_dataloader)

if __name__ == "__main__":
    train_model()