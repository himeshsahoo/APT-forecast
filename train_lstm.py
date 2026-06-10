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
        
        # LSTM layer with recurrent dropout (only applies if num_layers > 1)
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0.0
        )
        
        # Explicit dropout layer before the final classification step
        self.fc_dropout = nn.Dropout(p=dropout_prob)
        
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Forward pass through the LSTM
        out, _ = self.lstm(x)
        
        # Pull the final time-step vector representing the sequence context
        out = out[:, -1, :] 
        
        # Apply regularization before passing to the linear layer
        out = self.fc_dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

# 2. Define the Evaluation Function
def evaluate_model(model, dataloader):
    print("\n--- Starting Model Evaluation ---")
    model.eval()  # Lock the model weights and disable dropout for testing
    
    all_preds = []
    all_targets = []
    
    # Disable gradient calculation to save memory and speed up testing
    with torch.no_grad():
        for sequences, labels in dataloader:
            outputs = model(sequences)
            
            # Convert probabilities to strict 0 or 1 predictions (0.5 threshold)
            predictions = (outputs >= 0.5).float()
            
            all_preds.extend(predictions.numpy())
            all_targets.extend(labels.numpy())
            
    # Suppress the zero-division warning we expect from having only 1 class right now
    warnings.filterwarnings('ignore') 
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    
    print("\nClassification Report (Precision, Recall, F1):")
    # Added labels=[0, 1] to prevent crashes when a class is missing from the batch
    print(classification_report(all_targets, all_preds, labels=[0, 1], target_names=['Benign (0)', 'Malicious (1)']))

## 3. The Main Training Loop
def train_model():
    print("Loading prepped tensor data...")
    # Load Training Data
    X_train = np.load('X_train.npy').astype(np.float32)
    y_train = np.load('y_train.npy').astype(np.float32)
    
    # Load Testing Data (The Untouched Vault)
    X_test = np.load('X_test.npy').astype(np.float32)
    y_test = np.load('y_test.npy').astype(np.float32)
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).view(-1, 1))
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).view(-1, 1))
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Initialize the model
    model = APTForecaster(input_size=5, hidden_size=64, num_layers=2, dropout_prob=0.3)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 3
    print(f"\nStarting training for {epochs} epochs...")
    
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
    
    # Trigger the evaluation using the TEST DATALOADER
    evaluate_model(model, test_dataloader)

if __name__ == "__main__":
    train_model()