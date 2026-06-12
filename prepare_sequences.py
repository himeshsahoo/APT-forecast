import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def process_data(input_csv, seq_length=10):
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Sort by host and time to maintain chronology
    df = df.sort_values(by=['hostname', 'timestamp'])
    df.fillna('UNKNOWN', inplace=True)
    
    print("Encoding categorical features...")
    categorical_cols = ['action', 'actor_process', 'target_process', 'dest_ip', 'dest_port']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    print(f"Generating sequences of length {seq_length}...")
    X, y = [], []
    
    # Group by hostname so sequences don't cross between different computers
    for _, group in df.groupby('hostname'):
        features = group[categorical_cols].values
        labels = group['is_malicious'].values
        
        for i in range(len(features) - seq_length):
            X.append(features[i:(i + seq_length)])
            y.append(labels[i + seq_length - 1])
            
    X = np.array(X)
    y = np.array(y)
    
    # Split into Train/Test - NO SMOTE APPLIED
    print("\nSplitting data into 80% Training and 20% Testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Pure Training Set Reality Check: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Pure Testing Set Reality Check: {pd.Series(y_test).value_counts().to_dict()}")
    
    # Save the untouched, highly imbalanced tensors
    print("\nSaving final isolated tensors...")
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test)  
    np.save('y_test.npy', y_test)
    print("Successfully saved all pure tensor files.")

if __name__ == "__main__":
    INPUT_CSV = "optc_parsed_sequences.csv"
    process_data(INPUT_CSV, seq_length=10)