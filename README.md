# APT Forecast Project

This project parses OpTC ECAR event logs, labels likely malicious events using known indicators of compromise, prepares sequence data, and trains deep learning models (LSTMs and Ensembles) for forecasting Advanced Persistent Threats (APTs).

## Project Files

### Data Processing
- `parse_optc.py` - Parses ECAR JSON-lines into a CSV dataset, applying a Threat Intelligence Dictionary to label malicious events based on known compromised hosts and malicious keywords.
- `prepare_sequences.py` - Loads the CSV, encodes categorical features, and builds model-ready time-series sequences (default length 10) grouped by host. It splits the data into 80% train and 20% test sets, saving them as NumPy tensors (`X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`).
- `inspect_data.py` - Script to inspect the prepared dataset.

### Modeling
- `train_lstm.py` - Trains an optimized Bidirectional LSTM forecasting model. It includes features like layer normalization, learning rate scheduling, early stopping, and class weights to handle imbalanced data. Generates `best_model.pt`, `roc_curve.png`, and `training_history.png`.
- `train_ensemble.py` - Trains an ensemble of three Bidirectional LSTM models (Small, Medium, Large). It evaluates the ensemble using Average Voting, Majority (Hard) Voting, and Weighted Voting, generating `ensemble_results.png`.
- `test_model.py` - Performs advanced testing and evaluation on the single LSTM model (`best_model.pt`). It calculates advanced metrics like ROC-AUC, Matthews Correlation Coefficient, and performs threshold optimization. Generates `model_test_results.png`.

### Configuration
- `requirements.txt` - Python dependencies for the project.

## Workflow

1. **Parse the raw ECAR log into CSV:**
   ```bash
   python parse_optc.py
   ```
   *(Note: You may need to edit the `INPUT_JSON` path in the script to point to your specific ECAR file).*

2. **Prepare sequences from the parsed dataset:**
   ```bash
   python prepare_sequences.py
   ```
   This generates the necessary `.npy` tensor files.

3. **Train a single LSTM model:**
   ```bash
   python train_lstm.py
   ```
   *or*
   **Train the Ensemble model:**
   ```bash
   python train_ensemble.py
   ```

4. **Evaluate the single model (optional):**
   ```bash
   python test_model.py
   ```

## Model Architecture

The core architecture used in `train_lstm.py` and `test_model.py` is a Bidirectional LSTM that processes sequences of categorical events. The output of the LSTM passes through layer normalization, dropout layers, and dense layers to produce a final probability of an event being malicious.

The `train_ensemble.py` utilizes variations of this architecture with different hidden sizes (64, 96, 128 units) and layers to create a diverse set of models for better predictive performance.

## Handling Imbalanced Data

The data generated from ECAR logs is highly imbalanced, with benign events vastly outnumbering malicious ones. The training scripts (`train_lstm.py` and `train_ensemble.py`) handle this by using `BCEWithLogitsLoss` with a `pos_weight` parameter, adjusting the loss function to heavily penalize false negatives on the minority (malicious) class.