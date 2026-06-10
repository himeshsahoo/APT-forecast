# APT Forecast Project

This project parses OpTC ECAR event logs, labels likely malicious events using known indicators of compromise, prepares sequence data, and trains an LSTM model for forecasting.

## Files

- `parse_optc.py` - parse ECAR JSON-lines into a CSV dataset
- `prepare_sequences.py` - build model-ready train/test sequences
- `train_lstm.py` - train the forecasting model
- `inspect_data.py` - inspect the prepared dataset
- `requirements.txt` - Python dependencies

## Typical workflow

1. Parse the raw ECAR log into CSV.
2. Prepare sequences from the parsed dataset.
3. Train the LSTM model.

## Usage

Parse the log:

```bash
python parse_optc.py "AIA-1-25.ecar-last.json/AIA-1-25.ecar.json" optc_parsed_sequences.csv --max-lines 500000
```

Prepare sequences:

```bash
python prepare_sequences.py
```

Train the model:

```bash
python train_lstm.py
```

## Notes

- The parser expects ECAR data in JSON-lines format.
- Large inputs are processed in chunks to keep memory usage lower.

## Model performance (latest run)

The following scores were produced by running `python train_lstm.py` on this project data:

- Confusion matrix: `[[70433, 29469], [1, 47]]`
- Accuracy: `0.71`
- Benign (0) F1-score: `0.83`
- Malicious (1) precision: `0.00`
- Malicious (1) recall: `0.98`
- Malicious (1) F1-score: `0.00`

Interpretation in simple terms:

- The model catches almost all malicious events (`47/48`), so recall for malicious is very high.
- It also flags many benign events as malicious (`29469` false positives), so malicious precision is very low.
- This means the model is currently sensitive, but not precise, for malicious detection.

## How the scoring works

After training, the script evaluates on `X_test.npy` and `y_test.npy`.

1. The model outputs a probability between 0 and 1.
2. Any value `>= 0.5` is converted to malicious (`1`); otherwise benign (`0`).
3. Predictions are compared against true labels to compute:
	- Confusion matrix (`TN, FP, FN, TP`)
	- Precision, recall, and F1-score per class
	- Overall accuracy

To reproduce the score:

```bash
python train_lstm.py
```