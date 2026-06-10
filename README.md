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