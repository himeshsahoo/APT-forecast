# APT Forecast

APT Forecast is an AI-powered cybersecurity research prototype for detecting and forecasting Advanced Persistent Threat activity from system logs and network/security events.

The project currently contains:

- An OpTC ECAR log parser with red-team IOC labeling.
- A host-aware sequence preparation pipeline.
- Bidirectional LSTM training.
- A three-model LSTM ensemble.
- Advanced evaluation plots and threshold analysis.
- A SOC intelligence layer that generates MITRE ATT&CK mapping, attack timeline, risk scoring, forecasted next stage, and response recommendations.

## Architecture

```text
Raw OpTC ECAR logs
        |
        v
parse_optc.py
        |
        v
optc_parsed_sequences.csv
        |
        +--> prepare_sequences.py --> X/y tensors --> LSTM / Ensemble --> model + plots
        |
        +--> analyze_alerts.py --> SOC Markdown/JSON report
```

## Key Files

| File | Purpose |
| --- | --- |
| `parse_optc.py` | Parses ECAR JSON lines into a structured CSV and labels known malicious events. |
| `prepare_sequences.py` | Sorts events by host/time, encodes categorical fields, and creates fixed-length tensors. |
| `train_lstm.py` | Trains the main bidirectional LSTM detector. |
| `train_ensemble.py` | Trains small, medium, and large LSTM models and evaluates voting strategies. |
| `test_model.py` | Runs model evaluation, threshold optimization, and visualization generation. |
| `analyze_alerts.py` | Generates an analyst-facing SOC report from parsed events. |
| `src/apt_forecast/` | Reusable threat intelligence, risk scoring, forecasting, and explainability utilities. |
| `docs/PROJECT_REVIEW.md` | Full architecture review, ratings, risks, and roadmap. |

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Existing ML Pipeline

```bash
python parse_optc.py
python prepare_sequences.py
python train_lstm.py
python train_ensemble.py
python test_model.py
```

## Generate a SOC Report

```bash
python analyze_alerts.py --input optc_parsed_sequences.csv --output soc_report.md --json-output soc_report.json
```

The report includes:

- Executive summary
- Overall risk score
- Threat severity, likelihood, business impact, and confidence
- MITRE ATT&CK techniques
- Attack timeline
- Forecasted next attacker stage
- Human-readable reasons for detection
- Simulated or recommended response actions

Response actions are recommendation/simulation only. The project does not automatically block IPs, kill processes, disable accounts, or perform destructive containment.


## Run Realtime SOC Monitoring

The project can now run as a long-lived SOC monitor over an append-only parsed event CSV:

```bash
python realtime_monitor.py --input optc_parsed_sequences.csv --poll-seconds 5
```

How it works:

1. The monitor reads only rows that have not been processed yet.
2. It stores progress in `monitor_state.json`.
3. If a risky malicious batch is detected, it appends an alert to `alerts.jsonl`.
4. It writes the newest human-readable incident report to `latest_soc_report.md`.
5. It keeps running until you stop it with `Ctrl+C`.

Useful commands:

```bash
# Process from the beginning once, then exit
python realtime_monitor.py --from-start --once

# Run continuously and check every 2 seconds
python realtime_monitor.py --poll-seconds 2

# Raise alerts only when risk is at least 70%
python realtime_monitor.py --risk-threshold 70
```

This is a realtime monitoring loop, but the current input is still a parsed CSV. The next production step is to connect the same monitor interface to Sysmon, Zeek, Suricata, Windows Event Logs, or Linux audit logs.

## Current Model

The current model is a bidirectional LSTM over fixed-length event sequences. It uses dropout, layer normalization, class weighting for heavy imbalance, ROC-AUC evaluation, and dynamic threshold analysis.

This is a strong prototype, but the next model upgrade should move toward categorical embeddings plus a Transformer encoder or temporal attention model. See `docs/PROJECT_REVIEW.md` for the detailed roadmap.

## Enterprise Roadmap

Short-term:

- Persist encoders and feature metadata.
- Add chronological host-aware splitting.
- Add model-level SHAP/LIME explanations where practical.
- Add parser and inference tests.

Mid-term:

- Add FastAPI inference endpoints.
- Add an enterprise dashboard for risk, timeline, MITRE mapping, alerts, and recommendations.
- Add MLflow experiment tracking and Docker.

Long-term:

- Add live ingestion from Sysmon, Zeek, Suricata, or Windows Event Logs.
- Add threat intelligence enrichment for IP/domain reputation, CVEs, malware families, and actor profiles.
- Add graph neural networks for attack path modeling and reinforcement learning for simulated response selection.

