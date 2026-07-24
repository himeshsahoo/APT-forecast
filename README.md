# APT Forecast

APT Forecast is an AI-powered cybersecurity project that analyzes security events and tries to detect early signs of an Advanced Persistent Threat, or APT.

An APT is not a normal virus that immediately damages a system. It is usually a slow, stealthy attack where the attacker enters a network, hides, studies the environment, steals credentials, moves to other machines, and may finally exfiltrate data. This project tries to identify those suspicious patterns from logs and explain what may be happening.

In simple terms:

```text
Security logs go in
        |
        v
AI + threat intelligence analyze them
        |
        v
The system produces alerts, risk score, attack timeline, MITRE mapping, and response suggestions
```

## What This Project Does

APT Forecast currently has two main parts.

First, it has a machine learning pipeline. This part prepares security event sequences and trains LSTM-based models to classify activity as benign or malicious.

Second, it has a SOC monitoring layer. SOC means Security Operations Center. This part reads parsed security events, creates analyst-friendly alerts, maps suspicious behavior to MITRE ATT&CK techniques, estimates risk, predicts the likely next attacker stage, and recommends safe response actions.

The goal is to make the project look less like only a model-training experiment and more like a real AI security platform.

## What Problem It Solves

Security teams receive huge amounts of logs. Most events are normal, but a few may be signs of attack. Looking through everything manually is difficult.

APT Forecast helps by answering questions like:

- Is this host showing malicious behavior?
- What kind of attacker activity is this?
- Which MITRE ATT&CK technique does it resemble?
- How risky is the situation?
- What is the attacker likely to do next?
- What should a security analyst do now?

Example output:

```text
240 malicious events across 1 host(s). Overall risk is 93.2% (Critical).

Current Stage: Tool Transfer
Predicted Next Stage: Data Exfiltration
MITRE Technique: T1105 Ingress Tool Transfer
Recommended Response: Quarantine transferred binaries, block suspicious destination, start forensic collection
```

## Current Architecture

```text
Raw OpTC ECAR logs
        |
        v
parse_optc.py
        |
        v
optc_parsed_sequences.csv
        |
        +--> prepare_sequences.py
        |          |
        |          v
        |      X_train.npy / X_test.npy
        |          |
        |          v
        |      train_lstm.py / train_ensemble.py
        |          |
        |          v
        |      best_model.pt + evaluation plots
        |
        +--> analyze_alerts.py
        |          |
        |          v
        |      soc_report.md + soc_report.json
        |
        +--> realtime_monitor.py
                   |
                   v
               alerts.jsonl + latest_soc_report.md
```

## Input Data

The current input is a parsed security event CSV file:

```text
optc_parsed_sequences.csv
```

The expected columns are:

| Column | Meaning |
| --- | --- |
| `timestamp` | When the event happened. |
| `hostname` | Machine where the event happened. |
| `action` | Event type, such as `PROCESS_CREATE`, `MODULE_LOAD`, or `FILE_READ`. |
| `actor_process` | Process that caused the event. |
| `target_process` | Process or file affected by the event. |
| `dest_ip` | Destination IP address for network activity. |
| `dest_port` | Destination port for network activity. |
| `is_malicious` | `1` for malicious, `0` for benign. |

Example:

```csv
timestamp,hostname,action,actor_process,target_process,dest_ip,dest_port,is_malicious
2019-09-23T09:11:03.11-04:00,SysClient0004.systemia.com,THREAD_CREATE,None,None,None,None,0
```

Right now the CSV is created by `parse_optc.py` from OpTC ECAR logs. In a more production-ready version, the same format can be produced by live collectors from Sysmon, Zeek, Suricata, Windows Event Logs, or Linux audit logs.

## How Realtime Monitoring Works

The realtime monitor is a long-running loop. It does not scan your computer like antivirus software yet. Instead, it watches an append-only parsed event CSV.

Run it with:

```bash
python realtime_monitor.py --input optc_parsed_sequences.csv --poll-seconds 5
```

What happens internally:

```text
Start monitor
   |
   v
Read monitor_state.json
   |
   v
Find the last processed row
   |
   v
Read only new rows from the CSV
   |
   v
Analyze the new events
   |
   v
Calculate risk and map MITRE techniques
   |
   v
If risk is high enough, write an alert
   |
   v
Wait a few seconds and repeat
```

The monitor creates these runtime files:

| File | Purpose |
| --- | --- |
| `monitor_state.json` | Stores the last processed row so old events are not analyzed again. |
| `alerts.jsonl` | Stores alert history in machine-readable JSON lines format. |
| `latest_soc_report.md` | Stores the newest human-readable SOC report. |

Useful realtime commands:

```bash
# Process the file from the beginning once, then stop
python realtime_monitor.py --from-start --once

# Run continuously and check every 2 seconds
python realtime_monitor.py --poll-seconds 2

# Only create alerts when risk is at least 70%
python realtime_monitor.py --risk-threshold 70
```

## Offline SOC Report

If you do not want the long-running monitor, you can generate one report manually:

```bash
python analyze_alerts.py --input optc_parsed_sequences.csv --output soc_report.md --json-output soc_report.json
```

This produces:

| File | Purpose |
| --- | --- |
| `soc_report.md` | Human-readable incident report. |
| `soc_report.json` | Machine-readable structured report. |

The report includes:

- Executive summary
- Overall risk score
- Severity, likelihood, business impact, and confidence
- MITRE ATT&CK mapping
- Attack timeline
- Predicted next attacker stage
- Human-readable detection reasons
- Recommended response actions

## Machine Learning Pipeline

The project also includes an LSTM-based model pipeline.

Run the full pipeline like this:

```bash
python parse_optc.py
python prepare_sequences.py
python train_lstm.py
python train_ensemble.py
python test_model.py
```

What each step does:

| Step | Script | Explanation |
| --- | --- | --- |
| 1 | `parse_optc.py` | Reads raw OpTC ECAR JSON logs and converts them into a clean CSV. |
| 2 | `prepare_sequences.py` | Groups events by host and time, then creates fixed-length event sequences. |
| 3 | `train_lstm.py` | Trains the main bidirectional LSTM model. |
| 4 | `train_ensemble.py` | Trains three LSTM models and combines their predictions. |
| 5 | `test_model.py` | Evaluates the model using ROC-AUC, precision, recall, F1, MCC, threshold analysis, and plots. |

## SOC Intelligence Features

APT Forecast now includes reusable SOC intelligence modules inside `src/apt_forecast/`.

| Module | Purpose |
| --- | --- |
| `soc_engine.py` | Builds risk scores, timelines, reports, forecasts, and response recommendations. |
| `threat_intel.py` | Maps suspicious activity to MITRE ATT&CK techniques and attack stages. |
| `explainability.py` | Provides rule-based explanations and optional SHAP/LIME hooks. |

The SOC engine is intentionally separate from the training scripts. This makes it easier to later add a FastAPI backend, dashboard, database, or live log collector without rewriting the model code.

## Safety Note

The project does not automatically block IPs, kill processes, disable accounts, or modify a real system.

All response actions are currently either:

- `recommend`: show the analyst what should be done
- `simulate`: describe what the system would do in a controlled SOC workflow

This is intentional. In cybersecurity software, automatic destructive actions should require strict approval and policy controls.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
python -m unittest tests.test_soc_engine_unittest tests.test_realtime_monitor_unittest
```

## Project Files

| File or Folder | Description |
| --- | --- |
| `parse_optc.py` | Raw log parser and IOC-based labeler. |
| `prepare_sequences.py` | Sequence dataset builder for ML training. |
| `train_lstm.py` | Main LSTM training script. |
| `train_ensemble.py` | Ensemble model training script. |
| `test_model.py` | Evaluation and visualization script. |
| `analyze_alerts.py` | Offline SOC report generator. |
| `realtime_monitor.py` | Realtime append-only CSV monitoring loop. |
| `src/apt_forecast/` | Reusable SOC intelligence package. |
| `tests/` | Unit tests for SOC reporting and realtime monitoring. |
| `docs/PROJECT_REVIEW.md` | Full technical review and roadmap. |

## How To Explain This Project

A simple explanation:

> APT Forecast is an AI cybersecurity platform that analyzes security logs to detect advanced persistent threat behavior. It trains LSTM models on event sequences, then adds a SOC intelligence layer that explains alerts using MITRE ATT&CK, calculates risk, builds attack timelines, forecasts the next attacker stage, and recommends safe response actions.

A more technical explanation:

> The system parses OpTC ECAR telemetry into structured events, creates host-aware temporal sequences, trains bidirectional LSTM and ensemble models for malicious activity detection, and provides a streaming SOC monitor that processes newly appended events with checkpointing. The analyst layer maps detections to MITRE ATT&CK, produces risk-scored incident reports, and keeps automated responses in simulation mode for safety.

## Current Limitations

This is still a research prototype, not a finished enterprise product.

Current limitations:

- Realtime input is currently a parsed CSV, not direct live Windows/Sysmon/Zeek ingestion.
- Labels are based on known OpTC red-team indicators.
- Feature encoding should be improved with categorical embeddings or hashing.
- Encoders and model metadata should be persisted for reliable inference.
- There is no web dashboard yet.
- There is no FastAPI backend yet.
- SHAP and LIME are optional hooks, not fully integrated into every model prediction.

## Roadmap

Version 2.0:

- SOC report generation
- MITRE ATT&CK mapping
- Risk scoring
- Realtime CSV monitor
- Safe response recommendations

Version 2.1:

- Persist encoders and feature metadata
- Add chronological host-aware train/test splitting
- Add stronger model explanations
- Add more parser and inference tests

Version 2.2:

- Add FastAPI inference API
- Add dashboard with threat count, risk score, timeline, MITRE mapping, and recent alerts
- Add SQLite or PostgreSQL alert storage

Version 3.0:

- Add Transformer-based temporal model
- Add MLflow experiment tracking
- Add Docker deployment
- Add CI/CD checks

Version 4.0:

- Add live Sysmon, Zeek, Suricata, or Windows Event Log collectors
- Add threat intelligence enrichment for IPs, domains, CVEs, malware families, and threat actors
- Add graph neural networks for attack-path modeling
- Add reinforcement learning for simulated response planning
