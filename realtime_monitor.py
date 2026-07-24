from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.apt_forecast.soc_engine import build_soc_report_from_events, render_markdown_report


EVENT_COLUMNS = [
    "timestamp",
    "hostname",
    "action",
    "actor_process",
    "target_process",
    "dest_ip",
    "dest_port",
    "is_malicious",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run APT Forecast as a real-time SOC monitor over an append-only CSV.")
    parser.add_argument("--input", default="optc_parsed_sequences.csv", help="Append-only parsed event CSV to monitor.")
    parser.add_argument("--state", default="monitor_state.json", help="Checkpoint file storing the last processed row.")
    parser.add_argument("--alerts", default="alerts.jsonl", help="JSONL output for emitted SOC alerts.")
    parser.add_argument("--latest-report", default="latest_soc_report.md", help="Markdown file for the latest alert report.")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="Seconds between checks.")
    parser.add_argument("--batch-size", type=int, default=5000, help="Maximum new rows to process per poll.")
    parser.add_argument("--risk-threshold", type=float, default=45.0, help="Minimum risk score required to emit an alert.")
    parser.add_argument("--from-start", action="store_true", help="Ignore checkpoint and process from row 0.")
    parser.add_argument("--once", action="store_true", help="Run one polling cycle and exit.")
    return parser.parse_args()


def load_checkpoint(path: Path, from_start: bool) -> int:
    if from_start or not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    return int(data.get("last_row", 0))


def save_checkpoint(path: Path, last_row: int) -> None:
    path.write_text(json.dumps({"last_row": last_row}, indent=2), encoding="utf-8")


def read_new_events(input_path: Path, start_row: int, batch_size: int) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input log does not exist: {input_path}")

    return pd.read_csv(
        input_path,
        usecols=EVENT_COLUMNS,
        skiprows=range(1, start_row + 1),
        nrows=batch_size,
    ).fillna("None")


def append_alert(alerts_path: Path, report: dict[str, Any], processed_rows: int) -> None:
    envelope = {
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "processed_rows": processed_rows,
        "summary": report["executive_summary"],
        "risk": report["risk"],
        "forecast": report["forecast"],
        "mitre": report["mitre"],
        "top_affected_hosts": report["top_affected_hosts"],
        "recommended_responses": report["recommended_responses"],
    }
    with alerts_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(envelope) + "\n")


def run_cycle(
    input_path: Path,
    state_path: Path,
    alerts_path: Path,
    latest_report_path: Path,
    last_row: int,
    batch_size: int,
    risk_threshold: float,
) -> int:
    events = read_new_events(input_path, last_row, batch_size)
    if events.empty:
        print(f"No new events. Last processed row: {last_row}")
        return last_row

    report = build_soc_report_from_events(events)
    new_last_row = last_row + len(events)
    save_checkpoint(state_path, new_last_row)

    risk = float(report["risk"]["overall_risk"])
    alert_count = int(report["risk"]["alert_count"])
    print(
        f"Processed {len(events)} new event(s). "
        f"Alerts: {alert_count}. Risk: {risk}% ({report['risk']['severity']})."
    )

    if alert_count > 0 and risk >= risk_threshold:
        append_alert(alerts_path, report, processed_rows=len(events))
        latest_report_path.write_text(render_markdown_report(report), encoding="utf-8")
        print(f"ALERT emitted -> {alerts_path}")
    return new_last_row


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    state_path = Path(args.state)
    alerts_path = Path(args.alerts)
    latest_report_path = Path(args.latest_report)
    last_row = load_checkpoint(state_path, from_start=args.from_start)

    print("APT Forecast realtime SOC monitor started.")
    print(f"Watching: {input_path}")
    print(f"Checkpoint: {state_path} (starting row {last_row})")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            last_row = run_cycle(
                input_path=input_path,
                state_path=state_path,
                alerts_path=alerts_path,
                latest_report_path=latest_report_path,
                last_row=last_row,
                batch_size=args.batch_size,
                risk_threshold=args.risk_threshold,
            )
            if args.once:
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\nMonitor stopped by user.")


if __name__ == "__main__":
    main()
