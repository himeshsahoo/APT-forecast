import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from realtime_monitor import load_checkpoint, run_cycle


class RealtimeMonitorTests(unittest.TestCase):
    def test_run_cycle_emits_alert_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "events.csv"
            state_path = root / "monitor_state.json"
            alerts_path = root / "alerts.jsonl"
            latest_report_path = root / "latest_report.md"

            pd.DataFrame(
                [
                    {
                        "timestamp": "2019-09-23T09:18:00-04:00",
                        "hostname": "SysClient0005.systemia.com",
                        "action": "PROCESS_CREATE",
                        "actor_process": "powershell.exe",
                        "target_process": "mimikatz.exe",
                        "dest_ip": "None",
                        "dest_port": "None",
                        "is_malicious": 1,
                    }
                ]
            ).to_csv(input_path, index=False)

            last_row = run_cycle(
                input_path=input_path,
                state_path=state_path,
                alerts_path=alerts_path,
                latest_report_path=latest_report_path,
                last_row=0,
                batch_size=100,
                risk_threshold=45.0,
            )

            self.assertEqual(last_row, 1)
            self.assertEqual(load_checkpoint(state_path, from_start=False), 1)
            self.assertTrue(alerts_path.exists())
            self.assertTrue(latest_report_path.exists())

            alert = json.loads(alerts_path.read_text(encoding="utf-8").strip())
            self.assertEqual(alert["risk"]["alert_count"], 1)
            self.assertTrue(alert["mitre"])


if __name__ == "__main__":
    unittest.main()
