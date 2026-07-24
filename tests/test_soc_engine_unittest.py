import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.apt_forecast.soc_engine import build_soc_report, render_markdown_report


class SocEngineTests(unittest.TestCase):
    def test_soc_report_maps_mitre_and_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "events.csv"
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
                    },
                    {
                        "timestamp": "2019-09-23T09:20:00-04:00",
                        "hostname": "SysClient0005.systemia.com",
                        "action": "THREAD_REMOTE_CREATE",
                        "actor_process": "psinject.exe",
                        "target_process": "lsass.exe",
                        "dest_ip": "None",
                        "dest_port": "None",
                        "is_malicious": 1,
                    },
                ]
            ).to_csv(csv_path, index=False)

            report = build_soc_report(csv_path)

            technique_ids = {item["id"] for item in report["mitre"]}
            self.assertTrue({"T1059", "T1003", "T1055"}.issubset(technique_ids))
            self.assertIn(report["risk"]["severity"], {"High", "Critical"})
            self.assertTrue(report["forecast"]["predicted_next_stage"])
            self.assertTrue(any(item["mode"] == "simulate" for item in report["recommended_responses"]))
            self.assertIn("APT Forecast SOC Report", render_markdown_report(report))


if __name__ == "__main__":
    unittest.main()
