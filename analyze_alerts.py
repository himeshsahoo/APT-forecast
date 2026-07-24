from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.apt_forecast import build_soc_report, render_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an analyst-facing SOC report from parsed OpTC events.")
    parser.add_argument("--input", default="optc_parsed_sequences.csv", help="Parsed event CSV.")
    parser.add_argument("--output", default="soc_report.md", help="Markdown report output path.")
    parser.add_argument("--json-output", default="soc_report.json", help="JSON report output path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for faster local analysis.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_soc_report(args.input, limit=args.limit)

    markdown_path = Path(args.output)
    json_path = Path(args.json_output)
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {markdown_path}")
    print(f"Wrote {json_path}")
    print(report["executive_summary"])


if __name__ == "__main__":
    main()
