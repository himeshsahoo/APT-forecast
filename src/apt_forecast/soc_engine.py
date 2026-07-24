from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .threat_intel import (
    forecast_next_stage,
    infer_stage,
    map_event_to_techniques,
    severity_from_score,
)


@dataclass(frozen=True)
class TimelineEvent:
    timestamp: str
    hostname: str
    stage: str
    action: str
    evidence: str


@dataclass(frozen=True)
class ResponseRecommendation:
    action: str
    mode: str
    rationale: str


def load_events(csv_path: str | Path, limit: int | None = None) -> pd.DataFrame:
    columns = [
        "timestamp",
        "hostname",
        "action",
        "actor_process",
        "target_process",
        "dest_ip",
        "dest_port",
        "is_malicious",
    ]
    return pd.read_csv(csv_path, usecols=columns, nrows=limit).fillna("None")


def _event_text(row: pd.Series) -> str:
    values = [row.get("action"), row.get("actor_process"), row.get("target_process"), row.get("dest_ip")]
    return " | ".join(str(value) for value in values if str(value) not in {"", "None", "nan"})


def _important_features(alerts: pd.DataFrame) -> list[dict[str, Any]]:
    feature_votes: Counter[str] = Counter()
    for _, row in alerts.iterrows():
        techniques = map_event_to_techniques(
            row.get("action"),
            row.get("actor_process"),
            row.get("target_process"),
            row.get("dest_ip"),
            row.get("dest_port"),
        )
        for technique in techniques:
            feature_votes[f"{technique.technique_id} {technique.name}"] += 1
        if int(row.get("is_malicious", 0)) == 1:
            feature_votes["Known OpTC red-team IOC match"] += 1
    return [{"feature": key, "weight": value} for key, value in feature_votes.most_common(8)]


def _mitre_summary(alerts: pd.DataFrame) -> list[dict[str, Any]]:
    techniques: dict[str, dict[str, Any]] = {}
    for _, row in alerts.iterrows():
        for technique in map_event_to_techniques(
            row.get("action"),
            row.get("actor_process"),
            row.get("target_process"),
        ):
            item = techniques.setdefault(
                technique.technique_id,
                {
                    "id": technique.technique_id,
                    "name": technique.name,
                    "tactic": technique.tactic,
                    "stage": technique.stage,
                    "count": 0,
                    "response_actions": set(),
                },
            )
            item["count"] += 1
            item["response_actions"].update(technique.response_actions)

    result: list[dict[str, Any]] = []
    for item in sorted(techniques.values(), key=lambda value: value["count"], reverse=True):
        item["response_actions"] = sorted(item["response_actions"])
        result.append(item)
    return result


def _timeline(alerts: pd.DataFrame, max_events: int) -> list[TimelineEvent]:
    ordered = alerts.sort_values(["timestamp", "hostname"]).head(max_events)
    events: list[TimelineEvent] = []
    for _, row in ordered.iterrows():
        stage = infer_stage(row.get("action"), row.get("actor_process"), row.get("target_process"))
        events.append(
            TimelineEvent(
                timestamp=str(row.get("timestamp")),
                hostname=str(row.get("hostname")),
                stage=stage,
                action=str(row.get("action")),
                evidence=_event_text(row),
            )
        )
    return events


def _risk_score(total_events: int, alerts: pd.DataFrame, mitre: list[dict[str, Any]]) -> dict[str, Any]:
    alert_count = len(alerts)
    affected_hosts = alerts["hostname"].nunique() if alert_count else 0
    technique_count = len(mitre)
    malicious_ratio = alert_count / max(total_events, 1)

    if alert_count == 0:
        score = 10.0
    else:
        score = min(
            100.0,
            35.0
            + min(alert_count * 0.15, 35.0)
            + min(affected_hosts * 10.0, 25.0)
            + min(technique_count * 12.0, 30.0)
            + min(malicious_ratio * 2500.0, 15.0),
        )
    confidence = min(0.99, 0.55 + min(alert_count / 100.0, 0.30) + min(technique_count * 0.03, 0.14))
    return {
        "overall_risk": round(score, 2),
        "severity": severity_from_score(score),
        "likelihood": "High" if score >= 70 else "Medium" if score >= 45 else "Low",
        "business_impact": "High" if affected_hosts >= 3 or technique_count >= 3 else "Medium",
        "confidence": round(confidence, 2),
        "affected_hosts": int(affected_hosts),
        "alert_count": int(alert_count),
        "malicious_ratio": round(malicious_ratio, 6),
    }


def _responses(risk: dict[str, Any], mitre: list[dict[str, Any]]) -> list[ResponseRecommendation]:
    recommendations = [
        ResponseRecommendation("Alert administrator", "recommend", "Human review is required before containment."),
        ResponseRecommendation("Start forensic collection", "recommend", "Preserve volatile evidence before remediation."),
    ]
    if risk["severity"] in {"High", "Critical"}:
        recommendations.extend(
            [
                ResponseRecommendation("Isolate affected host", "simulate", "Contain spread while avoiding automatic disruption."),
                ResponseRecommendation("Reset credentials", "recommend", "Credential access or lateral movement may be present."),
            ]
        )
    for item in mitre[:5]:
        for action in item["response_actions"]:
            recommendations.append(ResponseRecommendation(action, "recommend", f"Mapped from {item['id']} {item['name']}."))
    return recommendations


def build_soc_report_from_events(events: pd.DataFrame, max_timeline_events: int = 25) -> dict[str, Any]:
    events = events.fillna("None")
    alerts = events[events["is_malicious"].astype(int) == 1].copy()
    mitre = _mitre_summary(alerts)
    timeline = _timeline(alerts, max_timeline_events)
    current_stage = timeline[-1].stage if timeline else "No active threat"
    next_stage, next_probability = forecast_next_stage(current_stage)
    risk = _risk_score(len(events), alerts, mitre)
    features = _important_features(alerts)

    return {
        "executive_summary": (
            f"{risk['alert_count']} malicious events across {risk['affected_hosts']} host(s). "
            f"Overall risk is {risk['overall_risk']}% ({risk['severity']})."
        ),
        "risk": risk,
        "forecast": {
            "current_stage": current_stage,
            "predicted_next_stage": next_stage,
            "probability": next_probability,
        },
        "mitre": mitre,
        "timeline": [asdict(item) for item in timeline],
        "important_features": features,
        "recommended_responses": [asdict(item) for item in _responses(risk, mitre)],
        "top_affected_hosts": alerts["hostname"].value_counts().head(10).to_dict(),
        "top_actions": alerts["action"].value_counts().head(10).to_dict(),
    }


def build_soc_report(csv_path: str | Path, limit: int | None = None, max_timeline_events: int = 25) -> dict[str, Any]:
    events = load_events(csv_path, limit=limit)
    return build_soc_report_from_events(events, max_timeline_events=max_timeline_events)

def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# APT Forecast SOC Report",
        "",
        f"## Executive Summary",
        report["executive_summary"],
        "",
        "## Risk Score",
        f"- Overall risk: {report['risk']['overall_risk']}%",
        f"- Severity: {report['risk']['severity']}",
        f"- Likelihood: {report['risk']['likelihood']}",
        f"- Business impact: {report['risk']['business_impact']}",
        f"- Confidence: {report['risk']['confidence']}",
        "",
        "## Attack Forecast",
        f"- Current stage: {report['forecast']['current_stage']}",
        f"- Predicted next stage: {report['forecast']['predicted_next_stage']}",
        f"- Probability: {report['forecast']['probability']}",
        "",
        "## MITRE ATT&CK Mapping",
    ]
    for item in report["mitre"] or [{"id": "N/A", "name": "No mapped technique", "tactic": "N/A", "count": 0}]:
        lines.append(f"- {item['id']} {item['name']} ({item['tactic']}): {item['count']} event(s)")

    lines.extend(["", "## Why It Was Detected"])
    for item in report["important_features"] or [{"feature": "No malicious indicators found", "weight": 0}]:
        lines.append(f"- {item['feature']}: {item['weight']}")

    lines.extend(["", "## Attack Timeline"])
    for item in report["timeline"][:20]:
        lines.append(f"- {item['timestamp']} | {item['hostname']} | {item['stage']} | {item['action']}")

    lines.extend(["", "## Recommended Response"])
    for item in report["recommended_responses"]:
        lines.append(f"- [{item['mode']}] {item['action']}: {item['rationale']}")
    return "\n".join(lines) + "\n"


