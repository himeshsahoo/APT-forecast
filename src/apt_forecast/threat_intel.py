from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MitreTechnique:
    technique_id: str
    name: str
    tactic: str
    indicators: tuple[str, ...]
    stage: str
    response_actions: tuple[str, ...]


MITRE_TECHNIQUES: tuple[MitreTechnique, ...] = (
    MitreTechnique(
        technique_id="T1059",
        name="Command and Scripting Interpreter",
        tactic="Execution",
        indicators=("powershell", "cmd.exe", "wscript", "cscript", "vbx", "bat"),
        stage="Execution",
        response_actions=("Capture command line and parent process", "Restrict script execution policy"),
    ),
    MitreTechnique(
        technique_id="T1003",
        name="OS Credential Dumping",
        tactic="Credential Access",
        indicators=("mimikatz", "lsass", "sekurlsa", "credential", "dump"),
        stage="Credential Dumping",
        response_actions=("Reset exposed credentials", "Collect memory image for forensics"),
    ),
    MitreTechnique(
        technique_id="T1055",
        name="Process Injection",
        tactic="Defense Evasion",
        indicators=("psinject", "inject", "thread_remote_create", "remote_thread"),
        stage="Privilege Escalation",
        response_actions=("Kill suspicious process tree", "Hunt for injected process ancestry"),
    ),
    MitreTechnique(
        technique_id="T1021",
        name="Remote Services",
        tactic="Lateral Movement",
        indicators=("invoke_wmi", "wmic", "psexec", "remote", "admin$"),
        stage="Lateral Movement",
        response_actions=("Disable compromised remote sessions", "Audit administrator logons"),
    ),
    MitreTechnique(
        technique_id="T1105",
        name="Ingress Tool Transfer",
        tactic="Command and Control",
        indicators=("nc.exe", "transfer", "download", "upload", "update.exe"),
        stage="Tool Transfer",
        response_actions=("Block suspicious destination", "Quarantine transferred binaries"),
    ),
    MitreTechnique(
        technique_id="T1041",
        name="Exfiltration Over C2 Channel",
        tactic="Exfiltration",
        indicators=("export.zip", "allgone.zip", "zip", "archive"),
        stage="Data Exfiltration",
        response_actions=("Block outbound transfer", "Review data access scope"),
    ),
)


STAGE_ORDER: tuple[str, ...] = (
    "Initial Access",
    "Execution",
    "Privilege Escalation",
    "Credential Dumping",
    "Lateral Movement",
    "Tool Transfer",
    "Data Exfiltration",
    "Impact",
)


def normalize_text(values: Iterable[object]) -> str:
    return " ".join("" if value is None else str(value).lower() for value in values)


def map_event_to_techniques(*values: object) -> list[MitreTechnique]:
    text = normalize_text(values)
    matches: list[MitreTechnique] = []
    for technique in MITRE_TECHNIQUES:
        if any(indicator in text for indicator in technique.indicators):
            matches.append(technique)
    return matches


def infer_stage(*values: object) -> str:
    matches = map_event_to_techniques(*values)
    if matches:
        return matches[0].stage
    text = normalize_text(values)
    if "network" in text or "flow" in text or "connect" in text:
        return "Command and Control"
    if "process_create" in text or "module_load" in text or "thread_create" in text:
        return "Execution"
    return "Reconnaissance"


def forecast_next_stage(current_stage: str) -> tuple[str, float]:
    try:
        index = STAGE_ORDER.index(current_stage)
    except ValueError:
        return "Execution", 0.55
    if index >= len(STAGE_ORDER) - 1:
        return "Impact", 0.58
    probability = max(0.62, 0.91 - (index * 0.04))
    return STAGE_ORDER[index + 1], round(probability, 2)


def severity_from_score(score: float) -> str:
    if score >= 85:
        return "Critical"
    if score >= 70:
        return "High"
    if score >= 45:
        return "Medium"
    return "Low"
