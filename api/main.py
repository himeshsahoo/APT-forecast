
import sys
from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from apt_forecast.soc_engine import build_soc_report_from_events
from api.model import APTForecaster

model = APTForecaster()
model.load_state_dict(torch.load(ROOT / "best_model.pt", map_location="cpu"))
model.eval()

app = FastAPI(title="APT Forecast API")


class Seq(BaseModel):
    sequence: list[list[float]] = Field(min_length=10, max_length=10)


class Event(BaseModel):
    timestamp: str
    hostname: str
    action: str
    actor_process: str = "None"
    target_process: str = "None"
    dest_ip: str = "None"
    dest_port: str = "None"
    is_malicious: int = 0


class Events(BaseModel):
    events: list[Event] = Field(min_length=1, max_length=5000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(b: Seq):
    if any(len(r) != 5 for r in b.sequence):
        raise HTTPException(422, "each row needs 5 values")
    x = torch.tensor([b.sequence], dtype=torch.float32)
    with torch.no_grad():
        p = torch.sigmoid(model(x)).item()
    return {"malicious_probability": round(p, 4), "label": int(p >= 0.5)}


@app.post("/analyze")
def analyze(b: Events):
    df = pd.DataFrame([e.model_dump() for e in b.events])
    return build_soc_report_from_events(df)
