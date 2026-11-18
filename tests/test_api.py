
from fastapi.testclient import TestClient

from api.main import app

c = TestClient(app)


def test_health():
    assert c.get("/health").json() == {"status": "ok"}


def test_predict():
    r = c.post("/predict", json={"sequence": [[1, 2, 3, 4, 5]] * 10})
    assert r.status_code == 200 and r.json()["label"] in (0, 1)


def test_predict_bad_shape():
    assert c.post("/predict", json={"sequence": [[1, 2, 3]] * 10}).status_code == 422
