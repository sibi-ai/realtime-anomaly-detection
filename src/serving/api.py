"""FastAPI inference service — async batch prediction endpoint."""
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import structlog

app = FastAPI(title="Anomaly Detection API")
log = structlog.get_logger()

# Initialise runner lazily (model path from env/config in production)
_runner = None


def get_runner():
    global _runner
    if _runner is None:
        from src.inference.onnx_runner import ONNXInferenceRunner
        import os
        path = os.environ.get("MODEL_PATH", "models/anomaly_detector_int8.onnx")
        _runner = ONNXInferenceRunner(path)
    return _runner


class PredictRequest(BaseModel):
    features: list[list[float]]
    threshold: float = 0.5


class PredictResponse(BaseModel):
    scores: list[float]
    labels: list[int]


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    runner = get_runner()
    X = np.array(req.features, dtype=np.float32)
    scores = runner.predict(X).tolist()
    labels = [int(s >= req.threshold) for s in scores]
    log.info("prediction", n=len(scores), anomalies=sum(labels))
    return PredictResponse(scores=scores, labels=labels)


@app.get("/health")
async def health():
    return {"status": "ok"}
