# Real-Time Anomaly Detection

Production anomaly detection and classification system for high-throughput traffic.
Built for sub-millisecond inference latency with online learning and concept drift handling.

## Features

- XGBoost / LightGBM ensemble on large-scale structured and categorical data
- ONNX export + INT8 quantisation for sub-millisecond inference
- Online / incremental learning — model updates from live data streams
- Concept drift detection — triggers targeted retraining automatically
- Shadow deployment and A/B testing framework for safe model rollouts
- SHAP-based prediction explanations
- FastAPI serving with async batch inference

## Architecture

```
Incoming requests
      ↓
Feature extraction  →  In-memory feature cache
      ↓
ONNX Runtime (INT8)  →  Prediction + confidence
      ↓
Drift monitor  →  Triggers retraining if drift detected
      ↓
Shadow model (A/B)  →  Gradual traffic shift
```

## Stack

| Component | Technology |
|-----------|-----------|
| Models | XGBoost, LightGBM, scikit-learn |
| Inference | ONNX Runtime (INT8 quantised) |
| Online learning | River |
| Drift detection | Evidently, custom KS-test |
| Serving | FastAPI, uvicorn |
| Explainability | SHAP |
| MLOps | MLflow, GitHub Actions |

## Project Structure

```
├── src/
│   ├── models/         # Training, evaluation, quantisation
│   ├── inference/      # ONNX runtime wrapper, batching
│   ├── drift/          # Drift detection and retraining triggers
│   ├── online/         # Incremental learning pipeline
│   └── serving/        # FastAPI endpoints
├── notebooks/          # EDA, model development, error analysis
├── configs/
├── tests/
├── .github/workflows/
├── Dockerfile
└── requirements.txt
```

## Quickstart

```bash
git clone https://github.com/sibi-ai/realtime-anomaly-detection
cd realtime-anomaly-detection
pip install -r requirements.txt
python src/models/train.py --config configs/model.yaml
python src/models/export_onnx.py --quantise int8
uvicorn src.serving.api:app --workers 4
```
