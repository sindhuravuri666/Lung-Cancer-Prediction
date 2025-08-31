from __future__ import annotations
from typing import Dict, Any
import joblib # for saving/loading models
import os
from app.core.config import settings


_model = None  # cached global


def load_model():
    global _model
    path = settings.MODEL_PATH #
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found at {path}. Train and save the model first.")
    _model = joblib.load(path)
    return _model


def predict(features: Dict[str, Any]) -> Dict[str, Any]:
    global _model
    if _model is None:
        _model = load_model()

    # Expect a pipeline that supports .predict and optionally .predict_proba
    import pandas as pd
    X = pd.DataFrame([features])
    y_pred = _model.predict(X)[0]
    proba = None
    if hasattr(_model, "predict_proba"):
        proba = float(_model.predict_proba(X)[0][1])  # probability for positive class

    return {"label": str(y_pred), "probability": proba}