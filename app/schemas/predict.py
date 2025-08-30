from pydantic import BaseModel
from typing import Any, Dict

# We'll replace this with a concrete schema after EDA.
# For now, accept an arbitrary mapping of feature names to values.
class PredictRequest(BaseModel):
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    label: str
    probability: float | None = None