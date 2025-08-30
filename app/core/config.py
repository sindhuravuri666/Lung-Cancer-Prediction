from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "Lung Cancer Predictor API"
    LOG_LEVEL: str = "INFO"
    MODEL_PATH: str = "app/models/artifacts/model.joblib"
    ALLOWED_ORIGINS: str | None = "http://localhost:5173"

    @property
    def cors_origins(self) -> List[str]:
        if not self.ALLOWED_ORIGINS:
            return []
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]


settings = Settings()  # reads from environment or .env if present