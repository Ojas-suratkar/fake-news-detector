from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    random_state: int = 42
    max_features: int = 20000
    test_size: float = 0.2
    model_filename: str = "fake_news_pipeline.joblib"

    @property
    def root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    @property
    def models_dir(self) -> Path:
        return self.root / "models"