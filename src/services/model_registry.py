"""Lazy loader for trained model artifacts.

Owner: P5 (Dashboard & API)
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import joblib

from src.utils.exceptions import ModelNotAvailableError
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR: Path = Path(os.environ.get("RADQUEUE_MODELS_DIR", PROJECT_ROOT / "models"))
WAIT_TIME_MODEL_PREFERENCE: list[str] = ["ensemble_best.pkl", "xgboost_best.pkl", "lightgbm_best.pkl"]
EXPLAINER_MODEL_PREFERENCE: list[str] = ["xgboost_best.pkl", "lightgbm_best.pkl"]
NOSHOW_MODEL_FILE: str = "xgboost_noshow.pkl"
TRAIN_HINT: str = (
    "Train it first: python -m src.data.generator && python -m src.data.preprocessor && "
    "python -m src.models.wait_time.trainer --model all --device cpu && "
    "python -m src.models.wait_time.ensemble && python -m src.models.noshow.trainer --device cpu "
    "(see MANUAL_COMMANDS.md)."
)


def _first_existing(directory: Path, names: list[str]) -> Path | None:
    return next((directory / n for n in names if (directory / n).exists()), None)


class ModelRegistry:
    """Loads wait-time, explainer and no-show models on first use (thread-safe)."""

    def __init__(self, models_dir: Path | str = DEFAULT_MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)
        self._lock = threading.Lock()
        self._cache: dict[str, Any] = {}

    def _load(self, key: str, factory: Any) -> Any:
        with self._lock:
            if key not in self._cache:
                self._cache[key] = factory()
            return self._cache[key]

    @property
    def wait_time_predictor(self) -> Any:
        """Best available wait-time predictor (ensemble > XGBoost > LightGBM)."""
        def factory() -> Any:
            from src.models.wait_time.predictor import WaitTimePredictor

            path = _first_existing(self.models_dir / "wait_time", WAIT_TIME_MODEL_PREFERENCE)
            if path is None:
                raise ModelNotAvailableError(f"No wait-time model found in {self.models_dir / 'wait_time'}. {TRAIN_HINT}")
            return WaitTimePredictor(path)
        return self._load("wait_time", factory)

    @property
    def explainer(self) -> Any:
        """SHAP explainer over the best tree model."""
        def factory() -> Any:
            from src.models.wait_time.explainer import WaitTimeExplainer

            path = _first_existing(self.models_dir / "wait_time", EXPLAINER_MODEL_PREFERENCE)
            if path is None:
                raise ModelNotAvailableError(f"No tree model for SHAP in {self.models_dir / 'wait_time'}. {TRAIN_HINT}")
            features_path = path.parent / "feature_names.json"
            feature_names = json.loads(features_path.read_text(encoding="utf-8")) if features_path.exists() else None
            explainer = WaitTimeExplainer(joblib.load(path), feature_names)
            explainer.model_name = path.stem
            return explainer
        return self._load("explainer", factory)

    @property
    def noshow_predictor(self) -> Any:
        """Calibrated no-show predictor."""
        def factory() -> Any:
            from src.models.noshow.predictor import NoShowPredictor

            path = self.models_dir / "noshow" / NOSHOW_MODEL_FILE
            if not path.exists():
                raise ModelNotAvailableError(f"No-show model not found at {path}. {TRAIN_HINT}")
            return NoShowPredictor(path)
        return self._load("noshow", factory)

    def try_wait_time_predictor(self) -> Any | None:
        """Wait-time predictor, or None if not trained yet (logged)."""
        try:
            return self.wait_time_predictor
        except ModelNotAvailableError as exc:
            logger.warning(str(exc))
            return None

    def model_status(self) -> dict[str, bool]:
        """Which model artifacts exist on disk."""
        wt = self.models_dir / "wait_time"
        return {
            "wait_time": _first_existing(wt, WAIT_TIME_MODEL_PREFERENCE) is not None,
            "explainer": _first_existing(wt, EXPLAINER_MODEL_PREFERENCE) is not None,
            "noshow": (self.models_dir / "noshow" / NOSHOW_MODEL_FILE).exists(),
        }

    def model_metadata(self) -> dict[str, Any]:
        """All committed/trained model metadata JSON files keyed by model name."""
        out: dict[str, Any] = {}
        for path in sorted(self.models_dir.glob("*/*.json")):
            if path.name in {"feature_names.json", "feature_encoders.json"}:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            data.pop("feature_names", None)
            out[path.stem.replace("_metadata", "")] = data
        return out
