"""Tests for the wait-time and no-show model pipelines.

Training tests use a tiny synthetic dataset; regression tests on the real
trained artifacts run only when they exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.noshow.trainer import NOSHOW_FEATURES, _get_noshow_feature_target
from src.models.wait_time.trainer import compute_evaluation_extras, train_elastic_net
from src.utils.constants import TARGET_R_SQUARED

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
# Regression floor: honest held-out R² must not fall far below the current level.
MIN_TEST_R_SQUARED = TARGET_R_SQUARED - 0.10


def _synthetic(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.uniform(0, 1, size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    df[TARGET_COLUMN] = 30 * df["current_queue_length_same_modality"] + 5 * df["urgency_encoded"] + rng.normal(0, 1, n)
    return df


def test_should_learn_queue_effect_and_report_test_metrics() -> None:
    train, val, test = _synthetic(300, 0), _synthetic(100, 1), _synthetic(100, 2)
    model, _, val_metrics = train_elastic_net(train, val)
    extras = compute_evaluation_extras(model, val, test)
    assert val_metrics["r_squared"] > 0.9
    assert extras["test_metrics"]["r_squared"] > 0.9
    assert extras["residual_quantiles"]["q05"] < 0 < extras["residual_quantiles"]["q95"]


def test_should_train_noshow_only_on_scheduled_patients() -> None:
    df = pd.DataFrame({c: [0, 0, 0] for c in NOSHOW_FEATURES})
    df["visit_type_encoded"] = [0, 1, 2]  # scheduled, walk-in, emergency
    df["showed_up"] = [False, True, True]
    X, y, _ = _get_noshow_feature_target(df)
    assert len(X) == 1 and y.tolist() == [1]
    assert "visit_type_encoded" not in X.columns


@pytest.mark.skipif(not (MODELS_DIR / "wait_time" / "ensemble_best.json").exists(), reason="model not trained")
def test_should_keep_test_r_squared_above_regression_floor() -> None:
    meta = json.loads((MODELS_DIR / "wait_time" / "ensemble_best.json").read_text(encoding="utf-8"))
    assert meta["test_metrics"]["r_squared"] >= MIN_TEST_R_SQUARED
