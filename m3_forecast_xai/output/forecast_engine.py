"""M3: recursive K-step forecasting using the frozen M2 GRU."""
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Optional
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
M1_SCHEMA = REPO_ROOT / "m1_data" / "output" / "feature_schema.json"
M2_DIR = REPO_ROOT / "m2_world_model" / "output"
if str(M2_DIR) not in sys.path:
    sys.path.insert(0, str(M2_DIR))
from world_model import WorldModel

CONFIG_PATH = M2_DIR / "model_config.json"
WEIGHTS_PATH = M2_DIR / "world_model.pt"
with CONFIG_PATH.open(encoding="utf-8") as f:
    CONFIG = json.load(f)

INPUT_DIM = int(CONFIG["input_dim"])
SEQUENCE_LENGTH = int(CONFIG["sequence_length"])
ATTACK_RATE_INDEX = INPUT_DIM - 1
_model: Optional[WorldModel] = None

def _load_model() -> WorldModel:
    global _model
    if _model is None:
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError("M2 world_model.pt is required at m2_world_model/output/world_model.pt")
        model = WorldModel(**CONFIG)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
        model.eval()
        _model = model
    return _model

def _validate_schema() -> None:
    with M1_SCHEMA.open(encoding="utf-8") as f:
        schema = json.load(f)
    if int(schema["state_feature_dim"]) != INPUT_DIM or int(schema["sequence_length"]) != SEQUENCE_LENGTH:
        raise ValueError("M1 feature_schema.json and M2 model_config.json disagree")

def _rollout_states(sequence: np.ndarray, k: int) -> tuple[list[np.ndarray], list[float]]:
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    _validate_schema()
    window = np.asarray(sequence, dtype=np.float32)
    expected = (SEQUENCE_LENGTH, INPUT_DIM)
    if window.shape != expected:
        raise ValueError(f"state_sequence must have shape {expected}; got {window.shape}")
    if not np.isfinite(window).all():
        raise ValueError("state_sequence contains NaN or infinite values")
    model = _load_model()
    states, probabilities = [], []
    with torch.no_grad():
        for _ in range(int(k)):
            pred = model(torch.from_numpy(window).unsqueeze(0)).squeeze(0).numpy().astype(np.float32)
            probabilities.append(float(np.clip(pred[ATTACK_RATE_INDEX], 0.0, 1.0)))
            states.append(pred)
            window = np.vstack([window[1:], pred]).astype(np.float32)
    return states, probabilities

def _trend(values: list[float]) -> str:
    if len(values) < 2:
        return "stable"
    delta = values[-1] - values[0]
    return "rising" if delta > 0.05 else "falling" if delta < -0.05 else "stable"

def rollout(state_sequence: np.ndarray, k: int) -> dict:
    """Run recursive K-step rollout and return the frozen six-field contract."""
    states, probabilities = _rollout_states(state_sequence, k)
    from .stage_mapper import _stage_confidence, map_stage
    from .explain import explain
    final_state = states[-1]
    xai = explain(final_state)
    return {
        "attack_probability_timeline": probabilities,
        "trend": _trend(probabilities),
        "predicted_stage": map_stage(final_state),
        "stage_confidence": _stage_confidence(final_state),
        "top_features": xai["top_features"],
        "explanation_text": xai["explanation_text"],
    }
