"""SHAP explanation of the M2 predicted attack_rate output."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import shap
import torch

# Keep CPU memory predictable on small deployment instances such as Render Free.
torch.set_num_threads(1)

REPO_ROOT=Path(__file__).resolve().parents[2]
M1_SCHEMA=REPO_ROOT/"m1_data"/"output"/"feature_schema.json"
M2_DIR=REPO_ROOT/"m2_world_model"/"output"
if str(M2_DIR) not in sys.path:
    sys.path.insert(0,str(M2_DIR))
from world_model import WorldModel

CONFIG_PATH=M2_DIR/"model_config.json"
WEIGHTS_PATH=M2_DIR/"world_model.pt"
with CONFIG_PATH.open(encoding="utf-8") as f:
    CONFIG=json.load(f)
INPUT_DIM=int(CONFIG["input_dim"])
SEQUENCE_LENGTH=int(CONFIG["sequence_length"])
ATTACK_RATE_INDEX=INPUT_DIM-1
_model=None
_names_cache=None
_explanation_cache_key=None
_explanation_cache_value=None

def _load_model():
    global _model
    if _model is None:
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError("M2 world_model.pt is required at m2_world_model/output/world_model.pt")
        model=WorldModel(**CONFIG)
        model.load_state_dict(torch.load(WEIGHTS_PATH,map_location="cpu",weights_only=True))
        model.eval()
        _model=model
    return _model

def _names():
    global _names_cache
    if _names_cache is None:
        with M1_SCHEMA.open(encoding="utf-8") as f:
            schema=json.load(f)
        _names_cache=list(schema["feature_columns"])+["attack_rate"]
        if len(_names_cache)!=INPUT_DIM:
            raise ValueError("M1 feature schema and M2 model dimensions disagree")
    return _names_cache

class _AttackRateWrapper(torch.nn.Module):
    def __init__(self,model):
        super().__init__(); self.model=model
    def forward(self,x):
        return self.model(x)[:,ATTACK_RATE_INDEX:ATTACK_RATE_INDEX+1]

def explain(state_vector: np.ndarray)->dict:
    global _explanation_cache_key, _explanation_cache_value
    state=np.asarray(state_vector,dtype=np.float32)
    if state.shape!=(INPUT_DIM,):
        raise ValueError(f"state_vector must have shape ({INPUT_DIM},); got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("state_vector contains NaN or infinite values")

    # Rollout polling can request the exact same deterministic forecast repeatedly.
    # Reusing the SHAP result avoids repeated gradient allocations and memory spikes.
    cache_key = state.tobytes()
    if cache_key == _explanation_cache_key and _explanation_cache_value is not None:
        return _explanation_cache_value

    window=np.repeat(state[None,:],SEQUENCE_LENGTH,axis=0)
    sample=torch.from_numpy(window).unsqueeze(0)
    background=torch.zeros((1,SEQUENCE_LENGTH,INPUT_DIM),dtype=torch.float32)
    values=shap.GradientExplainer(_AttackRateWrapper(_load_model()),background).shap_values(sample)
    values=np.asarray(values[0] if isinstance(values,list) else values)
    if values.ndim==4 and values.shape[-1]==1: values=values[...,0]
    if values.ndim==3 and values.shape[0]==1: values=values[0]
    if values.shape!=(SEQUENCE_LENGTH,INPUT_DIM):
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")
    importance=np.abs(values).sum(axis=0)
    indices=np.argsort(importance)[::-1][:5]
    names=_names()
    top=[{"feature":names[int(i)],"contribution":float(importance[int(i)])} for i in indices]
    if len(top)>=2:
        text=f"Risk is driven mainly by unusual {top[0]['feature']} and {top[1]['feature']} behavior."
    elif top:
        text=f"Risk is driven mainly by unusual {top[0]['feature']} behavior."
    else:
        text="No single feature stood out strongly in this forecast."
    result={"top_features":top,"explanation_text":text}
    _explanation_cache_key = cache_key
    _explanation_cache_value = result
    return result
