"""Rule-based MITRE-style stage mapping; not attack-label ground truth."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "m1_data" / "output" / "feature_schema.json"
ALIASES = {
    "syn":["SYN Flag Cnt","SYN Flag Count"], "ack":["ACK Flag Cnt","ACK Flag Count"],
    "rst":["RST Flag Cnt","RST Flag Count"], "duration":["Flow Duration"],
    "bytes_s":["Flow Byts/s","Flow Bytes/s"], "fwd_len":["TotLen Fwd Pkts","Total Length of Fwd Packets"],
    "bwd_len":["TotLen Bwd Pkts","Total Length of Bwd Packets"], "idle":["Idle Mean"], "active":["Active Mean"]
}
_schema: Optional[dict] = None
_indices: Optional[dict[str,int]] = None

def _load_schema() -> dict:
    global _schema, _indices
    if _schema is None:
        with SCHEMA_PATH.open(encoding="utf-8") as f:
            _schema=json.load(f)
        cols=list(_schema["feature_columns"])
        _indices={name:i for i,name in enumerate(cols)}
        _indices["attack_rate"]=len(cols)
    return _schema

def _idx(key:str)->Optional[int]:
    _load_schema()
    assert _indices is not None
    return next((i for name in ALIASES.get(key,[key]) if (i:=_indices.get(name)) is not None),None)

def _v(state,key):
    i=_idx(key)
    return None if i is None else float(state[i])

def map_stage(state_vector: np.ndarray) -> str:
    state=np.asarray(state_vector,dtype=np.float32)
    schema=_load_schema()
    if state.shape!=(int(schema["state_feature_dim"]),):
        raise ValueError("state_vector has the wrong dimension")
    if not np.isfinite(state).all():
        raise ValueError("state_vector contains NaN or infinite values")
    assert _indices is not None
    if float(state[_indices["attack_rate"]]) < 0.10:
        return "Uncertain"
    syn,ack,rst=_v(state,"syn"),_v(state,"ack"),_v(state,"rst")
    duration,bytes_s=_v(state,"duration"),_v(state,"bytes_s")
    fwd,bwd=_v(state,"fwd_len"),_v(state,"bwd_len")
    idle,active=_v(state,"idle"),_v(state,"active")
    if syn is not None and ack is not None and duration is not None and syn>1 and ack<-0.5 and duration<-0.5:
        return "Reconnaissance"
    if rst is not None and duration is not None and rst>1 and duration<-0.5:
        return "Initial Access"
    if idle is not None and active is not None and bytes_s is not None and idle>1 and active<-0.5 and bytes_s<-0.5:
        return "Command & Control"
    if bwd is not None and duration is not None and bwd>1 and duration>1:
        return "Exfiltration"
    if fwd is not None and bwd is not None and duration is not None and abs(fwd-bwd)<0.5 and duration>0:
        return "Lateral Movement"
    return "Uncertain"

def _stage_confidence(state_vector: np.ndarray, stage: Optional[str]=None) -> str:
    stage=stage or map_stage(state_vector)
    if stage=="Uncertain":
        return "uncertain"
    return "confident" if float(np.asarray(state_vector)[-1])>=0.30 else "uncertain"
