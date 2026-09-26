import json,sys
from pathlib import Path
import numpy as np,pytest,torch
BASE=Path(__file__).resolve().parent
REPO=BASE.parent
M2=REPO/"m2_world_model"/"output"
M1=REPO/"m1_data"/"output"
if str(M2) not in sys.path: sys.path.insert(0,str(M2))
from world_model import WorldModel
from output.forecast_engine import rollout

def test_model_contract():
    cfg=json.loads((M2/"model_config.json").read_text())
    schema=json.loads((M1/"feature_schema.json").read_text())
    assert cfg["input_dim"]==schema["state_feature_dim"]==79
    assert cfg["sequence_length"]==schema["sequence_length"]==20
    assert cfg["model_type"]=="GRU"

def test_weights_load():
    weights=M2/"world_model.pt"
    if not weights.exists(): pytest.skip("M2 world_model.pt is a local handoff artifact")
    cfg=json.loads((M2/"model_config.json").read_text())
    model=WorldModel(**cfg)
    model.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True))
    assert tuple(model(torch.zeros((1,20,79))).shape)==(1,79)

def test_rollout_schema():
    weights=M2/"world_model.pt"
    if not weights.exists(): pytest.skip("M2 world_model.pt is a local handoff artifact")
    result=rollout(np.zeros((20,79),dtype=np.float32),3)
    schema=json.loads((BASE/"output"/"forecast_output_schema.json").read_text())
    import jsonschema
    jsonschema.validate(result,schema)
    assert len(result["attack_probability_timeline"])==3
