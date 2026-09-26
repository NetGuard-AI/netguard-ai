"""M3/M4 wiring and real-sequence smoke test."""
import argparse,json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
REPO=BASE.parent
M2=REPO/"m2_world_model"/"output"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input"); p.add_argument("--index",type=int,default=0); p.add_argument("--k",type=int,default=5); a=p.parse_args()
    cfg=json.loads((M2/"model_config.json").read_text())
    if a.input:
        seq=np.asarray(np.load(a.input)[a.index],dtype=np.float32); source="REAL M1 sequence"
    else:
        seq=np.zeros((int(cfg["sequence_length"]),int(cfg["input_dim"])),dtype=np.float32); source="ZERO wiring test"
    from output.forecast_engine import rollout
    result=rollout(seq,a.k)
    schema=json.loads((BASE/"output"/"forecast_output_schema.json").read_text())
    import jsonschema
    jsonschema.validate(result,schema)
    print(f"[data] {source}: {seq.shape}")
    print(json.dumps(result,indent=2))
    print("[OK] output matches forecast_output_schema.json")
if __name__=="__main__": main()
