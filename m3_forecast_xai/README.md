# Module 3/4 — Attack Forecasting + Explainable AI (Shravani)

## Inputs
- `m1_data/output/feature_schema.json`
- `m2_world_model/output/world_model.py`
- `m2_world_model/output/model_config.json`
- `m2_world_model/output/world_model.pt` (runtime model artifact; kept with the M2 handoff)

This module does not fit a new scaler.

## Outputs
- `output/forecast_engine.py`
- `output/stage_mapper.py`
- `output/explain.py`
- `output/forecast_output_schema.json`

`forecast_output_schema.json` is the frozen M3/M4 → Anamika contract.

## Public interfaces
```python
rollout(state_sequence: np.ndarray, k: int) -> dict
map_stage(state_vector: np.ndarray) -> str
explain(state_vector: np.ndarray) -> dict
```

## Handoff
Anamika should build the Pydantic response model directly from `output/forecast_output_schema.json`.

## Validation
```bash
pip install -r requirements.txt
pytest -q test_m3_m4_contract.py
python smoke_test.py --input path/to/state_sequences_test.npy --index 0 --k 5
```

The no-input smoke test is only a wiring check, not real-data forecast validation.
