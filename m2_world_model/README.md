# Module 2 — Network State + World Model (Divyanshi)

## What this module does
Learns temporal network dynamics with one torch.nn.GRU: given 20 consecutive
79-dimensional network states, it predicts the following network state.

## Inputs
From M1 output:
- feature_schema.json
- state_sequences_train.npy
- state_sequences_val.npy
- state_sequences_test.npy
- next_state_train.npy
- next_state_val.npy
- next_state_test.npy

## Outputs
- world_model.py
- world_model.pt
- model_config.json
- eval_metrics.json

M3 consumes world_model.py, world_model.pt, and model_config.json.
eval_metrics.json is an M2 evaluation artifact, not a required M3 input.

## Final configuration
input_dim=79, hidden_dim=32, num_layers=1, sequence_length=20, model_type=GRU

## Final evaluation
- Validation MSE: 0.1274405771
- Validation MAE: 0.1328678351
- Test MSE: 0.1054958588
- Test MAE: 0.1225121365

## How to run
pip install -r requirements.txt
python train_world_model.py --data-dir /path/to/m1_data/output --out-dir ./output

The final checkpoint was execution-tested against the corrected M1 tensors.
