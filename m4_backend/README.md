# M5/M7 - Final backend

M5/M7 no longer performs M1 preprocessing. It does not load scaler.pkl,
encoders.pkl, or a placeholder scaler. It consumes the prepared M1
state_sequences_test.npy artifact and the frozen M3 forecast contract.

Dependency chain:
M1 prepared states -> M2 world model -> M3/M4 forecast -> M5/M7 risk,
alerts, evidence and cyber-help -> M6 frontend.

For the replay demo, POST /ingest accepts a replay row and advances through
the prepared M1 test-state timeline in 50-flow steps. This is intentionally
replay/demo oriented; arbitrary unlabeled live traffic is outside the current
model contract.

Optional environment variable:
NETGUARD_STATE_SEQUENCES=/absolute/path/to/state_sequences_test.npy

Default:
../m1_data/output/state_sequences_test.npy

Exact SRS endpoints:
POST /ingest
GET /predict
GET /rollout?k=<int>
GET /explain
GET /security-zone
GET /alerts
POST /alerts/trigger-attack
GET /evidence-pack/{incident_id}
GET /knowledge-center
GET /help-resources

Additional prototype endpoints:
POST /signup
POST /login
POST /incidents

SQLite tables:
users
devices
alerts
incidents

Run from repository root:
pip install -r m4_backend/requirements.txt
uvicorn m4_backend.main:app --reload --port 8000

Runtime requirements:
- canonical m3_forecast_xai source
- trained M2 world_model.pt
- M1 feature_schema.json
- M1 state_sequences_test.npy

No scaler.pkl is required and no placeholder is used.
