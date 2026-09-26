# Module 1 — Data & Network Telemetry (Kavya)

## What this module does
Ingests the pre-cleaned CSE-CIC-IDS2018 CSVs, fits one shared scaler and label encoder on the complete cleaned dataset, aggregates chronological flows into fixed-size network states, and builds leakage-safe temporal sequences for M2.

## Dataset status
All 10 required capture days are present and gap-free by `flow_seq`: 02-14, 02-15, 02-16, 02-20, 02-21, 02-22, 02-23, 02-28, 03-01, 03-02 (2018).

## Frozen M1 contract
- BIN_SIZE = 50 flows per state
- SEQUENCE_LENGTH = 20 states
- STATE_FEATURE_DIM = 79 = 78 flow features + attack_rate
- One shared StandardScaler and LabelEncoder fitted on the full cleaned dataset
- Chronological state split: 70% train / 15% validation / 15% test
- Sequences are built separately inside each split; no sequence crosses a split boundary

## Handoff
M2 reads only state_sequences_*.npy, next_state_*.npy, and feature_schema.json, as specified by the SRS.

## Note on repository storage
The generated NumPy artifacts are large binary files. They remain in the verified M1 handoff ZIP and are intentionally not committed to ordinary GitHub storage because several files exceed GitHub's 100 MB per-file limit. The source pipeline and schema are committed here.