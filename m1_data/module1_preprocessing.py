"""
NetGuard AI — M1 Data & Network Telemetry
Final preprocessing for the cleaned CSE-CIC-IDS2018 Parquet source.

The source contains 78 numeric flow features + Label and does not contain
flow_seq/capture_day. Capture dates are inferred from the canonical filenames.
Rows are processed in canonical day order while preserving stored row order.
"""
import argparse, json, os, pickle, re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder, StandardScaler

BIN_SIZE = 50
SEQUENCE_LENGTH = 20
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
CHUNK_SIZE = 300_000
LABEL_COL = "Label"
FULL_DAY_SET = [
    "02-14-2018", "02-15-2018", "02-16-2018", "02-20-2018",
    "02-21-2018", "02-22-2018", "02-23-2018", "02-28-2018",
    "03-01-2018", "03-02-2018",
]

def day_from_filename(path):
    name = os.path.basename(path)
    match = re.search(r"(\d{2})-(\d{2})-(\d{4})", name)
    if not match:
        raise ValueError(f"Could not infer DD-MM-YYYY date from filename: {name}")
    dd, mm, yyyy = match.groups()
    return f"{mm}-{dd}-{yyyy}"

def discover_files(raw_dir):
    paths = sorted(
        os.path.join(raw_dir, name)
        for name in os.listdir(raw_dir)
        if name.lower().endswith(".parquet")
    )
    if not paths:
        raise FileNotFoundError(f"No Parquet files found in {raw_dir}")
    info = []
    for path in paths:
        day = day_from_filename(path)
        parquet_file = pq.ParquetFile(path)
        columns = parquet_file.schema_arrow.names
        if LABEL_COL not in columns:
            raise ValueError(f"{LABEL_COL!r} column missing: {path}")
        n_rows = parquet_file.metadata.num_rows
        if n_rows == 0:
            raise ValueError(f"Empty Parquet file: {path}")
        info.append({"path": path, "day": day, "n_rows": int(n_rows)})
    present = {item["day"] for item in info}
    missing = [day for day in FULL_DAY_SET if day not in present]
    if missing:
        raise ValueError(f"Required capture day(s) missing: {missing}")
    if len(info) != len(FULL_DAY_SET):
        raise ValueError(f"Expected exactly 10 canonical day files; found {len(info)} files.")
    info.sort(key=lambda item: FULL_DAY_SET.index(item["day"]))
    return info

def get_feature_cols(sample_file):
    columns = pq.ParquetFile(sample_file).schema_arrow.names
    columns = [column for column in columns if column != LABEL_COL]
    if len(columns) != 78:
        raise ValueError(f"Expected 78 numeric flow features + Label; found {len(columns)} non-label columns")
    return columns

def iter_batches(path, columns):
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE, columns=columns):
        yield batch.to_pandas()

def normalize_labels(series):
    return series.astype(str).str.strip().str.replace("\ufffd", "-", regex=False)

def fit_shared_pipeline(files, feature_cols):
    scaler = StandardScaler()
    labels_seen = set()
    columns = feature_cols + [LABEL_COL]
    for info in files:
        for chunk in iter_batches(info["path"], columns):
            x = chunk[feature_cols].to_numpy(dtype=np.float32, copy=False)
            if not np.isfinite(x).all():
                raise ValueError(f"Non-finite values found in {info['path']}")
            labels = normalize_labels(chunk[LABEL_COL])
            labels_seen.update(labels.unique().tolist())
            scaler.partial_fit(x)
    encoder = LabelEncoder()
    encoder.fit(sorted(labels_seen))
    return encoder, scaler

def build_states(files, feature_cols, scaler):
    feature_dim = len(feature_cols)
    columns = feature_cols + [LABEL_COL]
    leftover_x = np.empty((0, feature_dim), dtype=np.float32)
    leftover_attack = np.empty((0,), dtype=np.float32)
    state_parts = []
    for info in files:
        for chunk in iter_batches(info["path"], columns):
            labels = normalize_labels(chunk[LABEL_COL]).to_numpy()
            x = chunk[feature_cols].to_numpy(dtype=np.float32, copy=False)
            if not np.isfinite(x).all():
                raise ValueError(f"Non-finite values found in {info['path']}")
            x = scaler.transform(x).astype(np.float32)
            attack = (np.char.upper(labels.astype(str)) != "BENIGN").astype(np.float32)
            combined_x = np.concatenate([leftover_x, x], axis=0)
            combined_attack = np.concatenate([leftover_attack, attack], axis=0)
            n_bins = len(combined_x) // BIN_SIZE
            if n_bins:
                trimmed = combined_x[: n_bins * BIN_SIZE].reshape(n_bins, BIN_SIZE, feature_dim)
                means = trimmed.mean(axis=1)
                rates = combined_attack[: n_bins * BIN_SIZE].reshape(n_bins, BIN_SIZE).mean(axis=1)
                state_parts.append(np.hstack([means, rates[:, None]]).astype(np.float32))
            leftover_x = combined_x[n_bins * BIN_SIZE:]
            leftover_attack = combined_attack[n_bins * BIN_SIZE:]
    if not state_parts:
        return np.empty((0, feature_dim + 1), dtype=np.float32)
    return np.concatenate(state_parts, axis=0)

def write_replay_stream(out_dir, files, feature_cols):
    """Write the canonical replay stream: 78 features + Label + capture_day + flow_seq."""
    output_path = os.path.join(out_dir, "replay_stream.csv")
    columns = feature_cols + [LABEL_COL]
    flow_seq = 0
    wrote_header = False

    for info in files:
        for chunk in iter_batches(info["path"], columns):
            chunk = chunk.copy()
            n_rows = len(chunk)
            chunk["capture_day"] = info["day"]
            chunk["flow_seq"] = np.arange(
                flow_seq,
                flow_seq + n_rows,
                dtype=np.int64,
            )
            flow_seq += n_rows

            ordered = feature_cols + [LABEL_COL, "capture_day", "flow_seq"]
            chunk.to_csv(
                output_path,
                mode="a",
                header=not wrote_header,
                index=False,
                columns=ordered,
            )
            wrote_header = True

    if not wrote_header:
        raise ValueError("Replay stream could not be created: no rows found.")

    print(f"Replay stream written: {output_path} ({flow_seq:,} rows)")


def build_sequences(states):
    n_states, state_dim = states.shape
    n_sequences = max(0, n_states - SEQUENCE_LENGTH)
    if n_sequences == 0:
        return (
            np.empty((0, SEQUENCE_LENGTH, state_dim), dtype=np.float32),
            np.empty((0, state_dim), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    windows = np.lib.stride_tricks.sliding_window_view(states, SEQUENCE_LENGTH, axis=0)
    sequences = windows.transpose(0, 2, 1)[:n_sequences].copy()
    next_states = states[SEQUENCE_LENGTH : SEQUENCE_LENGTH + n_sequences].copy()
    attack_labels = (next_states[:, -1] > 0).astype(np.int64)
    return sequences, next_states, attack_labels

def write_schema(out_dir, feature_cols, encoder, state_sizes, sequence_sizes, files):
    schema = {
        "dataset": "CSE-CIC-IDS2018",
        "input_format": "cleaned Parquet",
        "days_present": FULL_DAY_SET,
        "days_missing": [],
        "status": "complete",
        "num_segments": 1,
        "bin_size": BIN_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "state_feature_dim": 79,
        "feature_columns": feature_cols,
        "label_column": LABEL_COL,
        "label_classes": {str(value): int(index) for index, value in enumerate(encoder.classes_)},
        "benign_detection": "case-insensitive match on 'BENIGN'",
        "flow_order": "Rows are processed in canonical day order; within each Parquet file, stored row order is preserved. The cleaned Parquet source has no flow_seq column.",
        "split_fracs": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": 0.15},
        "split_sizes_states": state_sizes,
        "split_sizes_sequences": sequence_sizes,
        "split_method": "chronological state-level split before sequence construction; no sequence crosses a split boundary",
        "sequence_boundary_policy": "Build 20-state windows independently inside train, validation, and test.",
        "source_files": [
            {"filename": os.path.basename(item["path"]), "capture_day": item["day"], "rows": item["n_rows"]}
            for item in files
        ],
        "replay_stream": {"filename": "replay_stream.csv", "columns": 81},
        "artifacts": {
            "scaler": "scaler.pkl",
            "encoders": "encoders.pkl",
            "sequence_files": {
                "train": ["state_sequences_train.npy", "next_state_train.npy", "attack_label_train.npy"],
                "val": ["state_sequences_val.npy", "next_state_val.npy", "attack_label_val.npy"],
                "test": ["state_sequences_test.npy", "next_state_test.npy", "attack_label_test.npy"],
            },
        },
    }
    with open(os.path.join(out_dir, "feature_schema.json"), "w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", default="./runtime_artifacts")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    files = discover_files(args.raw_dir)
    feature_cols = get_feature_cols(files[0]["path"])
    print("Found canonical files:")
    for item in files:
        print(f"  {item['day']} | {item['n_rows']:,} rows | {os.path.basename(item['path'])}")
    print("Fitting shared StandardScaler...")
    encoder, scaler = fit_shared_pipeline(files, feature_cols)
    with open(os.path.join(args.out_dir, "encoders.pkl"), "wb") as handle:
        pickle.dump({"label_encoder": encoder, "feature_columns": feature_cols}, handle)
    with open(os.path.join(args.out_dir, "scaler.pkl"), "wb") as handle:
        pickle.dump(scaler, handle)
    print(f"Scaler fitted on {int(scaler.n_samples_seen_):,} rows.")
    print("Building replay stream...")
    replay_path = os.path.join(args.out_dir, "replay_stream.csv")
    if os.path.exists(replay_path):
        os.remove(replay_path)
    write_replay_stream(args.out_dir, files, feature_cols)
    print("Building chronological network states...")
    states = build_states(files, feature_cols, scaler)
    if states.shape[1] != 79:
        raise ValueError(f"Expected state dimension 79; found {states.shape[1]}")
    n_states = len(states)
    train_end = int(n_states * TRAIN_FRAC)
    val_end = train_end + int(n_states * VAL_FRAC)
    splits = {"train": states[:train_end], "val": states[train_end:val_end], "test": states[val_end:]}
    state_sizes = {name: int(split.shape[0]) for name, split in splits.items()}
    sequence_sizes = {name: max(0, split.shape[0] - SEQUENCE_LENGTH) for name, split in splits.items()}
    print("State split:", state_sizes)
    for name, split in splits.items():
        sequences, next_states, labels = build_sequences(split)
        np.save(os.path.join(args.out_dir, f"state_sequences_{name}.npy"), sequences)
        np.save(os.path.join(args.out_dir, f"next_state_{name}.npy"), next_states)
        np.save(os.path.join(args.out_dir, f"attack_label_{name}.npy"), labels)
    write_schema(args.out_dir, feature_cols, encoder, state_sizes, sequence_sizes, files)
    print("M1 Parquet preprocessing complete.")
    print(f"Artifacts written to: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()
