"""
Module 1 — Data & Network Telemetry
NetGuard AI
Owner: Kavya

Final CSE-CIC-IDS2018 preprocessing pipeline.
States are created chronologically first; the 70/15/15 split is then applied
at state level, and 20-state windows are created independently per split.
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

BIN_SIZE = 50
SEQUENCE_LENGTH = 20
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
CHUNK_SIZE = 300_000

LABEL_COL = "Label"
FLOW_SEQ_COL = "flow_seq"
CAPTURE_DAY_COL = "capture_day"

FULL_DAY_SET = [
    "02-14-2018", "02-15-2018", "02-16-2018", "02-20-2018",
    "02-21-2018", "02-22-2018", "02-23-2018", "02-28-2018",
    "03-01-2018", "03-02-2018",
]


def discover_files(raw_dir):
    files = [
        f for f in sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
        if "cleaning_report" not in os.path.basename(f).lower()
    ]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    info = []
    for path in files:
        df = pd.read_csv(path, usecols=[FLOW_SEQ_COL, CAPTURE_DAY_COL])
        if df.empty:
            raise ValueError(f"Empty CSV: {path}")
        info.append({
            "path": path,
            "day": str(df[CAPTURE_DAY_COL].iloc[0]),
            "min_seq": int(df[FLOW_SEQ_COL].min()),
            "max_seq": int(df[FLOW_SEQ_COL].max()),
            "n_rows": len(df),
        })

    info.sort(key=lambda x: x["min_seq"])

    for prev, cur in zip(info, info[1:]):
        if cur["min_seq"] != prev["max_seq"] + 1:
            raise ValueError(
                f"flow_seq gap/overlap: {prev['max_seq']} -> {cur['min_seq']}"
            )

    present = sorted({x["day"] for x in info}, key=FULL_DAY_SET.index)
    missing = [d for d in FULL_DAY_SET if d not in present]
    if missing:
        raise ValueError(f"Required capture day(s) missing: {missing}")

    return info


def get_feature_cols(sample_file):
    header = pd.read_csv(sample_file, nrows=0).columns.str.strip().tolist()
    cols = [c for c in header if c not in (FLOW_SEQ_COL, CAPTURE_DAY_COL, LABEL_COL)]
    if len(cols) != 78:
        raise ValueError(f"Expected 78 numeric flow features; found {len(cols)}")
    return cols


def fit_shared_pipeline(files, feature_cols):
    scaler = StandardScaler()
    labels_seen = set()
    dtype_map = {c: "float32" for c in feature_cols}

    for path in files:
        for chunk in pd.read_csv(
            path,
            usecols=feature_cols + [LABEL_COL],
            dtype=dtype_map,
            chunksize=CHUNK_SIZE,
        ):
            x = chunk[feature_cols].to_numpy(dtype=np.float32)
            if not np.isfinite(x).all():
                raise ValueError(f"Non-finite values found in {path}")
            labels = (
                chunk[LABEL_COL]
                .astype(str)
                .str.strip()
                .str.replace("\ufffd", "-", regex=False)
            )
            labels_seen.update(labels.unique().tolist())
            scaler.partial_fit(x)

    encoder = LabelEncoder()
    encoder.fit(sorted(labels_seen))
    return encoder, scaler


def build_states(files, feature_cols, scaler):
    feat_dim = len(feature_cols)
    dtype_map = {c: "float32" for c in feature_cols}

    leftover_x = np.empty((0, feat_dim), dtype=np.float32)
    leftover_attack = np.empty((0,), dtype=np.float32)
    state_parts = []

    for info in files:
        for chunk in pd.read_csv(
            info["path"],
            usecols=feature_cols + [LABEL_COL],
            dtype=dtype_map,
            chunksize=CHUNK_SIZE,
        ):
            labels = (
                chunk[LABEL_COL]
                .astype(str)
                .str.strip()
                .str.replace("\ufffd", "-", regex=False)
                .to_numpy()
            )
            x = chunk[feature_cols].to_numpy(dtype=np.float32)

            if not np.isfinite(x).all():
                raise ValueError(f"Non-finite values found in {info['path']}")

            x = scaler.transform(x).astype(np.float32)
            attack = (np.char.upper(labels.astype(str)) != "BENIGN").astype(np.float32)

            combined_x = np.concatenate([leftover_x, x])
            combined_attack = np.concatenate([leftover_attack, attack])
            n_bins = len(combined_x) // BIN_SIZE

            if n_bins:
                trimmed = combined_x[: n_bins * BIN_SIZE].reshape(
                    n_bins, BIN_SIZE, feat_dim
                )
                means = trimmed.mean(axis=1)
                rates = combined_attack[: n_bins * BIN_SIZE].reshape(
                    n_bins, BIN_SIZE
                ).mean(axis=1)
                state_parts.append(
                    np.hstack([means, rates[:, None]]).astype(np.float32)
                )

            leftover_x = combined_x[n_bins * BIN_SIZE:]
            leftover_attack = combined_attack[n_bins * BIN_SIZE:]

    if not state_parts:
        return np.empty((0, feat_dim + 1), dtype=np.float32)

    return np.concatenate(state_parts, axis=0)


def build_sequences(states):
    n_states, state_dim = states.shape
    n_sequences = max(0, n_states - SEQUENCE_LENGTH)

    if n_sequences == 0:
        return (
            np.empty((0, SEQUENCE_LENGTH, state_dim), dtype=np.float32),
            np.empty((0, state_dim), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    windows = np.lib.stride_tricks.sliding_window_view(
        states, SEQUENCE_LENGTH, axis=0
    )
    sequences = windows.transpose(0, 2, 1)[:n_sequences].copy()
    next_states = states[
        SEQUENCE_LENGTH : SEQUENCE_LENGTH + n_sequences
    ].copy()
    attack_labels = (next_states[:, -1] > 0).astype(np.int64)

    return sequences, next_states, attack_labels


def write_schema(out_dir, feature_cols, encoder, state_sizes, sequence_sizes):
    schema = {
        "dataset": "CSE-CIC-IDS2018",
        "days_present": FULL_DAY_SET,
        "days_missing": [],
        "status": "complete",
        "num_segments": 1,
        "bin_size": BIN_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "state_feature_dim": 79,
        "feature_columns": feature_cols,
        "label_column": LABEL_COL,
        "label_classes": {
            str(c): int(i) for i, c in enumerate(encoder.classes_)
        },
        "benign_detection": "case-insensitive match on 'BENIGN'",
        "split_fracs": {
            "train": TRAIN_FRAC,
            "val": VAL_FRAC,
            "test": 0.15,
        },
        "split_sizes_states": state_sizes,
        "split_sizes_sequences": sequence_sizes,
        "split_method": (
            "chronological state-level split before sequence construction; "
            "no sequence crosses a split boundary"
        ),
        "sequence_boundary_policy": (
            "Build 20-state windows independently inside train, validation, and test."
        ),
        "artifacts": {
            "scaler": "scaler.pkl",
            "encoders": "encoders.pkl",
            "sequence_files": {
                "train": [
                    "state_sequences_train.npy",
                    "next_state_train.npy",
                    "attack_label_train.npy",
                ],
                "val": [
                    "state_sequences_val.npy",
                    "next_state_val.npy",
                    "attack_label_val.npy",
                ],
                "test": [
                    "state_sequences_test.npy",
                    "next_state_test.npy",
                    "attack_label_test.npy",
                ],
            },
            "replay_stream_file": "replay_stream.csv",
        },
    }

    with open(
        os.path.join(out_dir, "feature_schema.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(schema, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", default="./output")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    file_info = discover_files(args.raw_dir)
    feature_cols = get_feature_cols(file_info[0]["path"])

    encoder, scaler = fit_shared_pipeline(
        [x["path"] for x in file_info],
        feature_cols,
    )

    with open(os.path.join(args.out_dir, "encoders.pkl"), "wb") as f:
        pickle.dump(
            {"label_encoder": encoder, "feature_columns": feature_cols},
            f,
        )

    with open(os.path.join(args.out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    states = build_states(file_info, feature_cols, scaler)

    if states.shape[1] != 79:
        raise ValueError(f"Expected state dimension 79; found {states.shape[1]}")

    n_states = len(states)
    train_end = int(n_states * TRAIN_FRAC)
    val_end = train_end + int(n_states * VAL_FRAC)

    splits = {
        "train": states[:train_end],
        "val": states[train_end:val_end],
        "test": states[val_end:],
    }

    state_sizes = {
        name: int(split.shape[0])
        for name, split in splits.items()
    }

    sequence_sizes = {
        name: max(0, split.shape[0] - SEQUENCE_LENGTH)
        for name, split in splits.items()
    }

    for name, split in splits.items():
        sequences, next_states, labels = build_sequences(split)

        np.save(
            os.path.join(args.out_dir, f"state_sequences_{name}.npy"),
            sequences,
        )
        np.save(
            os.path.join(args.out_dir, f"next_state_{name}.npy"),
            next_states,
        )
        np.save(
            os.path.join(args.out_dir, f"attack_label_{name}.npy"),
            labels,
        )

    write_schema(
        args.out_dir,
        feature_cols,
        encoder,
        state_sizes,
        sequence_sizes,
    )

    print("M1 complete: leakage-safe chronological state split created.")


if __name__ == "__main__":
    main()
