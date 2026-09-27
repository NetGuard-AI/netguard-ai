from __future__ import annotations
import os
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATES = REPO_ROOT / "m1_data" / "output" / "state_sequences_test.npy"
BIN_SIZE = 50
SEQUENCE_LENGTH = 20
STATE_DIM = 79

class PreparedStateProvider:
    """Consumes prepared M1 states; never performs M1 scaling or encoding."""
    def __init__(self, path=None):
        self.path = Path(path or os.getenv("NETGUARD_STATE_SEQUENCES", DEFAULT_STATES))
        self._sequences = None
        self.flow_count = 0
        self.state_count = 0

    def _load(self):
        if self._sequences is None:
            if not self.path.exists():
                raise FileNotFoundError(
                    "Prepared M1 state_sequences_test.npy is required. "
                    "Set NETGUARD_STATE_SEQUENCES to its path. "
                    "M5/M7 do not use scaler.pkl or a placeholder."
                )
            data = np.load(self.path, mmap_mode="r")
            if data.ndim != 3 or data.shape[1:] != (SEQUENCE_LENGTH, STATE_DIM):
                raise ValueError(f"Expected (N, {SEQUENCE_LENGTH}, {STATE_DIM}), got {data.shape}")
            self._sequences = data
        return self._sequences

    def add_replay_flow(self):
        self.flow_count += 1
        completed = self.flow_count % BIN_SIZE == 0
        if completed:
            self.state_count += 1
        return {
            "flow_count": self.flow_count,
            "state_count": self.state_count,
            "state_completed": completed,
            "flows_until_state": BIN_SIZE - (self.flow_count % BIN_SIZE) if self.flow_count % BIN_SIZE else 0,
        }

    def ready(self):
        return self.state_count >= SEQUENCE_LENGTH

    def current_sequence(self):
        data = self._load()
        if not self.ready():
            raise RuntimeError("Twenty prepared network states are required before forecasting.")
        index = self.state_count - SEQUENCE_LENGTH
        if index >= len(data):
            raise RuntimeError("Replay exceeded the prepared M1 state sequence artifact.")
        return np.asarray(data[index], dtype=np.float32)
