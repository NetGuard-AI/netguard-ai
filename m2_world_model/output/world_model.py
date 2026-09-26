"""Module 2 — Network State + World Model
Architecture definition only, importable by Module 3.
"""

import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """GRU world model that predicts the next 79-dimensional network state."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        sequence_length: int = None,
        model_type: str = "GRU",
        output_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if model_type != "GRU":
            raise ValueError(
                f"WorldModel only implements GRU per the SRS. Got {model_type!r}."
            )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.model_type = model_type
        self.output_dim = output_dim or input_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])
