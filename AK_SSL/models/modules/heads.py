import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class SimCLRProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_layers: int = 2,
        batch_norm: bool = True,
    ):
        """Initialize a new SimCLRProjectionHead instance.

        Args:
            input_dim: Number of input dimensions.
            hidden_dim: Number of hidden dimensions.
            output_dim: Number of output dimensions.
            num_layers: Number of hidden layers (2 for v1, 3+ for v2).
            batch_norm: Whether or not to use batch norms.
        """
        layers: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]] = []
        layers.append(
            (
                input_dim,
                hidden_dim,
                nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                nn.ReLU(inplace=True),
            )
        )
        for _ in range(2, num_layers):
            layers.append(
                (
                    hidden_dim,
                    hidden_dim,
                    nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                    nn.ReLU(inplace=True),
                )
            )
        layers.append(
            (
                hidden_dim,
                output_dim,
                nn.BatchNorm1d(output_dim) if batch_norm else None,
                None,
            )
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)
