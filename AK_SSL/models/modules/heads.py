import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class ProjectionHead(nn.Module):
    """
    Description:
        Base class for all projection and prediction heads.

    Args:
        blocks:
            List of tuples, each denoting one block of the projection head MLP.
            Each tuple reads (in_features, out_features, batch_norm_layer,
            non_linearity_layer).

    """

    def __init__(
        self, blocks: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]]
    ):
        super().__init__()

        layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            use_bias = not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class SimCLRProjectionHead(ProjectionHead):
    """
    Description:
        Initialize a new SimCLRProjectionHead instance.

    Args:
        input_dim: Number of input dimensions.
        hidden_dim: Number of hidden dimensions.
        output_dim: Number of output dimensions.
        num_layers: Number of hidden layers (2 for v1, 3+ for v2).
        batch_norm: Whether or not to use batch norms.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_layers: int = 2,
        batch_norm: bool = True,
        **kwargs,
    ):
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
        super().__init__(layers)


class BarlowTwinsProjectionHead(ProjectionHead):
    """
    Description:
        Projection head used for Barlow Twins.

    Args:
        input_dim: Number of input dimensions.
        hidden_dim: Number of hidden dimensions.
        output_dim: Number of output dimensions.
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 8192, output_dim: int = 8192
    ):
        super(BarlowTwinsProjectionHead, self).init(
            [
                (
                    input_dim,
                    hidden_dim,
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                ),
                (
                    hidden_dim,
                    hidden_dim,
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                ),
                (hidden_dim, output_dim, None, None),
            ]
        )
