from typing import List, Optional, Tuple

import torch
import torch.nn as nn


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
        super(BarlowTwinsProjectionHead, self).__init__(
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


class BYOLProjectionHead(ProjectionHead):
    """
    Projection head used for BYOL.
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 4096, output_dim: int = 256
    ):
        super(BYOLProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class BYOLPredictionHead(ProjectionHead):
    """
    Prediction head used for BYOL.
    """

    def __init__(
        self, input_dim: int = 256, hidden_dim: int = 4096, output_dim: int = 256
    ):
        super(BYOLPredictionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class SimSiamProjectionHead(ProjectionHead):
    """
    Projection head used for SimSiam.
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 2048
    ):
        super(SimSiamProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (
                    hidden_dim,
                    output_dim,
                    nn.BatchNorm1d(output_dim, affine=False),
                    None,
                ),
            ]
        )


class SimSiamPredictionHead(ProjectionHead):
    """
    Prediction head used for SimSiam.
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 2048
    ):
        super(SimSiamPredictionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class SwAVProjectionHead(ProjectionHead):
    """
    Projection head used for SwAV.
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 128
    ):
        super(SwAVProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class DINOProjectionHead(nn.Module):
    """
    Projection Head for DINO
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        use_bn: bool = False,
        norm_last_layer: bool = True,
        num_layers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        num_layers = max(num_layers, 1)
        if num_layers == 1:
            self.mlp = nn.Linear(input_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(input_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, output_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
