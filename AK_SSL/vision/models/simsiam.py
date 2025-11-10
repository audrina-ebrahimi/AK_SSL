import torch
import torch.nn as nn

from AK_SSL.vision.models.modules.heads import (SimSiamPredictionHead,
                                                SimSiamProjectionHead)


class SimSiam(nn.Module):
    """
    SimSiam: Exploring Simple Siamese Representation Learning
    Link: https://arxiv.org/abs/2011.10566
    Implementation: https://github.com/facebookresearch/simsiam
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_size: int,
        projection_dim: int = 2048,
        projection_hidden_dim: int = 2048,
        prediction_hidden_dim: int = 512,
        **kwargs
    ):
        super().__init__()
        self.feature_size = feature_size
        self.projection_dim = projection_dim
        self.projection_hidden_dim = projection_hidden_dim
        self.prediction_hidden_dim = prediction_hidden_dim
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(
            input_dim=self.feature_size,
            hidden_dim=self.projection_hidden_dim,
            output_dim=self.projection_dim,
        )
        self.prediction_head = SimSiamPredictionHead(
            input_dim=self.projection_dim,
            hidden_dim=self.prediction_hidden_dim,
            output_dim=self.projection_dim,
        )

    def forward(self, x0: torch.Tensor, x1: torch.Tensor = None):
        f0 = self.backbone(x0).flatten(start_dim=1)
        z0 = self.projection_head(f0)
        p0 = self.prediction_head(z0)

        out0 = (z0, p0)

        if x1 is None:
            return out0

        f1 = self.backbone(x1).flatten(start_dim=1)
        z1 = self.projection_head(f1)
        p1 = self.prediction_head(z1)

        out1 = (z1, p1)

        return out0, out1
