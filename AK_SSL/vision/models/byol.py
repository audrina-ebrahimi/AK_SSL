import copy

import torch
import torch.nn as nn

from AK_SSL.vision.models.modules.heads import (BYOLPredictionHead,
                                                BYOLProjectionHead)


class BYOL(nn.Module):
    """
    BYOL: Bootstrap your own latent: A new approach to self-supervised Learning
    Link: https://arxiv.org/abs/2006.07733
    Implementation: https://github.com/deepmind/deepmind-research/tree/master/byol
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_size: int,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        moving_average_decay: float = 0.99,
        **kwargs
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_size = feature_size
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.moving_average_decay = moving_average_decay
        self.projection_head = BYOLProjectionHead(
            feature_size, hidden_dim, projection_dim
        )
        self.prediction_head = BYOLPredictionHead(
            projection_dim, hidden_dim, projection_dim
        )

        self.online_encoder = self.encoder = nn.Sequential(
            self.backbone, self.projection_head
        )

        self.target_encoder = copy.deepcopy(
            self.online_encoder
        )  # target must be a deepcopy of online, since we will use the backbone trained by online

        self._init_target_encoder()

    def _init_target_encoder(self):
        for param_o, param_t in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        for param_o, param_t in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_t.data = self.moving_average_decay * param_t.data + (1.0 - self.moving_average_decay) * param_o.data

    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        z0_o, z1_o = self.online_encoder(x0), self.online_encoder(x1)
        p0_o, p1_o = self.prediction_head(z0_o), self.prediction_head(z1_o)
        with torch.no_grad():
            self._momentum_update_target_encoder()
            z0_t, z1_t = self.target_encoder(x0), self.target_encoder(x1)

        return (p0_o, z0_t), (p1_o, z1_t)
