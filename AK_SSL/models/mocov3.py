import copy
import torch
import torch.nn as nn

from .modules.heads import SimCLRProjectionHead, BYOLPredictionHead


class MoCov3(nn.Module):
    """
    MoCo v3: Momentum Contrast v3
    Link: https://arxiv.org/abs/2104.02057
    Implementation: https://github.com/facebookresearch/moco-v3
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_size: int,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        moving_average_decay: float = 1.0,
        **kwargs
    ):
        """
        Args:
            backbone: Backbone network.
            feature_size: Number of features.
            projection_dim: Dimension of projection head output.
            hidden_dim: Dimension of hidden layer in projection head.
            moving_average_decay: Decay factor for the moving average of the target encoder.
        """

        super().__init__()
        self.backbone = backbone
        self.feature_size = feature_size
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.moving_average_decay = moving_average_decay

        self.projection_head = SimCLRProjectionHead(
            input_dim=self.feature_size,
            hidden_dim=self.hidden_dim,
            output_dim=self.projection_dim,
        )

        self.encoder_q = self.encoder = nn.Sequential(
            self.backbone, self.projection_head
        )

        self.prediction_head = BYOLPredictionHead(
            input_dim=self.projection_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.projection_dim,
        )

        self.encoder_k = copy.deepcopy(self.encoder_q)

        self._init_encoder_k()

    @torch.no_grad()
    def _init_encoder_k(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _update_momentum_encoder(self):
        for param_b, param_m in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_m.data = param_m.data * self.m + param_b.data * (1.0 - self.m)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        q0 = self.predictor(self.encoder_q(x0))
        q1 = self.predictor(self.encoder_q(x1))
        with torch.no_grad():
            self._update_momentum_encoder()
            k0 = self.encoder_k(x0)
            k1 = self.encoder_k(x1)

        return (q0, q1), (k0, k1)
