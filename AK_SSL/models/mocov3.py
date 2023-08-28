import torch
import torch.nn as nn

from .modules.heads import SimCLRProjectionHead

class MoCov3(nn.Module):
    """
    MoCo v3: Momentum Contrast v3
    Link: https://arxiv.org/abs/2104.02057
    Implementation: https://github.com/facebookresearch/moco-v3
    MoCo v2: Momentum Contrast v2
    Link: https://arxiv.org/abs/2003.04297
    Implementation: https://github.com/facebookresearch/moco
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
            output_dim=self.projection_dim
        )

        self.encoder_q = self.encoder = nn.Sequential(self.backbone, self.projection_head)

        