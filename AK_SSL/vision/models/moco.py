import copy

import torch
import torch.nn as nn

from AK_SSL.vision.models.modules.heads import (BYOLPredictionHead,
                                                SimCLRProjectionHead)


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
        **kwargs,
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
            param_m.data = param_m.data * self.moving_average_decay + param_b.data * (
                1.0 - self.moving_average_decay
            )

    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        q0 = self.prediction_head(self.encoder_q(x0))
        q1 = self.prediction_head(self.encoder_q(x1))
        with torch.no_grad():
            self._update_momentum_encoder()
            k0 = self.encoder_k(x0)
            k1 = self.encoder_k(x1)

        return (q0, q1), (k0, k1)


class MoCoV2(nn.Module):
    """
    MoCo v2: Momentum Contrast v2
    Link: https://arxiv.org/abs/2003.04297
    Implementation: https://github.com/facebookresearch/moco
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_size: int,
        projection_dim: int = 128,
        temperature: float = 0.07,
        K: int = 65536,
        m: float = 0.999,
        **kwargs,
    ):
        """
        Args:
            backbone: Backbone network.
            feature_size: Number of features.
            projection_dim: Dimension of projection head output.
            K: Number of negative keys.
            m: Momentum for updating the key encoder.
        """
        super().__init__()
        self.backbone = backbone
        self.projection_dim = projection_dim
        self.feature_size = feature_size
        self.temperature = temperature
        self.K = K
        self.m = m

        self.projection_head = SimCLRProjectionHead(
            input_dim=self.feature_size,
            hidden_dim=self.feature_size,
            output_dim=self.projection_dim,
        )

        self.encoder_q = self.encoder = nn.Sequential(
            self.backbone, self.projection_head
        )
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self._init_encoder_k()

        self.register_buffer("queue", torch.randn(projection_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _init_encoder_k(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_encoder_k(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        idx_shuffle = torch.randperm(x.shape[0]).cuda()
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        return x[idx_unshuffle]

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        bz = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % bz == 0
        self.queue[:, ptr : (ptr + bz)] = keys.t()
        ptr = (ptr + bz) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        q = self.encoder_q(x0)
        q = nn.functional.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_encoder_k()
            x1, idx_unshuffle = self._batch_shuffle_single_gpu(x1)
            k = self.encoder_k(x1)
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

        self._dequeue_and_enqueue(k)

        return logits, labels
