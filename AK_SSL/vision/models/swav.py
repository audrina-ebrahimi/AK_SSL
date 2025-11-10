import copy

import torch
import torch.nn as nn

from AK_SSL.vision.models.modules.heads import SwAVProjectionHead


class SwAV(nn.Module):
    """
    SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments
    Link: https://arxiv.org/abs/2006.09882
    Implementation: https://github.com/facebookresearch/swav
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_size: int,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        epsilon: float = 0.05,
        sinkhorn_iterations: int = 3,
        num_prototypes: int = 3000,
        queue_length: int = 64,
        use_the_queue: bool = True,
        num_crops: int = 6,
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_size = feature_size
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.num_prototypes = num_prototypes
        self.queue_length = queue_length
        self.use_the_queue = use_the_queue
        self.num_crops = num_crops

        self.register_buffer(
            "queue", torch.zeros(2, self.queue_length, self.projection_dim)
        )

        self.projection_head = SwAVProjectionHead(
            feature_size, hidden_dim, projection_dim
        )
        self.encoder = nn.Sequential(self.backbone, self.projection_head)
        self.prototypes = nn.Linear(
            self.projection_dim, self.num_prototypes, bias=False
        )
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def sinkhorn(self, Q):
        with torch.no_grad():
            Q = torch.exp(Q / self.epsilon).t()
            B = Q.shape[1]
            K = Q.shape[0]
            sum_Q = torch.sum(Q)
            Q /= sum_Q
            for _ in range(self.sinkhorn_iterations):
                sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
                Q /= sum_of_rows
                Q /= K
                Q /= torch.sum(Q, dim=0, keepdim=True)
                Q /= B
            Q *= B
            return Q.t()

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, xc: list):
        bz = x0.shape[0]
        with torch.no_grad():  # normalize prototypes
            w = self.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)
        z1, z2 = self.encoder(x1), self.encoder(x1)
        z1, z2 = nn.functional.normalize(z1, dim=1, p=2), nn.functional.normalize(
            z2, dim=1, p=2
        )
        z1, z2 = z1.detach(), z2.detach()
        c1, c2 = self.prototypes(z1), self.prototypes(z2)
        _c1, _c2 = c1.detach(), c2.detach()
        with torch.no_grad():
            if self.queue is not None:
                if self.use_the_queue:
                    _c1 = torch.cat(
                        (torch.mm(self.queue[0], self.prototypes.weight.t()), _c1)
                    )
                    _c2 = torch.cat(
                        (torch.mm(self.queue[1], self.prototypes.weight.t()), _c2)
                    )
                    self.queue[0, bz:] = self.queue[0, :-bz].clone()
                    self.queue[0, :bz] = z1
                    self.queue[1, bz:] = self.queue[1, :-bz].clone()
                    self.queue[1, :bz] = z2
            q1, q2 = self.sinkhorn(_c1)[:bz, :], self.sinkhorn(_c2)[:bz, :]
        z_c, c_c = [], []
        for x in xc:
            z = self.encoder(x)
            z = nn.functional.normalize(z, dim=1, p=2)
            z = z.detach()
            z_c.append(z)
            c_c.append(self.prototypes(z))
        return (c1, c2, c_c), (q1, q2)
