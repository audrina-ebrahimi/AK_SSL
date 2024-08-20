import torch
import torch.nn as nn


class NegativeCosineSimilarity(torch.nn.Module):
    def __init__(self, dim: int = 1, eps: float = 1e-8, **kwargs) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.criterion = nn.CosineSimilarity(dim=self.dim, eps=self.eps)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        z0, p0 = x0
        z1, p1 = x1

        loss = -(self.criterion(p0, z1).mean() + self.criterion(p1, z0).mean()) * 0.5
        return loss
