import torch
import torch.nn as nn
import torch.nn.functional as F


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_param: float = 5e-3, **kwargs):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        # normalize repr. along the batch dimension
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)
        # normalize the representations along the batch dimension
        out_1_norm = (z_a - z_a.mean(dim=0)) / z_a.std(dim=0)
        out_2_norm = (z_b - z_b.mean(dim=0)) / z_b.std(dim=0)

        # cross-correlation matrix
        batch_size = z_a.size(0)
        c = torch.matmul(out_1_norm.T, out_2_norm) / batch_size

        # loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).add_(1).pow_(2).sum()
        loss = on_diag + self.lambda_param * off_diag

        return loss
