import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE_MoCoV3(nn.Module):
    def __init__(self, temperature: float = 0.07, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.eps = 1e-8
        if abs(self.temperature) < self.eps:
            raise ValueError(
                "Illegal temperature: abs({}) < 1e-8".format(self.temperature)
            )

    def _compute_loss(self, q, k):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        logits = torch.einsum("nc,mc->nm", [q, k])
        logits /= self.temperature
        labels = (torch.arange(logits.shape[0], dtype=torch.long)).to(q.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def forward(self, out0: torch.Tensor, out1: torch.Tensor):
        q0, q1 = out0
        k0, k1 = out1
        loss0 = self._compute_loss(q0, k1)
        loss1 = self._compute_loss(q1, k0)
        loss = loss0 + loss1
        return loss
