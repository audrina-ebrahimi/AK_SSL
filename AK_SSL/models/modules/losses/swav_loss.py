import torch
import torch.nn as nn
import torch.nn.functional as F


class SwAVLoss(nn.Module):
    def __init__(self, num_crops: int, temperature: float = 0.1, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.num_crops = num_crops

    def cross_entropy_loss(self, q, p):
        return torch.mean(torch.sum(q * F.log_softmax(p, dim=1), dim=1))

    def forward(self, c, q):
        loss = 0

        c1, c2, c_c = c
        q1, q2 = q

        p1, p2 = c1 / self.temperature, c2 / self.temperature
        loss += self.cross_entropy_loss(q1, p2) / (self.num_crops - 1)
        loss += self.cross_entropy_loss(q2, p1) / (self.num_crops - 1)

        for c in range(len(c_c)):
            p = c_c[c] / self.temperature
            loss += self.cross_entropy_loss(q1, p) / (self.num_crops - 1)
            loss += self.cross_entropy_loss(q2, p) / (self.num_crops - 1)

        return loss / 2
