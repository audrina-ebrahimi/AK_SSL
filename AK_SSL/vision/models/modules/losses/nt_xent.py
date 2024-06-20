import torch
import torch.nn as nn


class NT_Xent(nn.Module):
    def __init__(self, temperature: float = 0.5, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.eps = 1e-8
        if abs(self.temperature) < self.eps:
            raise ValueError(
                "Illegal temperature: abs({}) < 1e-8".format(self.temperature)
            )

    def forward(self, out0: torch.Tensor, out1: torch.Tensor):
        device = out0.device
        batch_size, _ = out0.shape

        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        out = torch.cat((out0, out1), dim=0)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (
            torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=device)
        ).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        pos_sim = torch.exp(torch.sum(out0 * out1, dim=-1) / self.temperature)
        pos_sim = torch.cat((pos_sim, pos_sim), dim=0)
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss
