import torch
import torch.nn as nn
import torch.nn.functional as F


class BYOLLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def loss_fn(x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward(self, out0: torch.Tensor, out1: torch.Tensor):
        online_pred_one, target_proj_one = out0
        online_pred_two, target_proj_two = out1

        loss_one = self.loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = self.loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()
