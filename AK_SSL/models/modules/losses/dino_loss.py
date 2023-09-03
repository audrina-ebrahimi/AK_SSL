import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    def __init__(
        self,
        projection_dim: int,
        temp_student: float,
        temp_teacher: float,
        center_momentum: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.projection_dim = projection_dim
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, self.projection_dim))

    def forward(self, student_output: list, teacher_output: list):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_output):
            for iv, v in enumerate(student_output):
                if iv == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                total_loss += self.cross_entropy_loss(q, v)
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self._update_center(teacher_output[0], teacher_output[1])
        return total_loss

    @torch.no_grad()
    def _update_center(self, z0_t, z1_t):
        self.center = self.center_momentum * self.center + (
            1 - self.center_momentum
        ) * torch.cat([z0_t, z1_t]).mean(dim=0)

    def cross_entropy_loss(self, z_t, z_s):
        z_t = z_t.detach()  # stop gradient
        z_s = z_s / self.temp_student
        z_t = F.softmax(
            (z_t - self.center) / self.temp_teacher, dim=1
        )  # center + sharpen
        return -(z_t * F.log_softmax(z_s, dim=1)).sum(dim=1).mean()
