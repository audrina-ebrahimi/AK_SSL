import copy

import torch
import torch.nn as nn

from AK_SSL.vision.models.modules.heads import DINOProjectionHead


class DINO(nn.Module):
    """
    DINO: Emerging Properties in Self-Supervised Vision Transformers
    Link: https://arxiv.org/abs/2104.14294
    Implementation: https://github.com/facebookresearch/dino
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_size: int,
        projection_dim: int = 256,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        temp_student: float = 0.1,
        temp_teacher: float = 0.5,
        projection_num_layers: int = 3,
        norm_last_layer: bool = True,
        momentum_teacher: float = 0.996,
        num_crops: int = 6,
        use_bn_in_head: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_size = feature_size
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher
        self.norm_last_layer = norm_last_layer
        self.use_bn_in_head = use_bn_in_head
        self.momentum_teacher = momentum_teacher  # EMA update
        self.num_crops = num_crops
        self.projection_num_layers = projection_num_layers

        self.student_projection_head = DINOProjectionHead(
            input_dim=self.feature_size,
            hidden_dim=self.hidden_dim,
            output_dim=self.projection_dim,
            bottleneck_dim=self.bottleneck_dim,
            use_bn=self.use_bn_in_head,
            norm_last_layer=self.norm_last_layer,
            num_layers=self.projection_num_layers,
        )
        self.student = self.encoder = nn.Sequential(
            self.backbone, self.student_projection_head
        )
        self.teacher_projection_head = DINOProjectionHead(
            input_dim=self.feature_size,
            hidden_dim=self.hidden_dim,
            output_dim=self.projection_dim,
            bottleneck_dim=self.bottleneck_dim,
            use_bn=self.use_bn_in_head,
            norm_last_layer=self.norm_last_layer,
            num_layers=self.projection_num_layers,
        )
        self.teacher = nn.Sequential(
            copy.deepcopy(self.backbone), self.teacher_projection_head
        )

        self._init_teacher()

    def _init_teacher(self):
        for param_q, param_k in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_teacher(self):
        for param_q, param_k in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            param_k.data = (
                self.momentum_teacher * param_k.data
                + (1.0 - self.momentum_teacher) * param_q.data
            )

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, xc: list):
        z1_s, z2_s = self.student(x0), self.student(x1)

        zc_s = []
        for x in xc:
            zc_s.append(self.student(x))

        with torch.no_grad():
            self._momentum_update_teacher()
            z1_t, z2_t = self.teacher(x0), self.teacher(x1)

        z_s = [z1_s, z2_s] + zc_s
        z_t = [z1_t, z2_t]

        return z_s, z_t
