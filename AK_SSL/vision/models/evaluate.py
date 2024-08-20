import torch
import torch.nn as nn


class EvaluateNet(nn.Module):
    """
    EvaluateNet: Perfoming a classifier based on a pre-trained backbone.
    """

    def __init__(
        self, backbone: nn.Module, feature_size: int, num_classes: int, is_linear: bool
    ):
        """
        Args:
            backbone (nn.Module): Backbone to extract features.
            feature_size (int): Feature size.
            num_classes (int): Number of classes.
            is_linear (bool): Whether to use a linear classifier.
        """
        super().__init__()
        self.backbone = backbone

        for param in self.backbone.parameters():
            param.requires_grad = not is_linear

        self.fc = nn.Linear(feature_size, num_classes, bias=True)

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.fc(x)
        return x
