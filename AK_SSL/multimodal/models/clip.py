import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CLIP(nn.Module):
    """
    CLIP: Contrastive Language-Image Pre-training
    Link: https://arxiv.org/abs/2103.00020
    Implementation: https://github.com/openai/CLIP
    """

    def __init__(
        self,
        vision_model: nn.Module,
        transformer_model: nn.Module,
        vision_feature_dim: int = 0,
        transformer_feature_dim: int = 768,
        embed_dim: int = 256,
        init_tau: float = np.log(1.0),
        init_b: float = 0.0,
    ):
        super(CLIP, self).__init__()

        if not vision_feature_dim:
            vision_feature_dim = self.get_feature_size(vision_model)

        self.vision_model = vision_model
        self.transformer_model = transformer_model

        self.image_projection = torch.nn.Sequential(
            torch.nn.Linear(vision_feature_dim, vision_feature_dim, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(vision_feature_dim, embed_dim, bias=False),
        )

        self.text_projection = torch.nn.Sequential(
            torch.nn.Linear(
                transformer_feature_dim, transformer_feature_dim, bias=False
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(transformer_feature_dim, embed_dim, bias=False),
        )

        self.t_prime = nn.Parameter(torch.ones([]) * init_tau)
        self.b = nn.Parameter(torch.ones([]) * init_b)

    def forward(
        self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ):
        image_features = self.extract_image_features(image)
        text_features = self.extract_text_features(input_ids, attention_mask)
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return image_features @ text_features.t() * self.t_prime.exp() + self.b

    def extract_image_features(self, images: torch.Tensor):
        image_features = self.vision_model(images)
        return self.image_projection(image_features)

    def extract_text_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ):
        text_features = self.transformer_model(input_ids, attention_mask)
        return self.text_projection(text_features)

    def get_feature_size(self, encoder: nn.Module):
        """Get the feature size from the encoder using a dummy input."""
        encoder.eval()
        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = encoder(dummy_input)
        return output.shape[1]

    def criterion_contrastive_loss(self, logits: torch.Tensor):
        targets = torch.arange(logits.size(0)).to(logits.device)
        loss_images = F.cross_entropy(logits, targets)
        loss_texts = F.cross_entropy(logits.t(), targets)
        return (loss_images + loss_texts) / 2

    def criterion_siglip_loss(self, logits: torch.Tensor):
        n = logits.size(0)
        # -1 --> off-diagonals
        #  1 --> diagonals
        labels = 2 * torch.eye(n, device=logits.device) - 1
        # pairwise sigmoid loss
        return -torch.sum(F.logsigmoid(labels * logits)) / n
