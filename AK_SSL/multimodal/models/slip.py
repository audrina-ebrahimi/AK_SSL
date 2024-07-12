import torch
import torch.nn as nn

from models.clip import CLIP

from AK_SSL.vision.models.modules.transformations import SimCLRViewTransform


class SLIP(nn.Module):
    def __init__(
        self,
        vision_model: nn.Module,
        transformer_model: nn.Module,
        mlp_dim: int = 4096,
        vision_feature_dim: int = 0,
        transformer_feature_dim: int = 768,
        embed_dim: int = 256,
    ) -> None:
        super(SLIP, self).__init__()

        self.clip = CLIP(
            image_encoder=vision_model,
            text_encoder=transformer_model,
            image_feature_dim=vision_feature_dim,
            text_feature_dim=transformer_feature_dim,
            embed_dim=embed_dim,
        )

        self.vision_mlp = nn.Sequential(
            nn.Linear(vision_feature_dim, mlp_dim),
            nn.SyncBatchNorm(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SyncBatchNorm(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> dict:
        augmented_image_1 = SimCLRViewTransform(image)
        augmented_image_2 = SimCLRViewTransform(image)

        aug1_embed = self.vision_mlp(self.clip.vision_model(augmented_image_1))
        aug2_embed = self.vision_mlp(self.clip.vision_model(augmented_image_2))

        image_embed = self.clip.extract_image_features(image)
        text_embed = self.clip.extract_text_features(text)

        return {
            "image_embed": image_embed,
            "text_embed": text_embed,
            "aug1_embed": aug1_embed,
            "aug2_embed": aug2_embed,
            "logit_scale": self.logit_scale.exp(),
        }

    def criterion(
        self, ssl_temp: float, ssl_loss: torch.Tensor, clip_loss
    ) -> torch.Tensor:
        return ssl_temp * ssl_loss + clip_loss
