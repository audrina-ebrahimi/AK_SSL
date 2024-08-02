import torch
import torch.nn as nn

from models.clip import CLIP

from AK_SSL.vision.models.modules.transformations import SimCLRViewTransform


class SLIP(nn.Module):
    """
    SLIP: Self-supervision meets Language-Image Pre-training
    Link: https://arxiv.org/abs/2112.12750
    Implementation: https://github.com/facebookresearch/SLIP
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        mlp_dim: int = 4096,
        vision_feature_dim: int = 0,
        transformer_feature_dim: int = 768,
        embed_dim: int = 256,
    ) -> None:
        """
        Args:
            image_encoder (nn.Module): Vision encoder model
            text_encoder (nn.Module): Transformer encoder model
            mlp_dim (int, optional): Dimension of the MLP. Defaults to 4096.
            vision_feature_dim (int, optional): Dimension of the vision features. Defaults to 0.
            transformer_feature_dim (int, optional): Dimension of the transformer features. Defaults to 768.
            embed_dim (int, optional): Dimension of the embeddings. Defaults to 256.
        """
        super(SLIP, self).__init__()

        self.clip = CLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
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

    def forward(
        self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict:
        augmented_image_1 = SimCLRViewTransform(image)
        augmented_image_2 = SimCLRViewTransform(image)

        aug1_embed = self.vision_mlp(self.clip.image_encoder(augmented_image_1))
        aug2_embed = self.vision_mlp(self.clip.image_encoder(augmented_image_2))

        clip_output = self.clip(image, input_ids, attention_mask)

        return {
            "aug1_embed": aug1_embed,
            "aug2_embed": aug2_embed,
            "clip_output": clip_output,
            "logit_scale": self.logit_scale.exp(),
        }

    def criterion(
        self, ssl_scale: float, ssl_loss: torch.Tensor, clip_loss: torch.Tensor
    ) -> torch.Tensor:
        return ssl_scale * ssl_loss + clip_loss
