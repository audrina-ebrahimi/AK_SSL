import torch
import torch.nn as nn

from AK_SSL.multimodal.models.clip import CLIP
from AK_SSL.vision.models.modules.transformations import SimCLRViewTransform


class SLIP(nn.Module):
    """
    SLIP: Self-supervision meets Language-Image Pre-training
    Paper Link: https://arxiv.org/abs/2112.12750
    Implementation Link: https://github.com/facebookresearch/SLIP
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        mlp_dim: int = 4096,
        image_feature_dim: int = 0,
        text_feature_dim: int = 768,
        embed_dim: int = 256,
        device: str = "cpu",
        **kwargs
    ) -> None:
        """
        Initialize the SLIP model.

        Args:
            image_encoder (nn.Module): Neural network to encode images
            text_encoder (nn.Module): Neural network to encode text
            mlp_dim (int): Dimension of the hidden layer in the MLP. (default: 4096)
            image_feature_dim (int): Dimensionality of image features (default: 0, will be determined later)
            text_feature_dim (int): Dimensionality of text features (default: 768)
            embed_dim (int): Dimensionality of the joint embedding space (default: 256)
        """
        super(SLIP, self).__init__()

        self.mlp_dim = mlp_dim
        self.image_feature_dim = image_feature_dim
        self.text_feature_dim = text_feature_dim
        self.embed_dim = embed_dim

        # Initialize the CLIP model with the given encoders and dimensions
        self.clip = CLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            image_feature_dim=self.image_feature_dim,
            text_feature_dim=self.text_feature_dim,
            embed_dim=self.embed_dim,
            device=device,
        )

        self.image_feature_dim = self.clip.image_feature_dim

        # Define the vision MLP for feature transformation and projection
        self.vision_mlp = nn.Sequential(
            nn.Linear(self.image_feature_dim, self.mlp_dim),
            nn.SyncBatchNorm(self.mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.SyncBatchNorm(self.mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.mlp_dim, embed_dim),
        )

        self.simclr_view_transform = SimCLRViewTransform(**kwargs)

    def forward(
        self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict:

        # Apply SimCLR transformation to the image twice to get two augmented views
        augmented_image_1 = self.simclr_view_transform(image)
        augmented_image_2 = self.simclr_view_transform(image)

        # Pass the augmented images through the vision MLP
        aug1_embed = self.vision_mlp(self.clip.image_encoder(augmented_image_1))
        aug2_embed = self.vision_mlp(self.clip.image_encoder(augmented_image_2))

        # Get the CLIP model's output for the original image and text inputs
        clip_output = self.clip(image, input_ids, attention_mask)

        return {
            "aug1_embed": aug1_embed,
            "aug2_embed": aug2_embed,
            "clip_output": clip_output,
        }

    def criterion(
        self, ssl_scale: float, ssl_loss: torch.Tensor, clip_loss: torch.Tensor
    ) -> torch.Tensor:
        return ssl_scale * ssl_loss + clip_loss
