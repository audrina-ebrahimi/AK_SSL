import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIP(nn.Module):
    """
    CLIP: Contrastive Language-Image Pre-training
    Paper Link: https://arxiv.org/abs/2103.00020
    Implementation Link: https://github.com/openai/CLIP, https://github.com/filipbasara0/simple-clip
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        image_feature_dim: int = 0,
        text_feature_dim: int = 768,
        embed_dim: int = 256,
        init_tau: float = np.log(1.0),
        init_bias: float = 0.0,
        use_siglip: bool = False,
        device: str = 'cpu'
    ):
        """
        Initialize the CLIP model.
        Args:
            image_encoder (nn.Module): Neural network to encode images
            text_encoder (nn.Module): Neural network to encode text
            image_feature_dim (int): Dimensionality of image features (default: 0, will be determined later)
            text_feature_dim (int): Dimensionality of text features (default: 768)
            embed_dim (int): Dimensionality of the joint embedding space (default: 256)
            init_tau (float): Initial value for the temperature parameter (default: log(1.0))
            init_bias (float): Initial value for the bias parameter (default: 0.0)
            use_siglip (bool): Flag to indicate if SIGLIP loss should be used (default: False)
        """
        super(CLIP, self).__init__()
        
        self.device = device

        # Determine image feature dimensionality if not provided
        if image_feature_dim:
            self.image_feature_dim = image_feature_dim
        elif not image_feature_dim:
            self.image_feature_dim = self.get_feature_size(image_encoder)

        self.text_feature_dim = text_feature_dim
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.embed_dim = embed_dim
        self.use_siglip = use_siglip
        self.init_tau = init_tau
        self.init_bias = init_bias

        # Define the image projection network
        self.image_projection = torch.nn.Sequential(
            torch.nn.Linear(self.image_feature_dim, self.image_feature_dim, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(self.image_feature_dim, self.embed_dim, bias=False),
        )

        # Define the text projection network
        self.text_projection = torch.nn.Sequential(
            torch.nn.Linear(self.text_feature_dim, self.text_feature_dim, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(self.text_feature_dim, self.embed_dim, bias=False),
        )

        # Initialize temperature and bias parameters
        self.t_prime = nn.Parameter(torch.ones([]) * self.init_tau)
        self.b = nn.Parameter(torch.ones([]) * self.init_bias)

    def forward(
        self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ):
        # Extract and normalize image features
        image_features = self.extract_image_features(image)
        text_features = self.extract_text_features(input_ids, attention_mask)
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        # Compute similarity matrix with temperature scaling and bias
        return image_features @ text_features.t() * self.t_prime.exp() + self.b

    def extract_image_features(self, images: torch.Tensor):
        # Extract features from the image encoder and project them to the embedding space
        image_features = self.image_encoder(images)
        return self.image_projection(image_features)

    def extract_text_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ):
        # Extract features from the text encoder and project them to the embedding space
        text_features = self.text_encoder(input_ids, attention_mask)
        return self.text_projection(text_features)

    def get_feature_size(self, encoder: nn.Module):
        encoder = encoder.to(self.device)
        encoder.eval()
        dummy_input = torch.randn(1, 3, 32, 32).to(self.device)  # Create a dummy input for the encoder
        with torch.no_grad():
            output = encoder(dummy_input)  # Get the output features from the encoder
        return output.shape[1]  # Return the dimensionality of the output features

    def criterion_contrastive_loss(self, logits: torch.Tensor):
        # Create target labels for contrastive loss
        targets = torch.arange(logits.size(0)).to(logits.device)
        # Compute cross-entropy loss for images and texts
        loss_images = F.cross_entropy(logits, targets)
        loss_texts = F.cross_entropy(logits.t(), targets)
        # Return the average loss
        return (loss_images + loss_texts) / 2

    def criterion_siglip_loss(self, logits: torch.Tensor):
        n = logits.size(0)
        # Create labels with -1 for off-diagonals and 1 for diagonals
        labels = 2 * torch.eye(n, device=logits.device) - 1
        # Compute pairwise sigmoid loss
        return -torch.sum(F.logsigmoid(labels * logits)) / n
