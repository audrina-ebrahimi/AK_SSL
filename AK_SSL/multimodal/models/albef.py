# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import copy
from typing import Optional
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor


ALBEFOutput = namedtuple(
    "ALBEFOutput",
    [
        "image_embeddings",
        "image_embeddings_m",
        "text_embeddings",
        "text_embeddings_m",
        "multimodal_embeddings",
        "multimodal_embeddings_m",
    ],
    defaults=(None, None, None, None, None, None),
)


class ALBEFModel(nn.Module):
    """
    ALBEF: ALign the image and text representations BEfore Fusing
    Link: https://arxiv.org/pdf/2107.07651.pdf
    Implementation: https://github.com/salesforce/ALBEF

    Args:
        vision_encoder (nn.Module): Instantiated vision encoder.
        text_encoder (nn.Module): Instantiated text encoder.
        multimodal_encoder (nn.Module): Instantiated multimodal encoder.
        momentum (float): Momentum parameter. Default is 0.995.

    Inputs:
        image (Tensor): Tensor of shape (B, C, H, W) containing image features.
        text (Tensor): Tensor of shape (B, L) containing text features.
        text_atts (Tensor): Tensor of shape (B, L) containing text attention mask.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        multimodal_encoder: nn.Module,
        momentum: float = 0.995,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.multimodal_encoder = multimodal_encoder
        self.vision_encoder_m = copy.deepcopy(vision_encoder)
        self.text_encoder_m = copy.deepcopy(text_encoder)
        self.multimodal_encoder_m = copy.deepcopy(multimodal_encoder)

        self.remove_grad(self.vision_encoder_m)
        self.remove_grad(self.text_encoder_m)
        self.remove_grad(self.multimodal_encoder_m)
        self.momentum = momentum

    def forward(
        self,
        image: Tensor,
        text: Tensor,
        text_atts: Tensor,
    ) -> ALBEFOutput:
        image_embeds = self.vision_encoder(image)
        text_embeds = self.text_encoder(text, text_atts)
        multimodal_embeddings = self.multimodal_encoder(
            hidden_states=text_embeds.last_hidden_state,
            attention_mask=text_atts,
            encoder_hidden_states=image_embeds,
        )

        with torch.no_grad():
            self.momentum_update(
                self.vision_encoder, self.vision_encoder_m, self.momentum
            )
            self.momentum_update(self.text_encoder, self.text_encoder_m, self.momentum)
            self.momentum_update(
                self.multimodal_encoder, self.multimodal_encoder_m, self.momentum
            )
            image_embeds_m = self.vision_encoder_m(image)
            text_embeds_m = self.text_encoder_m(text, text_atts)
            multimodal_embeddings_m = self.multimodal_encoder_m(
                hidden_states=text_embeds_m.last_hidden_state,
                attention_mask=text_atts,
                encoder_hidden_states=image_embeds_m,
            )

        return ALBEFOutput(
            image_embeddings=image_embeds,
            image_embeddings_m=image_embeds_m,
            text_embeddings=text_embeds.last_hidden_state,
            text_embeddings_m=text_embeds_m.last_hidden_state,
            multimodal_embeddings=multimodal_embeddings,
            multimodal_embeddings_m=multimodal_embeddings_m,
        )

    @torch.no_grad()
    def remove_grad(self, model: nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def momentum_update(
        self, model: nn.Module, model_m: nn.Module, momentum: float
    ) -> None:
        for param, param_m in zip(model.parameters(), model_m.parameters()):
            param_m.data = param_m.data * momentum + param.data * (1 - momentum)

    def image_text_contrastive_loss(
        self,
        image_to_text_sim: Tensor,
        text_to_image_sim: Tensor,
        image_to_text_sim_m: Optional[Tensor] = None,
        text_to_image_sim_m: Optional[Tensor] = None,
        sim_targets: Optional[Tensor] = None,
        alpha: Optional[float] = 0.0,
    ) -> Tensor:
        """
        Inputs:
            image_to_text_sim (Tensor): Image to text similarity.
            text_to_image_sim (Tensor): Text to image similarity.
            image_to_text_sim_m (Optional[Tensor]): Image to text similarity from momentum models. (Required if alpha is non-zero.)
            text_to_image_sim_m (Optional[Tensor]): Text to image similarity from momentum models. (Required if alpha is non-zero.)
            sim_targets (Optional[Tensor]): Similarity pseudo-targets from momentum models. Default is the diagonal matrix. (Requires all Tensor inputs to have the same size.)
            alpha (Optional[float]): The interpolation value of momentum similarity and sim_targets. (Default is 0.)
        """

        if sim_targets is None:
            sim_targets = torch.zeros(image_to_text_sim.size()).to(
                image_to_text_sim.device
            )
            sim_targets.fill_diagonal_(1)

        if alpha != 0:
            assert (
                image_to_text_sim_m is not None and text_to_image_sim_m is not None
            ), "sim_i2t_m and sim_t2i_m cannot be none for non-zero alpha"

            with torch.no_grad():
                image_to_text_sim_targets = (
                    alpha * F.softmax(image_to_text_sim_m, dim=1)
                    + (1 - alpha) * sim_targets
                )
                text_to_image_sim_targets = (
                    alpha * F.softmax(text_to_image_sim_m, dim=1)
                    + (1 - alpha) * sim_targets
                )
        else:
            image_to_text_sim_targets = sim_targets
            text_to_image_sim_targets = sim_targets

        loss_i2t = -torch.sum(
            F.log_softmax(image_to_text_sim, dim=1) * image_to_text_sim_targets, dim=1
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(text_to_image_sim, dim=1) * text_to_image_sim_targets, dim=1
        ).mean()

        loss_itc = (loss_i2t + loss_t2i) / 2
        return loss_itc

    def causal_language_modeling_loss(
        self,
        labels: Tensor,
        prediction_scores: Tensor,
        prediction_scores_m: Optional[Tensor] = None,
        mask_token_id: int = -100,
        alpha: Optional[float] = 0.0,
    ) -> Tensor:
        """
        Inputs:
            mask_token_id (int): The token id indicating a masked token. Default is -100.
            labels (Tensor of shape (batch_size, seq_length)): The masked output tokens.
            prediction_scores (Tensor of shape (batch_size, seq_length, vocab_size)): The prediction scores from a prediction head.
            prediction_scores_m (Optional[Tensor] of shape (batch_size, seq_length, vocab_size)): The prediction scores from a momentum prediction head.(Required if alpha is non-zero.)
            alpha (float): The interpolation value between mlm_loss and loss_distill. (Default is 0.)
        """

        batch_size = labels.size(0)
        # shift prediction scores and labels by one for next-token predict
        prediction_scores = prediction_scores[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        mlm_loss = F.cross_entropy(
            prediction_scores.view(-1, prediction_scores.shape[-1]),
            labels.view(-1),
            reduction="none",
        )
        mlm_loss = mlm_loss.view(batch_size, -1).sum(1)

        if alpha != 0:
            assert (
                prediction_scores_m is not None
            ), "prediction_scores_m cannot be None for non-zero alpha"

            with torch.no_grad():
                prediction_scores_m = prediction_scores_m[:, :-1, :].contiguous()
            loss_distill = -torch.sum(
                F.log_softmax(prediction_scores, dim=-1)
                * F.softmax(prediction_scores_m, dim=-1),
                dim=-1,
            )
            loss_distill = (loss_distill * (labels != mask_token_id)).sum(1)
            mlm_loss = (1 - alpha) * mlm_loss + alpha * loss_distill

        return mlm_loss
