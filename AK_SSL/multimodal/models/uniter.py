"""
Copyright (c) Microsoft Corporation.

some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)

"""

import torch
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm


class UNITERForVQA(nn.Module):
    """
    UNITER: UNiversal Image-TExt Representation Learning
    Link: https://arxiv.org/pdf/1909.11740
    Implementation: https://github.com/ChenRocks/UNITER

    Args:
        image_encoder (nn.Module): image encoder
        text_encoder (nn.Module): text encoder
        pooler (nn.Module): pooler
        encoder (nn.Module): transformer encoder
        num_answer (int): number of answer classes
        hidden_size (int): hidden size
        attention_probs_dropout_prob (float): dropout rate
        initializer_range (float): initializer range
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        pooler: nn.Module,
        encoder: nn.Module,
        num_answer: int,
        hidden_size: int = 768,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.pooler = pooler
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range

        self.text_embeddings = text_encoder
        self.image_embeddings = image_encoder

        self.vqa_output = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            FusedLayerNorm(self.hidden_size * 2, eps=1e-12),
            nn.Linear(self.hidden_size * 2, num_answer),
        )

        self.apply(self.init_weights)

    def _compute_txt_embeddings(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        txt_type_ids: torch.Tensor = None,
    ):
        output = self.text_embeddings(input_ids, position_ids, txt_type_ids)
        return output

    def _compute_img_embeddings(
        self,
        img_feat: torch.Tensor,
        img_pos_feat: torch.Tensor,
        img_masks: torch.Tensor = None,
        img_type_ids: torch.Tensor = None,
    ):
        if img_type_ids is None:
            img_type_ids = torch.ones_like(img_feat[:, :, 0].long())
        img_type_embeddings = self.text_embeddings.token_type_embeddings(img_type_ids)
        output = self.image_embeddings(
            img_feat, img_pos_feat, img_type_embeddings, img_masks
        )
        return output

    def _compute_img_txt_embeddings(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        img_feat: torch.Tensor,
        img_pos_feat: torch.Tensor,
        gather_index: torch.Tensor,
        img_masks: torch.Tensor = None,
        txt_type_ids: torch.Tensor = None,
        img_type_ids: torch.Tensor = None,
    ):
        txt_emb = self._compute_txt_embeddings(input_ids, position_ids, txt_type_ids)
        img_emb = self._compute_img_embeddings(
            img_feat, img_pos_feat, img_masks, img_type_ids
        )
        # align back to most compact input
        gather_index = gather_index.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        embedding_output = torch.gather(
            torch.cat([txt_emb, img_emb], dim=1), dim=1, index=gather_index
        )
        return embedding_output

    def init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        img_feat: torch.Tensor,
        img_pos_feat: torch.Tensor,
        attention_mask: torch.Tensor,
        gather_index: torch.Tensor = None,
        img_masks: torch.Tensor = None,
        output_all_encoded_layers: torch.Tensor = True,
        txt_type_ids: torch.Tensor = None,
        img_type_ids: torch.Tensor = None,
    ):
        # compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding layer
        if input_ids is None:
            # image only
            embedding_output = self._compute_img_embeddings(
                img_feat, img_pos_feat, img_masks, img_type_ids
            )
        elif img_feat is None:
            # text only
            embedding_output = self._compute_txt_embeddings(
                input_ids, position_ids, txt_type_ids
            )
        else:
            embedding_output = self._compute_img_txt_embeddings(
                input_ids,
                position_ids,
                img_feat,
                img_pos_feat,
                gather_index,
                img_masks,
                txt_type_ids,
                img_type_ids,
            )

        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
        )
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        pooled_output = self.uniter.pooler(encoded_layers)
        answer_scores = self.vqa_output(pooled_output)
        return answer_scores

    def criterion(self, targets, answer_scores):
        vqa_loss = F.binary_cross_entropy_with_logits(
            answer_scores, targets, reduction="none"
        )
        return vqa_loss
