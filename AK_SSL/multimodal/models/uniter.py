"""
Copyright (c) Microsoft Corporation.

some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)

"""

import torch
from torch import nn
from apex.normalization.fused_layer_norm import FusedLayerNorm


class UniterTextEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        hidden_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = FusedLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class UniterImageEmbeddings(nn.Module):
    def __init__(
        self, img_dim: int, hidden_size: int = 768, hidden_dropout_prob: float = 0.1
    ):
        super().__init__()
        self.img_linear = nn.Linear(img_dim, hidden_size)
        self.img_layer_norm = FusedLayerNorm(hidden_size, eps=1e-12)
        self.pos_layer_norm = FusedLayerNorm(hidden_size, eps=1e-12)
        self.pos_linear = nn.Linear(7, hidden_size)
        self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)

        # tf naming convention for layer norm
        self.LayerNorm = FusedLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        img_feat: torch.Tensor,
        img_pos_feat: torch.Tensor,
        type_embeddings: torch.Tensor,
        img_masks: torch.Tensor = None,
    ):
        if img_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(img_masks.long())
            img_feat = img_feat + mask

        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_pos = self.pos_layer_norm(self.pos_linear(img_pos_feat))
        embeddings = transformed_im + transformed_pos + type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class UniterModel(nn.Module):
    """
    UNITER: UNiversal Image-TExt Representation Learning
    Link: https://arxiv.org/pdf/1909.11740
    Implementation: https://github.com/ChenRocks/UNITER

    Args:
        vocab_size_or_config_json_file: Vocabulary size of `inputs_ids`.
        hidden_size: Size of the encoder layers and the pooler layer.
        num_hidden_layers: Number of hidden layers in the Transformer encoder.
        num_attention_heads: Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size: The size of the "intermediate" (i.e. feed-forward) layer in the Transformer encoder.
        hidden_act: The non-linear activation function (function or string) in the encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
        hidden_dropout_prob: The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob: The dropout ratio for the attention probabilities.
        max_position_embeddings: The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size: The vocabulary size of the `token_type_ids`
        initializer_range: The sttdev of the truncated_normal_initializer for initializing all weight matrices.
    """

    def __init__(
        self,
        img_dim: int,
        pooler: nn.Module,
        encoder: nn.Module,
        vocab_size: int,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.pooler = pooler
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

        self.text_embeddings = UniterTextEmbeddings(
            self.vocab_size,
            self.hidden_size,
            self.max_position_embeddings,
            self.type_vocab_size,
            self.hidden_dropout_prob,
        )
        self.image_embeddings = UniterImageEmbeddings(
            self, img_dim, self.hidden_size, self.hidden_dropout_prob
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
        return encoded_layers
