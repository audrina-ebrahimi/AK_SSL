"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy


class ALBEF(nn.Module):
    """
    ALBEF: ALign the image and text representations BEfore Fusing
    Link: https://arxiv.org/pdf/2107.07651.pdf
    Implementation: https://github.com/salesforce/ALBEF

    Args:
        vision_encoder (nn.Module): Instantiated vision encoder.
        text_encoder (nn.Module): Instantiated text encoder.
        mlm_probability (nn.Module): Masked language modeling probability. Default is 0.15.
        embed_dim (int): Embedding dimension. Default is 768.
        vision_width (int): Vision encoder output width. Default is 256.
        temp (float): Temperature parameter. Default is 0.07.
        queue_size (int): Queue size. Default is 1024.
        momentum (float): Momentum parameter. Default is 0.9.


    """

    def __init__(
        self,
        vision_encoder: nn.Module = None,
        text_encoder: nn.Module = None,
        mlm_probability: float = 0.15,
        embed_dim: int = 768,
        vision_width: int = 256,
        temp: float = 0.07,
        queue_size: int = 1024,
        momentum: float = 0.9,
    ):
        super().__init__()

        self.visual_encoder = vision_encoder
        self.text_encoder = text_encoder

        self.mlm_probability = mlm_probability
        self.embed_dim = embed_dim
        self.temp = nn.Parameter(torch.ones([]) * temp)
        self.queue_size = queue_size
        self.momentum = momentum
        self.itm_head = nn.Linear(self.text_width, 2)

        self.vision_width = vision_width
        self.text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)

        # create momentum models
        self.visual_encoder_m = deepcopy(self.visual_encoder)
        self.vision_proj_m = nn.Linear(self.vision_width, self.embed_dim)

        self.text_encoder_m = deepcopy(self.text_encoder)
        self.text_proj_m = nn.Linear(self.text_width, self.embed_dim)

        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
        ]

        self.copy_params()

        # create the queue
        self.register_buffer(
            "image_queue", torch.randn(self.embed_dim, self.queue_size)
        )
        self.register_buffer("text_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def forward(self, image: torch.Tensor, text: torch.Tensor, alpha: float = 0):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        text_output = self.text_encoder.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image)
            image_feat_m = F.normalize(
                self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1
            )
            image_feat_all = torch.cat(
                [image_feat_m.t(), self.image_queue.clone().detach()], dim=1
            )
            text_output_m = self.text_encoder_m.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            text_feat_m = F.normalize(
                self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1
            )
            text_feat_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1
            )

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = (
                alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            )
            sim_t2i_targets = (
                alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            )

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1
        ).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        # forward the positve image-text pair
        output_pos = self.text_encoder.bert(
            encoder_embeds=text_embeds,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode="fusion",
        )
        with torch.no_grad():
            bs = image.size(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_encoder.bert(
            encoder_embeds=text_embeds_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
            mode="fusion",
        )

        vl_embeddings = torch.cat(
            [
                output_pos.last_hidden_state[:, 0, :],
                output_neg.last_hidden_state[:, 0, :],
            ],
            dim=0,
        )
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        # MLM
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        input_ids, labels = self.mask(
            input_ids,
            self.text_encoder.config.vocab_size,
            image.device,
            targets=labels,
            probability_matrix=probability_matrix,
        )

        with torch.no_grad():
            logits_m = self.text_encoder_m(
                input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds_m,
                encoder_attention_mask=image_atts,
                return_dict=True,
                return_logits=True,
            )
        mlm_output = self.text_encoder(
            input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            labels=labels,
            soft_labels=F.softmax(logits_m, dim=-1),
            alpha=alpha,
        )
        loss_mlm = mlm_output.loss

        return loss_mlm, loss_ita, loss_itm

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data = param_m.data * self.momentum + param.data * (
                    1.0 - self.momentum
                )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr : ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr : ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
