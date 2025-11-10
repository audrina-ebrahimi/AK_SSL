from random import randint
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange

from AK_SSL.multimodal.models.utils.simvlm.resblock import BottleneckBlock


class ResBlock(nn.Sequential):
    def __init__(
        self,
        kernel_size: int,
        stride: int,
        in_channels: int = 3,
        out_channels: int = 256,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=kernel_size,
                stride=stride,
                bias=True,
            ),
            BottleneckBlock(in_channels=64, out_channels=256, bottleneck_channels=64),
            BottleneckBlock(
                in_channels=256, out_channels=out_channels, bottleneck_channels=128
            ),
        )


class SimVLM(nn.Module):
    """
    SimVLM: Simple Visual Language Model Pretraining with Weak Supervision
    Paper Link: https://arxiv.org/abs/2108.10904
    Implementation Link: https://github.com/YulongBonjour/SimVLM
    """

    def __init__(
        self,
        transformer_encoder: nn.Module,
        transformer_decoder: nn.Module,
        vocab_size: int = 10000,
        feature_dim: int = 512,
        max_seq_len: int = 60,
        max_trunc_txt_len: int = 15,
        prefix_txt_len: int = 20,
        target_txt_len: int = 60,
        pad_idx: int = 0,
        image_resolution: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
    ) -> None:
        """
        Initialize the SimVLM model.

        Args:
            transformer_encoder (nn.Module): The transformer encoder.
            transformer_decoder (nn.Module): The transformer decoder.
            vocab_size (int, optional): The size of the vocabulary. Defaults to 10000.
            feature_dim (int, optional): The dimension of the features. Defaults to 512.
            max_seq_len (int, optional): The maximum sequence length. Defaults to 60.
            max_trunc_txt_len (int, optional): The maximum truncated text length. Defaults to 15.
            prefix_txt_len (int, optional): The prefix text length. Defaults to 20.
            target_txt_len (int, optional): The target text length. Defaults to 60.
            pad_idx (int, optional): The padding index. Defaults to 0.
            image_resolution (int, optional): The image resolution. Defaults to 224.
            patch_size (int, optional): The patch size. Defaults to 16.
            num_channels (int, optional): The number of channels. Defaults to 3.
        """
        super(SimVLM, self).__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.max_trunc_txt_len = max_trunc_txt_len
        self.prefix_txt_len = prefix_txt_len
        self.pad_idx = pad_idx
        self.feature_dim = feature_dim
        self.target_txt_len = target_txt_len
        self.image_resolution = image_resolution
        self.patch_size = patch_size
        self.num_channels = num_channels

        # Define word embedding and positional embedding layers for text
        self.word_embedding = nn.Embedding(self.vocab_size, feature_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, feature_dim)
        image_feat_map_size = self.image_resolution // self.patch_size
        self.image_tokens_len = image_feat_map_size**2

        # Define a patch embedding layer for image patches
        self.patch_embedding = AxialPositionalEmbedding(
            feature_dim, axial_shape=(image_feat_map_size, image_feat_map_size)
        )

        # Define a ResBlock for image feature extraction
        self.ResBlock = ResBlock(
            in_channels=self.num_channels,
            out_channels=feature_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Define a sequence of normalization and linear layers to convert features to logits
        self.to_logits = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, self.vocab_size)
        )

        # Assign transformer encoder and decoder
        self.transformer_encoder = transformer_encoder
        self.transformer_decoder = transformer_decoder

        # Define a parameter for the beginning of sequence embedding
        self.bos_emb = nn.Parameter(torch.randn(1, feature_dim))

        # Initialize model parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initiate parameters in the transformer model.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(sz: int):
        """
        Generate a subsequent mask to prevent attending to future positions
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:

        device, n = text.device, text.shape[0]

        # Extract image embeddings
        image_embeddings = self.ResBlock(image)
        image_embeddings = rearrange(image_embeddings, "b c h w -> b (h w) c")
        image_embeddings += self.patch_embedding(image_embeddings)

        # Create key padding mask for attention
        key_padding_mask = torch.zeros(
            n,
            self.image_tokens_len + self.prefix_txt_len,
            dtype=torch.bool,
            device=device,
        )

        # Embed text and add positional encodings
        text_embeddings = self.word_embedding(text) + self.pos_embedding(
            torch.arange(self.max_seq_len, device=device)
        )  # b n c

        # Randomly truncate text length
        l = randint(0, self.max_trunc_txt_len)

        # Create prefix text tensor with padding
        prefix_text = torch.zeros(
            (n, self.prefix_txt_len), device=device, dtype=torch.long
        )

        prefix_text[:, :l] = text[:, :l]
        key_padding_mask[:, self.image_tokens_len :] = prefix_text == self.pad_idx

        # Create prefix text embeddings
        prefix_text_embeddings = torch.zeros(
            (n, self.prefix_txt_len, self.feature_dim), device=device
        )
        prefix_text_embeddings[:, :l] = text_embeddings[:, :l]

        # Create target text embeddings
        target_text_embeddings = torch.zeros(
            (n, self.target_txt_len, self.feature_dim), device=device
        )
        target_text_embeddings[:, : (self.max_seq_len - l)] = text_embeddings[:, l:]

        # Create labels for the target text
        labels = torch.zeros((n, self.target_txt_len), device=device, dtype=torch.long)
        labels[:, : (self.txt_seq_len - l)] = text[:, l:]

        del text, image

        # Add beginning of sequence embedding to target text embeddings
        target_text_embeddings = torch.cat(
            [
                torch.zeros(n, 1, self.feature_dim, device=device) + self.bos_emb,
                target_text_embeddings[:, :-1],
            ],
            dim=1,
        )

        # Concatenate image and prefix text embeddings
        prefix = torch.cat((image_embeddings, prefix_text_embeddings), dim=1)

        target_text_embeddings = rearrange(target_text_embeddings, "b n c -> n b c")
        prefix = rearrange(prefix, "b n c -> n b c")
        target_mask = self.generate_square_subsequent_mask(self.target_txt_len).to(
            device
        )

        # Encode the prefix using the transformer encoder
        memory = self.encoder(prefix, mask=None, src_key_padding_mask=key_padding_mask)

        # Decode the target text embeddings using the transformer decoder
        output = self.decoder(
            target_text_embeddings,
            memory,
            tgt_mask=target_mask,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=key_padding_mask,
        )

        # Convert the decoder output to logits
        logits = self.to_logits(output)  # seq_len, batch,vocab_size
        return logits, labels

    def criterion(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Define the loss function (cross entropy with padding ignored)
        """
        return F.cross_entropy(logits, labels, ignore_index=0)

    def generate(
        self,
        image: torch.Tensor,
        prefix_text: torch.Tensor,
        sampling_method: str = "nucleus",
        eos_id: int = 0,
        top_k: int = 256,
        top_p: float = 0.9,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        device = image.device
        n = image.shape[0]

        # Extract image embeddings
        image_embeddings = self.ResBlock(image)
        image_embeddings = rearrange(image_embeddings, "b c h w -> b (h w) c")
        image_embeddings += self.patch_embedding(image_embeddings)

        # Compute the length of the prefix text
        prefix_txt_len = (
            (prefix_text != self.pad_idx).sum(dim=-1).unsqueeze(-1)
        )  # [B,1]

        # Create key padding mask for attention
        key_padding_mask = torch.zeros(
            n,
            self.image_tokens_len + self.prefix_txt_len,
            dtype=torch.bool,
            device=device,
        )

        key_padding_mask[:, self.image_tokens_len :] = prefix_text == self.pad_idx

        # Embed prefix text and add positional encodings
        prefix_text_embeddings = self.txt_embed(prefix_text) + self.txt_pos_embed(
            torch.arange(self.prefix_txt_len, device=device)
        )

        # Concatenate image and prefix text embeddings
        prefix = torch.cat((image_embeddings, prefix_text_embeddings), dim=1)
        prefix = rearrange(prefix, "b n c -> n b c")

        # Encode the prefix using the transformer encoder
        memory = self.encoder(prefix, mask=None, src_key_padding_mask=key_padding_mask)

        # Perform sampling to generate captions
        if sampling_method == "nucleus":
            cap_tokens = self.nucleus_sampling(
                memory,
                prefix_len=prefix_txt_len,
                eos_id=eos_id,
                memory_key_padding_mask=key_padding_mask,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
        elif sampling_method == "greedy":
            cap_tokens = self.sampling(
                memory,
                prefix_len=prefix_txt_len,
                eos_id=eos_id,
                memory_key_padding_mask=key_padding_mask,
                mode="greedy",
            )
        else:
            cap_tokens = self.sampling(
                memory,
                prefix_len=prefix_txt_len,
                eos_id=eos_id,
                memory_key_padding_mask=key_padding_mask,
                mode="random",
            )
        return cap_tokens

    def core(self, target, memory, memory_key_padding_mask):
        """
        :param memory:   n b d
        :param memory_key_padding_mask:   b n
        :return:  logits for next token
        """
        out = self.decoder(
            target,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        # logits = self.to_logits(output)  # seq_len, batch,vocab_size
        return out[-1, :]  # [B, D]

    def get_logprobs_state(
        self, memory, target_embeddings, memory_key_padding_mask, output_logsoftmax=1
    ):
        output = self.core(
            target_embeddings, memory, memory_key_padding_mask=memory_key_padding_mask
        )
        if output_logsoftmax == 1:
            logprobs = F.log_softmax(self.to_logits(output), dim=1)
        else:
            logprobs = self.to_logits(output)
        return logprobs  # [B,Vocab]

    def sampling(
        self,
        memory,
        prefix_len,
        eos_id,
        memory_key_padding_mask,
        return_logprobs=False,
        mode="greedy",
    ):
        b, device = memory.shape[1], memory.device
        seq = torch.zeros(b, self.target_txt_len, dtype=torch.long).to(device)
        seqlogprobs = torch.zeros(b, self.target_txt_len, self.num_text_tokens).to(
            device
        )
        done = torch.tensor([False for _ in range(b)], device=device)  # [B]
        cap = torch.tensor([[]] * b, dtype=torch.long, device=device)  # [B 1]
        cap_embs = (
            torch.zeros(b, 1, self.dim_embed, device=device) + self.bos_emb
        )  # b 1 d
        cap_embs = rearrange(cap_embs, "b n c -> n b c")
        cur_len = 0
        trigrams = []
        max_pre_len = max(prefix_len)
        while cur_len < self.txt_seq_len - max_pre_len:
            logprobs = self.get_logprobs_state(
                memory,
                cap_embs,
                memory_key_padding_mask=memory_key_padding_mask,
                output_logsoftmax=1,
            )  # B V
            # Mess with trigrams
            # Copy from https://github.com/lukemelas/image-paragraph-captioning
            if cur_len >= 3:
                # Store trigram generated at last step
                prev_two_batch = cap[:, cur_len - 3 : cur_len - 1]
                for i in range(b):  # = seq.size(0)
                    prev_two = (
                        prev_two_batch[i][0].item(),
                        prev_two_batch[i][1].item(),
                    )
                    current = cap[i][cur_len - 1].item()
                    if cur_len == 3:  # initialize
                        trigrams.append(
                            {prev_two: [current]}
                        )  # {LongTensor: list containing 1 int}
                    elif cur_len > 3:
                        if prev_two in trigrams[i]:  # add to list
                            trigrams[i][prev_two].append(current)
                        else:  # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = cap[:, cur_len - 2 : cur_len]
                mask = torch.zeros(logprobs.size(), requires_grad=False).to(
                    logprobs.device
                )  # batch_size x vocab_size
                for i in range(b):
                    prev_two = (
                        prev_two_batch[i][0].item(),
                        prev_two_batch[i][1].item(),
                    )
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i, j] += 1
                # print(mask)
                # Apply mask to log probs
                # logprobs = logprobs - (mask * 1e9)
                alpha = 2.0  # = 4
                logprobs = logprobs + (
                    mask * -0.693 * alpha
                )  # ln(1/2) * alpha (alpha -> infty works best)
            # print(trigrams)
            seqlogprobs[:, cur_len] = logprobs
            if mode == "greedy":
                sample = torch.argmax(logprobs, dim=-1)  # [B]
            else:
                sample = torch.distributions.Categorical(
                    logits=logprobs.detach()
                ).sample()
            sample[done] = self.pad_idx
            is_done = sample == eos_id
            sample = sample.unsqueeze(-1)  # [B 1]
            cap = torch.cat((cap, sample), dim=-1)
            new_cap_embs = self.txt_embed(sample) + self.txt_pos_embed(
                prefix_len + cur_len
            )  # [B 1,D]
            new_cap_embs = rearrange(new_cap_embs, "b n c -> n b c")
            cap_embs = torch.cat([cap_embs, new_cap_embs], dim=0)
            done += is_done
            cur_len += 1
            all_done = False not in done
            if all_done:
                break
        seq[:, :cur_len] = cap[:, :]
        if return_logprobs:
            return seq, seqlogprobs
        else:
            return seq

    def nucleus_sampling(
        self,
        memory,
        prefix_len,
        eos_id,
        memory_key_padding_mask,
        top_k,
        top_p,
        temperature,
    ):
        """
        prefix_len: [B 1]
        """
        # logit
        b, device = memory.shape[1], memory.device
        seq = torch.zeros(b, self.target_txt_len, dtype=torch.long).to(device)
        done = torch.tensor([False for _ in range(b)], device=device)
        cap = torch.tensor([[]] * b, dtype=torch.long, device=device)  # [B 1]
        cap_embs = (
            torch.zeros(b, 1, self.dim_embed, device=device) + self.bos_emb
        )  # b 1 d
        cap_embs = rearrange(cap_embs, "b n c -> n b c")
        cur_len = 0
        max_pre_len = max(prefix_len)
        while cur_len < self.txt_seq_len - max_pre_len:
            logit = (
                self.get_logprobs_state(
                    memory,
                    cap_embs,
                    memory_key_padding_mask=memory_key_padding_mask,
                    output_logsoftmax=0,
                )
                / temperature
            )
            probs = self.top_k_top_p_filtering(
                logit, top_k=top_k, top_p=top_p, device=device
            )
            sample = torch.multinomial(probs, 1)[:, 0]  # [B 1]
            sample[done] = self.pad_idx
            is_done = sample == eos_id
            sample = sample.unsqueeze(-1)  # [B 1]
            cap = torch.cat((cap, sample), dim=-1)
            new_cap_embs = self.txt_embed(sample) + self.txt_pos_embed(
                prefix_len + cur_len
            )  # [B 1,D]
            new_cap_embs = rearrange(new_cap_embs, "b n c -> n b c")
            cap_embs = torch.cat([cap_embs, new_cap_embs], dim=0)
            done += is_done
            cur_len += 1
            all_done = False not in done
            if all_done:
                break
        seq[:, :cur_len] = cap[:, :]
        return seq

    def top_k_top_p_filtering(
        self,
        next_token_logits: torch.FloatTensor,
        top_k: Optional[float] = None,
        top_p: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.FloatTensor:
        if top_k is None:
            top_k = next_token_logits.shape[-1]
        if top_p is None:
            top_p = 1.0
        p, largest_p_idx = F.softmax(next_token_logits, dim=-1).topk(top_k, dim=-1)
        cumulative_p = p.cumsum(dim=-1)
        threshold_repeated = top_p + torch.zeros((len(p), 1)).to(device)
        idx = (
            torch.searchsorted(cumulative_p, threshold_repeated)
            .clip(max=top_k - 1)
            .squeeze()
        )
        cutoffs = cumulative_p[torch.arange(len(cumulative_p)), idx]
        censored_p = (cumulative_p <= cutoffs[:, None]) * p
        renormalized_p = censored_p / censored_p.sum(dim=-1, keepdims=True)
        final_p = torch.zeros_like(next_token_logits)
        row_idx = torch.arange(len(p)).unsqueeze(1).repeat(1, top_k).to(device)
        final_p[row_idx, largest_p_idx] = renormalized_p.to(final_p.dtype)
        return final_p
