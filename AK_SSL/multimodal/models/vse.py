import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class VSE(nn.Module):
    """
    VSE: Improving Visual-Semantic Embedding with Adaptive Pooling and Optimization Objective
    Paper Link: https://arxiv.org/abs/2210.02206v1
    Implementation Link: https://github.com/96-Zachary/vse_2ad
    """

    def __init__(
        self, image_encoder: nn.Module, text_encoder: nn.Module, margin: float = 0.2
    ):
        """
        Initializes the VSE model.

        Args:
            image_encoder (nn.Module): The model to encode images.
            text_encoder (nn.Module): The model to encode text.
            margin (float): The margin used in loss functions (default: 0.2).
        """
        super(VSE, self).__init__()
        self.margin = margin
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.txt_enc_params = list(self.text_encoder.parameters())
        self.img_enc_params = list(self.image_encoder.parameters())
        self.enc_params = self.img_enc_params + self.txt_enc_params

    def forward(
        self,
        image: torch.Tensor,
        image_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        # img = [batch_size, n_region, img_dim]
        # txt = [batch_size, seq_len]
        # img_emb = [batch_size, n_region, emb_size]
        # txt_emb  = [batch_size, seq_len, emb_size]
        # Encode images and texts
        img_emb = self.image_encoder(image, image_lengths)
        txt_emb, txt_lens = self.text_encoder(text, text_lengths)

        return img_emb, txt_emb, txt_lens

    def conterastive_loss(
        self, img: torch.Tensor, txt: torch.Tensor, txt_lengths: torch.Tensor
    ):
        contrastive_loss = ContrastiveLoss(
            scale=1, const_num=1, margin=self.margin, device=txt.device
        )
        return contrastive_loss(img, txt, txt_lengths)

    def triplet_loss(
        self, img: torch.Tensor, txt: torch.Tensor, txt_lengths: torch.Tensor
    ):
        triplet_loss = TripletLoss(margin=self.margin, max_violation=True)
        return triplet_loss(img, txt, txt_lengths)


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        scale: float,
        const_num: int,
        margin: float,
        device: str,
        lamda: float = 0.01,
        acc_mode: str = "acc",
    ):
        """
        Initializes the contrastive loss function.

        Args:
            scale (float): Scaling factor for loss calculation.
            const_num (int): Constant number of negatives.
            margin (float): Margin for the loss.
            device (str): Device to run the computations.
            lamda (float): Regularization term (default: 0.01).
            acc_mode (str): Mode to determine number of negatives (default: "acc").
        """
        super(ContrastiveLoss, self).__init__()
        self.device = device
        self.lamda = lamda
        self.acc_mode = acc_mode
        self.scale = scale
        self.margin = margin
        self.const_num = const_num
        self.loss_func = F.cross_entropy

    def align(self, d: torch.Tensor):
        return d.diag().mean()

    def uniform(self, d: torch.Tensor):
        d = d - torch.eye(d.shape[0]).to(d.device) * d
        return d.mul(-1).exp().mean()

    def ADCTO(
        self,
        d: torch.Tensor,
        query: torch.Tensor,
        keys: torch.Tensor,
        acc_mode: str,
        scale: float,
        num_negs: int = None,
    ):
        """
        d: [batch_size, batch_size] <--> query x keys
        querys: [batch_size, emb_size]
        keys: [batch_size, emb_size]
        """
        # positive_logit = [batch_size, 1]
        positive_logit = torch.diag(d).unsqueeze(-1)
        if acc_mode == "acc":
            align = self.align(d).detach().data.cpu().numpy()
            uniform = self.uniform(d).detach().data.cpu().numpy()
            tmp = align + uniform
            num_negs = max(int(np.cos(tmp**scale) * d.shape[0]) + 1, 1)
            num_negs = min(num_negs, d.shape[0] - 1)
        elif acc_mode == "constant":
            num_negs = 1 if not num_negs else num_negs
        elif acc_mode == "random":
            num_negs = np.random.randint(1, d.shape[0])

        # mask = [batch_size, batch_size]
        mask = (torch.eye(d.size(0)) > 0.5).to(d.device)
        d = d.masked_fill(mask, 0)

        # sorted_idx = [batch_size, num_negs, emb_size]
        _, sorted_idx = torch.sort(d, dim=-1, descending=True)
        sorted_idx = sorted_idx[:, :num_negs]
        sorted_idx = sorted_idx.unsqueeze(-1).repeat(1, 1, keys.shape[-1])

        # negative_keys = [batch_size, num_negs, emb_size]
        negative_keys = torch.gather(
            keys.repeat(d.shape[0], 1).view(d.shape[0], d.shape[0], -1), 1, sorted_idx
        )

        # negative_logits = [batch_size, num_negs]
        negative_logits = (
            torch.matmul(query.unsqueeze(1), negative_keys.transpose(-2, -1)).squeeze(1)
            + self.margin
        )

        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=d.device)

        return self.loss_func(logits / self.lamda, labels), num_negs

    def forward(self, img: torch.Tensor, txt: torch.Tensor, txt_lens: torch.Tensor):
        # cos_sim = [batch_size, batch_size]
        cos_sim = txt.mm(img.t())
        t2i_loss, t2i_num_negs = self.ADCTO(
            cos_sim, txt, img, self.acc_mode, self.scale, self.const_num
        )
        i2t_loss, i2t_num_negs = self.ADCTO(
            cos_sim.t(), img, txt, self.acc_mode, self.scale, self.const_num
        )
        return (t2i_loss + i2t_loss) / 2, [t2i_num_negs, i2t_num_negs]


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0, max_violation: bool = False):
        """
        Initializes the triplet loss function.

        Args:
            margin (float): Margin for the loss.
            max_violation (bool): If true, use the maximum violating negative for each query (default: False).
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, img: torch.Tensor, txt: torch.Tensor, txt_lens: torch.Tensor):
        scores = img.mm(txt.t())

        # scores = [batch_size, batch_size]
        # diagonal = [batch_size, 1]
        diagonal = scores.diag().view(img.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)
        mask = torch.eye(scores.size(0)) > 0.5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum(), [1, 1]
