import os
import re
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_

from AK_SSL.multimodal.models import *
from AK_SSL.vision.models.modules.losses.nt_xent import NT_Xent


class Trainer:
    def __init__(
        self,
        method: str,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        tokenizer,
        mixed_precision_training: bool = True,
        save_dir: str = ".",
        checkpoint_interval: int = 10,
        reload_checkpoint: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> None:

        self.method = method
        self.checkpoint_interval = checkpoint_interval
        self.reload_checkpoint = reload_checkpoint
        self.verbose = verbose
        self.tokenizer = tokenizer
        self.mixed_precision_training = mixed_precision_training

        self.save_dir = save_dir + f"/{self.method}/"

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_workers = os.cpu_count()

        if self.verbose:
            print("----------------AK_SSL: Multimodal----------------")
            print("Number of workers:", self.num_workers)
            print("Device:", self.device)
            print("--------------------------------------")
            print(f"Method: {self.method}")

        match self.method.lower():
            case "clip":
                self.model = CLIP(
                    image_encoder=image_encoder, text_model=text_encoder, **kwargs
                )
                if self.verbose:
                    print("Embedding Dimension:", self.model.embed_dim)
                    print(
                        "Dimension of the image features:", self.model.image_feature_dim
                    )
                    print(
                        "Dimension of the text features:", self.model.text_feature_dim
                    )
                    print(f"Loss Function:{"SigLIP loss" if self.model.use_siglip else "Contrastive loss"}")
            case "albef":
                self.model = ALBEF(
                    vision_model=image_encoder, text_model=text_encoder, **kwargs
                )
            case "simvlm":
                self.model = SimVLM(
                    vision_model=image_encoder, text_model=text_encoder, **kwargs
                )
            case "slip":
                self.model = SLIP(
                    vision_model=image_encoder, text_model=text_encoder, **kwargs
                )
            case "uniter":
                self.model = UNITER(
                    vision_model=image_encoder, text_model=text_encoder, **kwargs
                )
            case "vse":
                self.model = VSE(
                    vision_model=image_encoder, text_model=text_encoder, **kwargs
                )
            case _:
                raise ValueError(f"Method {self.method} not supported")

        self.model = self.model.to(self.device)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter("{}/Logs/{}".format(self.save_dir, self.timestamp))

    def __del__(self):
        self.writer.close()
        
    def _train_clip(self, train_loader, optimizer, scaler):
        epoch_loss = 0.0
        for step, (batch) in enumerate(train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'image']}

            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                logits = self.model(**batch)
                if self.model.use_siglip:
                    loss = self.model.criterion_siglip_loss(logits)
                else:
                    loss = self.model.criterion_contrastive_loss(logits)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            train_loader.set_postfix(loss=loss.item(), temp=self.model.t_prime.exp().item(), bias=self.model.b.item(), lr=optimizer.param_groups[0]['lr'])

        return epoch_loss

    def _train_slip(self, train_loader, optimizer, scaler):
        epoch_loss = 0.0
        for step, (batch) in enumerate(train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'image']}

            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                logits = self.model(**batch)
                ssl_loss = NT_Xent(temperature=0.1)
                ssl_loss = ssl_loss(logits['aug1_embed'], logits['aug2_embed'])
                clip_loss = self.model.clip.criterion_contrastive_loss(logits['clip_output'])

                loss = self.model.criterion(ssl_scale=1.0, ssl_loss=ssl_loss, clip_loss=clip_loss)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            train_loader.set_postfix(loss=loss.item(), temp=self.model.t_prime.exp().item(), bias=self.model.b.item(), lr=optimizer.param_groups[0]['lr'])

        return epoch_loss

    def _train_simvlm(self, train_loader, optimizer, scaler):
        epoch_loss = 0.0
        for step, (batch) in enumerate(train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items() if k in ['text', 'image']}

            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                logits, labels = self.model(**batch)
                loss = self.model.criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            train_loader.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        return epoch_loss
    
    def _train_vse(self, train_loader, optimizer, scaler):
        epoch_loss = 0.0
        num_negs = []
        for step, (batch) in enumerate(train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items() if k in ['image', 'image_lengths', 'text', 'text_lengths']}

            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                img_emb, txt_emb, txt_lens = self.model(**batch)
                loss, tmp_num_negs = self.model.conterastive_loss(img_emb, txt_emb, txt_lens)
                num_negs.extend(tmp_num_negs)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            clip_grad_norm_(self.model.enc_params, 2.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            train_loader.set_postfix(loss=loss.item(), epoch_negs=np.mean(num_negs))

        return epoch_loss

    def _train__train_unitervqa(self, train_loader, optimizer, scaler):
        epoch_loss = 0.0
        for step, (batch) in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
                logits = self.model(**batch)
                loss = self.model.criterion(batch['targets'], logits)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            train_loader.set_postfix(loss=loss.item(), temp=self.model.t_prime.exp().item(), bias=self.model.b.item(), lr=optimizer.param_groups[0]['lr'])

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        batch_size: int = 256,
        start_epoch: int = 1,
        epochs: int = 100,
        optimizer: str = "Adam",
        weight_decay: float = 1e-6,
        learning_rate: float = 1e-3,
    ):

        number_of_epochs = epochs - start_epoch + 1

        match optimizer.lower():
            case "adam":
                optimizer = torch.optim.Adam(
                    list(self.model.parameters()),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
            case "adamw":
                optimizer = torch.optim.AdamW(
                    list(self.model.parameters()),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
            case "sgd":
                optimizer = torch.optim.SGD(
                    list(self.model.parameters()),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
            case _:
                raise ValueError(f"Optimizer {optimizer} not supported")
    
        if self.reload_checkpoint:
            start_epoch = self._reload_latest_checkpoint() + 1
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision_training)
        self.model.train()

        match self.method.lower():
            case "clip":
                tmax = number_of_epochs * len(train_loader) + len(train_loader) // 4
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=tmax, eta_min=1e-8)

                for epoch in tqdm(range(start_epoch - 1, epochs), unit="epoch", desc="CLIP Training", leave=True,):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_clip(train_loader, optimizer, scaler)
                        lr_scheduler.step()

                    self.writer.add_scalar(f"{self.method.upper()}/Train/Loss", loss_per_epoch / len(train_loader), epoch + 1)
                    self.writer.flush()
                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.checkpoint_path + "{}_model_{}_epoch{}.pth".format(
                            self.method, self.timestamp, epoch + 1
                        )
                        torch.save(self.model.state_dict(), model_path)

            case "slip":
                tmax = number_of_epochs * len(train_loader) + len(train_loader) // 4
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=tmax, eta_min=1e-5)

                for epoch in tqdm(range(start_epoch - 1, epochs), unit="epoch", desc="SLIP Training", leave=True,):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_slip(train_loader, optimizer, scaler)
                        lr_scheduler.step()

                    self.writer.add_scalar(f"{self.method.upper()}/Train/Loss", loss_per_epoch / len(train_loader), epoch + 1)
                    self.writer.flush()
                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.checkpoint_path + "{}_model_{}_epoch{}.pth".format(
                            self.method, self.timestamp, epoch + 1
                        )
                        torch.save(self.model.state_dict(), model_path)
            
            case "albef":
                raise NotImplementedError("Training for ALBEF is not implemented yet.")
            
            case "simvlm":
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2000)

                for epoch in tqdm(range(start_epoch - 1, epochs), unit="epoch", desc="SimVLM Training", leave=True,):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_simvlm(train_loader, optimizer, scaler)
                        lr_scheduler.step()

                    self.writer.add_scalar(f"{self.method.upper()}/Train/Loss", loss_per_epoch / len(train_loader), epoch + 1)
                    self.writer.flush()
                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.checkpoint_path + "{}_model_{}_epoch{}.pth".format(
                            self.method, self.timestamp, epoch + 1
                        )
                        torch.save(self.model.state_dict(), model_path)
                        
            case "uniter_vqa":
                for epoch in tqdm(range(start_epoch - 1, epochs), unit="epoch", desc="Uniter For VQA Training", leave=True,):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_unitervqa(train_loader, optimizer, scaler)

                    self.writer.add_scalar(f"{self.method.upper()}/Train/Loss", loss_per_epoch / len(train_loader), epoch + 1)
                    self.writer.flush()
                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.checkpoint_path + "{}_model_{}_epoch{}.pth".format(
                            self.method, self.timestamp, epoch + 1
                        )
                        torch.save(self.model.state_dict(), model_path)
            
            case "vse":
                for epoch in tqdm(range(start_epoch - 1, epochs), unit="epoch", desc="VSE Training", leave=True,):
                    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        loss_per_epoch = self._train_vse(train_loader, optimizer, scaler)

                    self.writer.add_scalar(f"{self.method.upper()}/Train/Loss", loss_per_epoch / len(train_loader), epoch + 1)
                    self.writer.flush()
                    if (epoch + 1) % self.checkpoint_interval == 0:
                        model_path = self.checkpoint_path + "{}_model_{}_epoch{}.pth".format(
                            self.method, self.timestamp, epoch + 1
                        )
                        torch.save(self.model.state_dict(), model_path)

            case _:
                raise ValueError(f"Method {self.method} not supported")
        
        model_path = self.checkpoint_path + "{}_model_{}_epoch{}.pth".format(
            self.method, self.timestamp, epoch + 1
        )
        torch.save(self.model.state_dict(), model_path)
        
    def load_checkpoint(self, checkpont_dir: str):
        self.model.load_state_dict(torch.load(checkpont_dir))
        if self.verbose:
            print("Checkpoint loaded.")
    
    def _reload_latest_checkpoint(self):
        checkpoints = os.listdir(self.checkpoint_path)
        sorted_checkpoints = sorted(
            [os.path.join(self.checkpoint_path, i) for i in checkpoints],
            key=os.path.getmtime,
        )

        if len(sorted_checkpoints) == 0:
            raise ("No checkpoints found.")

        self.load_checkpoint(sorted_checkpoints[-1])

        match = re.search(r"epoch(\d+)", sorted_checkpoints[-1])
        if match:
            epoch = int(match.group(1))
            if self.verbose:
                print(f"Starting Epoch: {epoch}")
        else:
            raise ValueError("No epoch number found in the checkpoint name.")

        return epoch
