import os
import torch
from torch import nn
from tqdm.auto import tqdm
from datetime import datetime
from torch.utils.data import Subset

from torch.utils.tensorboard import SummaryWriter


from .models.simclr import SimCLR
from .models.evaluate import EvaluateNet
from .models.modules.losses.nt_xent import NT_Xent
from .models.modules.transformations.simclr import SimCLRViewTransform


class Trainer:
    def __init__(
        self,
        method: str,
        backbone: nn.Module,
        feature_size: int,
        dataset: torch.utils.data.Dataset,
        image_size: int,
        save_dir: str,
        checkpoint_interval: int,
        reload_checkpoint: bool,
        **kwargs,
    ):
        """
        Description:
            Trainer class to train the model with self-supervised methods.

        Args:
            method (str): Self-supervised method to use. Options: [BarlowTwins, BYOL, DINO, MoCo, Rotation, SimCLR, SimSiam, SwAV, VICReg]
            backbone (nn.Module): Backbone to use.
            feature_size (int): Feature size.
            dataset (torch.utils.data.Dataset): Dataset to use.
            image_size (int): Image size.
            save_dir (str): Directory to save the model.
            checkpoint_interval (int): Interval to save the model.
            reload_checkpoint (bool): Whether to reload the checkpoint.
            **kwargs: Keyword arguments.
        """

        self.method = method
        self.dataset = dataset
        self.image_size = image_size
        self.backbone = backbone
        self.feature_size = feature_size
        self.reload_checkpoint = reload_checkpoint
        self.checkpoint_interval = checkpoint_interval

        self.save_dir = save_dir + f"/{self.method}/"

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.checkpoint_path = self.save_dir + "Pretext/"

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_workers = os.cpu_count()

        print("----------------AK_SSL----------------")
        print("Number of workers:", self.num_workers)
        print("Device:", self.device)
        print("--------------------------------------")

        match self.method:
            case "BarlowTwins":
                pass
            case "BYOL":
                pass
            case "DINO":
                pass
            case "MoCo":
                pass
            case "Rotation":
                pass
            case "SimCLR":
                self.model = SimCLR(self.backbone, self.feature_size, **kwargs)
                self.loss = NT_Xent(**kwargs)
                self.transformation = SimCLRViewTransform(
                    image_size=self.image_size, **kwargs
                )
                print("Method: SimCLR")
                print(f"Projection Dimension: {self.model.projection_dim}")
                print(
                    f"Projection number of layers: {self.model.projection_num_layers}"
                )
                print(
                    f"Projection batch normalization: {self.model.projection_batch_norm}"
                )
                print("Loss: NT_Xent Loss")
                print("Transformation: SimCLRViewTransform")
                print("--------------------------------------")
                print(self.dataset)
                print("--------------------------------------")
            case "SimSiam":
                pass
            case "SwAV":
                pass
            case "VICReg":
                pass
            case _:
                raise Exception("Method not found.")

        self.model = self.model.to(self.device)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter("{}/Logs/{}".format(self.save_dir, self.timestamp))

    def get_backbone(self):
        return self.model.backbone

    def train_one_epoch(self, tepoch, optimizer):
        loss_hist_train = 0.0
        for images, _ in tepoch:
            images = images.to(self.device)
            view0 = self.transformation(images)
            view1 = self.transformation(images)
            z0, z1 = self.model(view0, view1)

            optimizer.zero_grad()
            loss = self.loss(z0, z1)
            loss.backward()
            optimizer.step()
            loss_hist_train += loss.item()
            tepoch.set_postfix(loss=loss.item())

        return loss_hist_train

    def train(
        self,
        batch_size: int = 256,
        start_epoch: int = 1,
        epochs: int = 100,
        optimizer: str = "Adam",
        weight_decay: float = 1e-6,
        learning_rate: float = 1e-3,
    ):
        """
        Description:
            Train the model.

        Args:
            batch_size (int): Batch size.
            start_epoch (int): Epoch to start the training.
            epochs (int): Number of epochs.
            optimizer (str): Optimizer to train the model. Options: [Adam, SGD]
            weight_decay (float): Weight decay.
            learning_rate (float): Learning rate.
        """
        match optimizer:
            case "Adam":
                optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
            case "SGD":
                optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
                )

        train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        self.model.train(True)

        for epoch in tqdm(
            range(start_epoch - 1, epochs),
            unit="epoch",
            desc="Pretext Task Model Training",
            leave=True,
        ):
            with tqdm(train_loader, unit="batch", leave=False) as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                loss_per_epoch = self.train_one_epoch(tepoch, optimizer)

            self.writer.add_scalar(
                "Pretext Task/Loss/train",
                loss_per_epoch / len(train_loader),
                epoch + 1,
            )

            self.writer.flush()
            if (epoch + 1) % self.checkpoint_interval == 0:
                model_path = self.checkpoint_path + "SimCLR_model_{}_epoch{}".format(
                    self.timestamp, epoch + 1
                )
                torch.save(self.model.state_dict(), model_path)

        model_path = self.checkpoint_path + "SimCLR_model_{}_epoch{}".format(
            self.timestamp, epoch + 1
        )
        torch.save(self.model.state_dict(), model_path)

    def evaluate(
        self,
        dataset_train: torch.utils.data.Dataset,
        dataset_test: torch.utils.data.Dataset,
        eval_method: str = "linear",
        top_k: int = 1,
        epochs: int = 100,
        optimizer: str = "Adam",
        weight_decay: float = 1e-6,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        fine_tuning_data_proportion: float = 1,
    ):
        """
        Description:
            Evaluate the model using the given evaluating method.

        Args:
            eval_method (str): Evaluation method. Options: [linear, finetune]
            top_k (int): Top k accuracy.
            epochs (int): Number of epochs.
            optimizer (str): Optimizer to train the model. Options: [Adam, SGD]
            weight_decay (float): Weight decay.
            learning_rate (float): Learning rate.
            batch_size (int): Batch size.
            dataset_train (torch.utils.data.Dataset): Dataset to train the downstream model.
            dataset_test (torch.utils.data.Dataset): Dataset to test the downstream model.
            fine_tuning_data_proportion (float): Proportion of the dataset between 0 and 1 to use for fine-tuning.

        """
        match optimizer:
            case "Adam":
                optimizer_eval = torch.optim.Adam(
                    self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
            case "SGD":
                optimizer_eval = torch.optim.SGD(
                    self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
                )

        match eval_method:
            case "linear":
                net = EvaluateNet(
                    self.model.backbone,
                    self.feature_size,
                    len(dataset_train.classes),
                    True,
                )
            case "finetune":
                net = EvaluateNet(
                    self.model.backbone,
                    self.feature_size,
                    len(dataset_train.classes),
                    False,
                )

                num_samples = len(dataset_train)
                subset_size = int(num_samples * fine_tuning_data_proportion)

                indices = torch.randperm(num_samples)[:subset_size]

                dataset_train = Subset(dataset_train, indices)

        net = net.to(self.device)
        criterion = nn.CrossEntropyLoss()

        train_loader_ds = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True
        )

        net.train(True)

        for epoch in tqdm(
            range(epochs),
            unit="epoch",
            desc="Evaluate Model Training",
            leave=True,
        ):
            with tqdm(train_loader_ds, unit="batch", leave=False) as tepoch_ds:
                tepoch_ds.set_description(f"Epoch {epoch + 1}")
                loss_hist_train, acc_hist_train = 0.0, 0.0

                for images, labels in tepoch_ds:
                    correct, total = 0, 0

                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer_eval.zero_grad()
                    outputs = net(images)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    acc = 100 * correct / total
                    acc_hist_train += acc

                    # compute loss
                    loss = criterion(outputs, labels)
                    tepoch_ds.set_postfix(loss=loss.item(), accuracy=f"{acc:.2f}")
                    loss_hist_train += loss.item()
                    loss.backward()
                    optimizer_eval.step()

                self.writer.add_scalar(
                    "Downstream Task/Loss/train",
                    loss_hist_train / len(train_loader_ds),
                    epoch + 1,
                )

                self.writer.add_scalar(
                    "Downstream Task/Accuracy/train",
                    acc_hist_train / len(train_loader_ds),
                    epoch + 1,
                )

                self.writer.flush()

        test_loader_ds = torch.utils.data.DataLoader(
            dataset_test, batch_size=batch_size, shuffle=True
        )

        correct = 0
        total = 0

        net.eval()
        with torch.no_grad():
            for images, labels in tqdm(test_loader_ds, unit="batch"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                _, top = torch.topk(outputs.data, k=top_k, dim=1)
                correct_predictions = torch.eq(labels[:, None], top).any(dim=1)
                total += labels.size(0)
                correct += correct_predictions.sum().item()

        print(
            f"The top_{top_k} accuracy of the network on the {len(dataset_test)} test images: {(100 * correct / total)}%"
        )

        self.writer.close()

    def load_checkpoint(self, optimizer: nn.Module):
        """ """
        pass

    def save_checkpoint(
        self, epoch: int, model: nn.Module, optimizer: nn.Module, loss: float
    ):
        pass