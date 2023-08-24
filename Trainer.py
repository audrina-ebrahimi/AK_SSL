import os
import torch
from torch import nn


class Trainer:
    def __init__(
        self,
        method: str,
        backbone: nn.Module,
        dataset_dir: str,
        dataset: torch.utils.data.Dataset,
        save_dir: str,
        checkpoint_interval: int,
        reload_checkpoint: bool,
    ):
        """
        Description:
            Trainer class to train the model with self-supervised methods.

        Args:
            method (str): Method to train the model. Options: [BarlowTwins, BYOL, DINO, MoCo, Rotation, SimCLR, SimSiam, SwAV, VICReg]
            backbone (nn.Module): Backbone model to train.
            dataset_dir (str): Path to the dataset directory.
            dataset (torch.utils.data.Dataset): Dataset to train the model.
            save_dir (str): Path to the directory where the model will be saved.
            checkpoint_interval (int): Interval to save the model.
            reload_checkpoint (bool): If True, the model will be loaded from the last checkpoint.

        """

        self.method = method
        self.dataset = dataset
        self.backbone = backbone
        self.dataset_dir = dataset_dir
        self.reload_checkpoint = reload_checkpoint
        self.checkpoint_interval = checkpoint_interval

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        self.save_dir = save_dir + f"/{self.method}/"

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_workers = os.cpu_count()

        print("----------------AK_SSL----------------")
        print("Number of workers:", self.num_workers)
        print("Device:", device)
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
                pass
            case "SimSiam":
                pass
            case "SwAV":
                pass
            case "VICReg":
                pass
            case _:
                raise Exception("Method not found.")

    def get_backbone(self):
        return self.model.backbone

    def train(
        batch_size: int,
        start_epoch: int,
        epochs: int,
        optimizer: str,
        weight_decay: float,
        learning_rate: float,
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
                optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            case "SGD":
                optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def evaluate(
        self,
        eval_method: str,
        top_k: int,
        epochs: int,
        optimizer: str,
        weight_decay: float,
        learning_rate: float,
        batch_size: int,
        dataset_train: torch.utils.data.Dataset,
        dataset_test: torch.utils.data.Dataset,
        fine_tuning_data_proportion: float
    ):
        """
        Description:
            Evaluate the model using the given evaluating method.

        Args:
            eval_method (str): Evaluation method. Options: [inear, finetune]
            top_k (int): Top k accuracy.
            epochs (int): Number of epochs.
            optimizer (str): Optimizer to train the model. Options: [Adam, SGD]
            weight_decay (float): Weight decay.
            learning_rate (float): Learning rate.
            batch_size (int): Batch size.
            dataset_train (torch.utils.data.Dataset): Dataset to train the downstream model.
            dataset_test (torch.utils.data.Dataset): Dataset to test the downstream model.
            fine_tuning_data_proportion (float): Proportion of the dataset to use for fine-tuning.

        """
        match optimizer:
            case "Adam":
                optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            case "SGD":
                optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def load_checkpoint(self, optimizer: nn.Module):
        """ """
        pass

    def save_checkpoint(
        self, epoch: int, model: nn.Module, optimizer: nn.Module, loss: float
    ):
        pass
