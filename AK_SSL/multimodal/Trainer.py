import os
import torch
import torch.nn as nn

from AK_SSL.multimodal.models import *


class Trainer:
    def __init__(
        self,
        method: str,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
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
                    vision_model=image_encoder, text_model=text_encoder, **kwargs
                )
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
