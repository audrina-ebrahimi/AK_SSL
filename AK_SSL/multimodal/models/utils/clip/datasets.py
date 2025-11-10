from typing import Union

import torch
import torchvision.transforms as transforms
import transformers
from PIL import Image


def get_image_transform(
    image_size: tuple[int, int] = (224, 224), mode: str = "train"
) -> transforms.Compose:
    """
    Returns a composition of image transformations to be applied.

    Args:
        image_size (tuple[int, int]): Desired image size after transformation.
        mode (str): Mode of the transformation, either "train" or "test".

    Returns:
        transforms.Compose: A composition of image transformations.
    """
    if mode == "train":
        return transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(image_size),
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(image_size),
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )


class CustomClipDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for loading images and captions for CLIP.

    Args:
        images (list[str]): List of paths to image files.
        captions (list[str]): List of captions corresponding to images. if there are multiple captions for an image, if there are multiple captions for each image, the images must have repetitive file paths.
        tokenizer (Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]): Tokenizer to encode captions.
        image_transform (transforms.Compose): Transformations to apply to images.
    """

    def __init__(
        self,
        images: list[str],
        captions: list[str],
        tokenizer: Union[
            transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast
        ],
        image_transform: transforms.Compose,
    ) -> None:

        assert len(images) == len(
            captions
        ), "Number of images and captions should be same"

        self.captions = list(captions)
        self.images = images
        self.image_transform = image_transform

        self.encoded_captions = tokenizer(
            captions,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image = Image.open(self.images[idx])
        image = self.image_transform(image)
        items = {key: val[idx] for key, val in self.encoded_captions.items()}
        items["image"] = image
        items["caption"] = self.captions[idx]
        return items
