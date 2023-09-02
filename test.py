from AK_SSL import Trainer
import torch
import torchvision

train_dataset = torchvision.datasets.CIFAR10(
    root="../datasets/" + "cifar10",
    download=True,
    transform=torchvision.transforms.ToTensor(),
)


backbone = torchvision.models.resnet18(weights=None, zero_init_residual=True)
feature_size = backbone.fc.in_features
backbone.fc = torch.nn.Identity()

trainer = Trainer(
    method="dino",
    backbone=backbone,
    feature_size=feature_size,
    dataset=train_dataset,
    image_size=32,
    save_dir="../temp_save/",
    checkpoint_interval=10,
    reload_checkpoint=False,
)

trainer.train(
    batch_size=256,
    start_epoch=1,
    epochs=100,
    optimizer="Adam",
    weight_decay=1e-6,
    learning_rate=1e-3,
)
