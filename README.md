<p align="center">
  <img src="https://raw.githubusercontent.com/audrina-ebrahimi/AK_SSL/main/Documents/logo.png" alt="AK_SSL Logo"  width="50%"/>
</p>

<h1>
<br>AK_SSL: A Self-Supervised Learning Library
</h1>



![GitHub](https://img.shields.io/github/license/audrina-ebrahimi/AK_SSL) ![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg) ![PyPI - Version](https://img.shields.io/pypi/v/AK_SSL)


---

## 📒 Table of Contents
- [📒 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [✍️ Self Supervised Learning](#-self-supervised-learning)
- [🔎 Supported Methods](#-supported-methods)
- [📦 Installation](#-installation)
- [💡 Tutorial](#-tutorial)
- [📊 Benchmarks](#-benchmarks)
- [📜 References Used](#-reference-used)
- [💯 License](#-license)
- [🤝 Collaborators](#-collaborators)


---
## 📍 Overview
Welcome to the Self-Supervised Learning Library! This repository hosts a collection of tools and implementations for self-supervised learning. Self-supervised learning is a powerful paradigm that leverages unlabeled data to pre-trained models, which can then be fine-tuned on specific tasks with smaller labeled datasets. This library aims to provide researchers and practitioners with a comprehensive set of tools to experiment, learn, and apply self-supervised learning techniques effectively.
This project was our assignment during the summer apprenticeship in the newly established Intelligent and Learning System ([ILS](http://ils.ui.ac.ir/)) laboratory at the University of Isfahan.

---

## ✍️ Self Supervised Learning

Self-supervised learning is a subfield of machine learning where models are trained to predict certain aspects of the input data without relying on manual labeling. This approach has gained significant attention due to its ability to leverage large amounts of unlabeled data, which is often easier to obtain than fully annotated datasets. This library provides implementations of various self-supervised techniques, allowing you to experiment with and apply these methods in your own projects.

---

## 🔎 Supported Methods

### [BarlowTwins](./AK_SSL/models/barlowtwins.py)
Barlow Twins is a self-supervised learning method that aims to learn embeddings invariant to distortions of the input sample. It achieves this by applying two distinct sets of augmentations to the same input sample, resulting in two distorted views of the same image. The objective function measures the cross-correlation matrix between the outputs of two identical networks fed with these distorted sample versions, striving to make it as close to the identity matrix as possible. This causes the embedding vectors of the distorted sample versions to become similar while minimizing redundancy among the components of these vectors. Barlow Twins particularly benefits from utilizing high-dimensional output vectors.

<details><summary>Details of this method</summary>

  | Loss         | Transformation     | Transformation Prime | Projection Head         | Paper     | Original Code |
  |--------------|--------------------|----------------------|-------------------------|-----------|---------------|
  |[BarlowTwins Loss](./AK_SSL/models/modules/losses/barlow_twins_loss.py)|[SimCLR Transformation](./AK_SSL/models/modules/transformations/simclr.py)|[SimCLR Transformation](./AK_SSL/models/modules/transformations/simclr.py)|[BarlowTwins Projection Head](./AK_SSL/models/modules/heads.py)|[Link](https://arxiv.org/abs/2103.03230v3)|[Link](https://github.com/facebookresearch/barlowtwins)|
  
  
  BarlowTwins Loss is inspired by [HSIC loss](https://github.com/yaohungt/Barlow-Twins-HSIC).

</details>

### [BYOL](./AK_SSL/models/byol.py)
BYOL (Bootstrap Your Own Latent) is one of the new approaches to self-supervised learning. Like other methods, BYOL aims to learn a representation that can be utilized for downstream tasks. It employs two neural networks for learning: the online and target networks. The online network is trained to predict the target network's representation of the same image from a different augmented view. Simultaneously, the target network is updated with a slow-moving average of the online network's parameters. While state-of-the-art methods rely on negative pairs, BYOL achieves a new state of the art without them. It directly minimizes the similarity between the representations of the same image from different augmented views (positive pair).

<details><summary>Details of this method</summary>

  | Loss         | Transformation     | Transformation Prime | Projection Head         | Prediction Head         | Paper     | Original Code |
  |--------------|--------------------|----------------------|-------------------------|-------------------------|-----------|---------------|
  |[BYOL Loss](./AK_SSL/models/modules/losses/byol_loss.py)|[SimCLR Transformation](./AK_SSL/models/modules/transformations/simclr.py)|[SimCLR Transformation](./AK_SSL/models/modules/transformations/simclr.py)|[BarlowTwins Projection Head](./AK_SSL/models/modules/heads.py)|[BarlowTwins Prediction Head](./AK_SSL/models/modules/heads.py)|[Link](https://arxiv.org/abs/2006.07733)|[Link](https://github.com/deepmind/deepmind-research/tree/master/byol)|
  

</details>

### [DINO](./AK_SSL/models/dino.py)
DINO (self-distillation with no labels) is a self-supervised learning method that directly predicts the output of a teacher network—constructed with a momentum encoder—by utilizing a standard cross-entropy loss. It is an innovative self-supervised learning algorithm developed by Facebook AI. Through the utilization of self-supervised learning with Transformers, DINO paves the way for creating machines that can comprehend images and videos at a much deeper level.

<details><summary>Details of this method</summary>

  | Loss         | Transformation Global 1 | Transformation Global 2 |Transformation Local | Projection Head    | Paper  | Original Code |
  |--------------|-------------------------|-------------------------|---------------------|--------------------|--------|---------------|
  |[DINO Loss](./AK_SSL/models/modules/losses/dino_loss.py)|[SimCLR Transformation](./AK_SSL/models/modules/transformations/simclr.py)|[SimCLR Transformation](./AK_SSL/models/modules/transformations/simclr.py)|[SimCLR Transformation](./AK_SSL/models/modules/transformations/simclr.py)|[DINO Projection Head](./AK_SSL/models/modules/heads.py)|[Link](https://arxiv.org/abs/2304.07193)|[Link](https://github.com/facebookresearch/dinov2)|
  
</details>

### [MoCos](./AK_SSL/models/moco.py)
MoCo, short for Momentum Contrast, is a self-supervised learning algorithm that employs a contrastive loss. MoCo v2 represents an enhanced iteration of the original Momentum Contrast self-supervised learning algorithm. Motivated by the findings outlined in the SimCLR paper, the authors introduced several modifications in MoCo v1, which included replacing the 1-layer fully connected layer with a 2-layer MLP head featuring ReLU activation for the unsupervised training stage. Additionally, they incorporated blur augmentation and adopted a cosine learning rate schedule. These adjustments enabled MoCo to outperform the state-of-the-art SimCLR, even when utilizing a smaller batch size and fewer epochs.

MoCo v3, introduced in the paper "An Empirical Study of Training Self-Supervised Vision Transformers," represents another advancement in self-supervised learning. It builds upon the foundation of MoCo v1 / MoCo v2 and addresses the instability issue observed when employing ViT for self-supervised learning.

In contrast to MoCo v2, MoCo v3 adopts a different approach where the keys naturally coexist within the same batch. The memory queue (memory bank) is discarded, resulting in a setting similar to that of SimCLR. The encoder fq comprises a backbone (e.g., ResNet, ViT), a projection head, and an additional prediction head.

<details><summary>Details of this method</summary>

  | Method | Loss         | Transformation |Transformation Prime | Projection Head    | Prediction Head    | Paper  | Original Code |
  |--------|--------------|----------------|---------------------|--------------------|--------------------|--------|---------------|
  |MoCo v2|InfoNCE|[SimCLR Transformation](./AK_SSL/models/modules/transformations/simclr.py)|None|[SimCLR Projection Head](./AK_SSL/models/modules/heads.py)|None|[Link](https://arxiv.org/abs/2003.04297)|[Link](https://github.com/facebookresearch/moco)|
  |MoCo v3|[InfoNCE](./AK_SSL/models/modules/losses/info_nce.py)|[SimCLR Transformation](./AK_SSL/models/modules/transformations/simclr.py)|[SimCLR Transformation](./AK_SSL/models/modules/transformations/simclr.py)|[SimCLR Projection Head](./AK_SSL/models/modules/heads.py)|[BYOL Prediction Head](./AK_SSL/models/modules/heads.py)|[Link](https://arxiv.org/abs/2104.02057)|[Link](https://github.com/facebookresearch/moco-v3)|
  
</details>

### [SimCLRs](./AK_SSL/models/simclr.py)
SimCLR (Simple Framework for Contrastive Learning of Representations) is a self-supervised technique used to learn image representations. The fundamental building blocks of contrastive self-supervised methods, such as SimCLR, are image transformations. Each image is transformed into multiple new images through randomly applied augmentations. The goal of the self-supervised model is to identify images that originate from the same original source among a set of negative examples. SimCLR operates on the principle of maximizing the similarity between positive pairs of augmented images while minimizing the similarity with negative pairs. The training process can be summarized as follows: Data Augmentation - SimCLR employs robust data augmentation techniques to generate multiple augmented versions of each input image.

<details><summary>Details of this method</summary>

  | Loss         | Transformation     | Projection Head         | Paper     | Original Code |
  |--------------|--------------------|-------------------------|-----------|---------------|
  |[NT_Xent](./AK_SSL/models/modules/losses/nt_xent.py)|[SimCLR Transformation](./AK_SSL/models/modules/transformations/simclr.py)|[SimCLR Projection Head](./AK_SSL/models/modules/heads.py)|[Link](https://arxiv.org/abs/2006.10029)|[Link](https://github.com/google-research/simclr)|

  
</details>

### [SimSiam](./AK_SSL/models/simsiam.py)
SimSiam is a self-supervised representation learning model that was proposed by Facebook AI Research (FAIR). It is a simple Siamese network designed to learn meaningful representations without requiring negative sample pairs, large batches, or momentum encoders.

<details><summary>Details of this method</summary>

  | Loss         | Transformation     | Projection Head         | Prediction Head         | Paper     | Original Code |
  |--------------|--------------------|-------------------------|-------------------------|-----------|---------------|
  |[Negative Cosine Similarity](./AK_SSL/models/modules/losses/negative_cosine_similarity.py)|[SimCLR Transformation](./AK_SSL/models/modules/transformations/simclr.py)|[SimSiam Projection Head](./AK_SSL/models/modules/heads.py)|[SimSiam Prediction Head](./AK_SSL/models/modules/heads.py)|[Link](https://arxiv.org/abs/2011.10566)|[Link](https://github.com/facebookresearch/simsiam)|
  

</details>

### [SwAV](./AK_SSL/models/swav.py)
SwAV, or Swapping Assignments Between Views, is a self-supervised learning approach that takes advantage of contrastive methods without requiring to compute pairwise comparisons. Specifically, it simultaneously clusters the data while enforcing consistency between cluster assignments produced for different augmentations (or views) of the same image, instead of comparing features directly as in contrastive learning. Simply put, SwAV uses a swapped prediction mechanism where we predict the cluster assignment of a view from the representation of another view.

<details><summary>Details of this method</summary>

  | Loss         | Transformation Global| Transformation Local | Projection Head         | Paper     | Original Code |
  |--------------|----------------------|----------------------|-------------------------|-----------|---------------|
  |[SwAV Loss](./AK_SSL/models/modules/losses/swav.py)|[SimCLR Transformation](./AK_SSL/models/modules/transformations/simclr.py)|[SimCLR Transformation](./AK_SSL/models/modules/transformations/simclr.py)|[SwAV Projection Head](./AK_SSL/models/modules/heads.py)|[Link](https://arxiv.org/abs/2006.09882)|[Link](https://github.com/facebookresearch/swav)|
  

</details>

---

## 📦 Installation

You can install AK_SSL and its dependencies from PyPI with:


```sh
pip install AK-SSL
```

We strongly recommend that you install AK_SSL in a dedicated virtualenv, to avoid conflicting with your system packages

---

## 💡 Tutorial

Using AK_SSL, you have the flexibility to leverage the most recent self-supervised learning techniques seamlessly, harnessing the complete capabilities of PyTorch. You can explore diverse backbones, models, and optimizer while benefiting from a user-friendly framework that has been purposefully crafted for ease of use.

You can easily import Trainer module from AK_SSL library and start utilizing it right away.

```python
from AK_SSL import Trainer
```

### Initializing the Trainer
Now, let's initialize the self-supervised trainer with our chosen method, backbone, dataset, and other configurations.

```python
trainer = Trainer(
    method="barlowtwins",           # training method as string
    backbone=backbone,              # backbone architecture as torch.Module
    feature_size=feature_size,      # size of the extracted features as integer
    dataset=train_dataset,          # training dataset as torch.utils.data.Dataset
    image_size=32,                  # dataset image size as integer
    save_dir="./save_for_report/",  # directory to save training checkpoints and Tensorboard logs as string
    checkpoint_interval=50,         # interval (in epochs) for saving checkpoints as integer
    reload_checkpoint=False,        # reload a previously saved checkpoint as boolean
    verbose=True,                   # enable verbose output for training progress as a boolean
    **kwargs                        # other arguments 
)
```
Note: The use of **kwargs can differ between methods, depending on the specific method, loss function, transformation, and other factors. If you are utilizing any of the objectives listed below, you must provide their arguments during the initialization of the Trainer class.

- <details><summary>SimCLR Transformation</summary>
  
  ```
    color_jitter_strength     # a float to Set the strength of color
    use_blur                  # a boolean to specify whether to apply blur augmentation
    mean                      # a float to specify the mean values for each channel
    std                       # a float to specify the standard deviation values for each channel
  ```
  
  </details>

- <details><summary>BarlowTwins</summary>

  - Method
    ```
      projection_dim          # an integer to specify dimensionality of the projection head
      hidden_dim              # an integer to specify dimensionality of the hidden layers in the neural network
      moving_average_decay    # a float to specify decay rate for moving averages during training
    ```
  - Loss
    ```
      lambda_param            # a float to controlling the balance between the main loss and the orthogonality loss
    ```
  
  </details>


- <details><summary>DINO Method</summary>

  - Method
    ```
      projection_dim          # an integer to specify dimensionality of the projection head
      hidden_dim              # an integer to specify dimensionality of the hidden layers in the projection head neural network
      bottleneck_dim          # an integer to specify dimensionality of the bottleneck layer in the student network
      temp_student            # a float to specify temperature parameter for the student's logits
      temp_teacher            # a float to specify temperature parameter for the teacher's logits
      norm_last_layer         # a boolean to specify whether to normalize the last layer of the network
      momentum_teacher        # a float to control momentum coefficient for updating the teacher network
      num_crops               # an integer to determines the number of augmentations applied to each input image
      use_bn_in_head          # a boolean to spcecify whether to use batch normalization in the projection head
    ```
  - Loss
    ```
      center_momentum        # a float to control momentum coefficient for updating the center of cluster assignments
    ```

  </details>


- <details><summary>MoCo v2</summary>

  - Method
    ```
      projection_dim          # an integer to specify dimensionality of the projection head
      K                       # an integer to specify number of negative samples per positive sample in the contrastive loss
      m                       # a float to control momentum coefficient for updating the moving-average encoder
    ```
  - Loss
    ```
      temperature             # a float to control the temperature for the contrastive loss function
    ```

  </details>
  
- <details><summary>MoCo v3</summary>

  - Method  
    ```
      projection_dim          # an integer to specify dimensionality of the projection head
      hidden_dim              # an integer to specify dimensionality of the hidden layers in the projection head neural network
      moving_average_decay    # a float to specify decay rate for moving averages during training
    ```
  - Loss
    ```
      temperature             # a float to control the temperature for the contrastive loss function
    ```

  </details>


- <details><summary>SimCLR</summary>

  - Method
    ```
      projection_dim          # an integer to specify dimensionality of the projection head
      projection_num_layers   # an integer to specify the number of layers in the projection head (1: SimCLR v1, 2: SimCLR v2)
      projection_batch_norm   # a boolean to indicate whether to use batch normalization in the projection head
    ```
  - Loss
    ```
      temperature             # a float to control the temperature for the contrastive loss function
    ```

  </details>

- <details><summary>SimSiam</summary>
  
  - Method
    ```
      projection_dim          # an integer to specify dimensionality of the projection head
    ```
  - Loss
    ```
      eps                     # a float to control the stability of the loss function
    ```

  </details>
  

- <details><summary>SwAV</summary>

  - Method
    ```
      projection_dim          # an integer to specify dimensionality of the projection head
      hidden_dim              # an integer to specify dimensionality of the hidden layers in the projection head neural network
      epsilon                 # a float to control numerical stability in the algorithm
      sinkhorn_iterations     # an integer to specify the number of iterations in the Sinkhorn-Knopp algorithm
      num_prototypes          # an integer to specify the number of prototypes or clusters for contrastive learning
      queue_length            # an integer to specify rhe length of the queue for maintaining negative samples
      use_the_queue           # a boolean to indicate whether to use the queue for negative samples
      num_crops               # an integer to determines the number of augmentations applied to each input image
    ```
  - Loss
    ```
      temperature             # a float to control the temperature for the contrastive loss function
    ```

  </details>


### Training the Self-Supervised Model

Then, we'll train the self-supervised model using the specified parameters.

```python
  trainer.train(               
      batch_size=256,          # the number of training examples used in each iteration as integer
      start_epoch=1,           # the starting epoch for training as integer (if 'reload_checkpoint' parameter was True, start epoch equals to the latest checkpoint epoch)
      epochs=100,              # the total number of training epochs as integer
      optimizer="Adam",        # the optimization algorithm used for training as string (Adam, SGD, or AdamW)
      weight_decay=1e-6,       # a regularization term to prevent overfitting by penalizing large weights as float
      learning_rate=1e-3,      # the learning rate for the optimizer as float
)
```


### Evaluating th Self-Supervised Model
This evaluation assesses how well the pre-trained model performs on a dataset, specifically for tasks related to linear evaluation.
```python
trainer.evaluate(
    train_dataset=train_dataset,      # to specify the training dataset as torch.utils.data.Dataset
    test_dataset=test_dataset,        # to specify the testing dataset as torch.utils.data.Dataset
    eval_method="linear",             # the evaluation method to use as string (linear or finetune)
    top_k=1,                          # the number of top-k predictions to consider during evaluation as integer
    epochs=100,                       # the number of evaluation epochs as integer
    optimizer='Adam',                 # the optimization algorithm used during evaluation as string (Adam, SGD, or AdamW)
    weight_decay=1e-6,                # a regularization term applied during evaluation to prevent overfitting as float
    learning_rate=1e-3,               # the learning rate for the optimizer during evaluation as float
    batch_size=256,                   # the batch size used for evaluation in integer
    fine_tuning_data_proportion=1,    # the proportion of training data to use during evalutation as float in range of (0.0, 1]
)
```

### Get the Self-Supervised Model backbone

In case you want to use the pre-trained network in your own downstream task, you need to define a downstream task model. This model should include the self-supervised model backbone as one of its components. Here's an example of how to define a simple downstream model class:

```python
  class DownstreamNet(nn.Module):
      def __init__(self, backbone, **kwargs):
          super().__init__()
          self.backbone = backbone
  
          # You can define your downstream task model here
  
      def forward(self, x):
          x = self.backbone(x)
          # ...
  
  
  downstream_model = DownstreamNet(trainer.get_backbone())
```

### Loading Self-Supervised Model Checkpoint

To load a previous checkpoint into the network, you can do as below.
```python
path = 'YOUR CHECKPOINT PATH'
trainer.load_checkpoint(path)
```

### Saving Self-Supervised Model backbone
To save model backbone, you can do as below.

```python
trainer.save_backbone()
```


That's it! You've successfully trained and evaluate a self-supervised model using the AK_SSL Python library. You can further customize and experiment with different self-supervised methods, backbones, and hyperparameters to suit your specific tasks.
You can find the description of Trainer class and its function using `help` built in fuction in python.

---

## 📊 Benchmarks

We executed models and obtained results on the CIFAR10 dataset, with plans to expand our experimentation to other datasets. Please note that hyperparameters were not optimized for maximum accuracy.

|    Method    | Backbone | Batch Size | Epoch | Optimizer | Learning Rate | Weight Decay | Linear Top1 | Tensorboard | Download Backbone | Download Full Checkpoint |
|--------------|----------|------------|-------|-----------|---------------|--------------|-------------|-------------|-------------------|--------------------------|
|  BarlowTwins | Resnet18 |    256     |  800  |   Adam    |     1e-3      |     1e-6     |             |             |                   |                          |
|     BYOL     | Resnet18 |    256     |  800  |   Adam    |     1e-3      |     1e-6     |             |             |                   |                          |
|     DINO     | Resnet18 |    256     |  800  |   Adam    |     1e-3      |     1e-6     |             |             |                   |                          |
|    MoCo v2   | Resnet18 |    256     |  800  |   Adam    |     1e-3      |     1e-6     |             |             |                   |                          |
|    MoCo v3   | Resnet18 |    256     |  800  |   Adam    |     1e-3      |     1e-6     |             |             |                   |                          |
|   SimCLR v1  | Resnet18 |    256     |  800  |   Adam    |     1e-3      |     1e-6     |             |             |                   |                          |
|   SimCLR v2  | Resnet18 |    256     |  800  |   Adam    |     1e-3      |     1e-6     |             |             |                   |                          |
|    SimSiam   | Resnet18 |    256     |  800  |   Adam    |     1e-3      |     1e-6     |             |             |                   |                          |
|     SwAv     | Resnet18 |    256     |  800  |   Adam    |     1e-3      |     1e-6     |             |             |                   |                          |

---

## 📜 References Used

In the development of this project, we have drawn inspiration and utilized code, libraries, and resources from various sources. We would like to acknowledge and express our gratitude to the following references and their respective authors:

- [Lightly Library](https://github.com/lightly-ai/lightly)
- [PYSSL Library](https://github.com/giakou4/pyssl)
- [SimCLR Implementation](https://github.com/Spijkervet/SimCLR)
- All original codes of supported methods

These references have played a crucial role in enhancing the functionality and quality of our project. We extend our thanks to the authors and contributors of these resources for their valuable work.

---

## 💯 License

This project is licensed under the [MIT License](./LICENSE).

---

## 🤝 Collaborators
By:
  - [Kian Majlessi](https://github.com/kianmajl)
  - [Audrina Ebrahimi](https://github.com/audrina-ebrahimi)

Thanks to [Dr. Peyman Adibi](https://scholar.google.com/citations?user=u-FQZMkAAAAJ) and [Dr. Hossein Karshenas](https://scholar.google.com/citations?user=BjMFkWEAAAAJ), for their invaluable guidance and support throughout this project.
