
<h1>
<br>AK_SSL: A Self-Supervised Learning Library
</h1>



![GitHub](https://img.shields.io/github/license/audrina-ebrahimi/AK_SSL) ![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

---

## 📒 Table of Contents
- [📒 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [✍️ Self Supervised Learning](#-self-supervised-learning)
- [🔎 Supported Methods](#-supported-methods)
- [🚀 Getting Started](#-getting-started)
- [💡 Tutorial](#-tutorial)
- [📊 Benchmarks](#-benchmarks)
- [📜 References Used](#-reference-used)
- [💯 License](#-license)
- [🤝 Collaborators](#-collaborators)


---
## 📍 Overview
Welcome to the Self-Supervised Learning Library! This repository hosts a collection of tools and implementations for self-supervised learning. Self-supervised learning is a powerful paradigm that leverages unlabeled data to pre-train models, which can then be fine-tuned on specific tasks with smaller labeled datasets. This library aims to provide researchers and practitioners with a comprehensive set of tools to experiment, learn, and apply self-supervised learning techniques effectively.
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
## 🚀 Getting Started

### ✔️ Requirements

Before you begin, ensure that you have the packages in `requirements.txt` installed.

### 📦 Installation

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
    save_dir="./save_for_report/",  # directory to save training checkpoints and logs as string
    checkpoint_interval=50,         # interval (in epochs) for saving checkpoints as integer
    reload_checkpoint=False,        # reload a previously saved checkpoint as boolean
    **kwargs                        # other argument 
)
```
Note: The use of **kwargs can differ between methods, depending on the specific method, loss function, transformation, and other factors. If you are utilizing any of the objectives listed below, you must provide their arguments during the initialization of the Trainer class.

- <details><summary>SimCLR Transformation</summary>
  
  ```
    color_jitter_strength     # as float to Set the strength of color
    use_blur                  # s boolean to apply blur augmentation
    mean
    std
  ```
  
  </details>

- <details><summary>BarlowTwins</summary>

  - Method
    ```
      projection_dim
      hidden_dim
      moving_average_decay
    ```
  - Loss
    ```
      lambda_param
    ```
  
  </details>


- <details><summary>DINO Method</summary>

  - Method
    ```
      projection_dim
      hidden_dim
      bottleneck_dim
      temp_student
      temp_teacher
      norm_last_layer
      momentum_teacher
      num_crops
      use_bn_in_head
    ```
  - Loss
    ```
      center_momentum
    ```

  </details>


- <details><summary>MoCo v2</summary>

  - Method
    ```
      projection_dim
      temperature
      K
      m
    ```

  </details>
  
- <details><summary>MoCo v3</summary>

  - Method  
    ```
      projection_dim
      hidden_dim
      moving_average_decay
    ```
  - Loss
    ```
      temperature
    ```

  </details>


- <details><summary>SimCLR</summary>

  - Method
    ```
      projection_dim
      projection_num_layers
      projection_batch_norm
    ```
  - Loss
    ```
      temperature
    ```

  </details>

- <details><summary>SimSiam</summary>
  
  - Method
    ```
      projection_dim
    ```
  - Loss
    ```
      eps
    ```

  </details>
  

- <details><summary>SwAV</summary>

  - Method
    ```
      projection_dim
      hidden_dim
      epsilon
      sinkhorn_iterations
      num_prototypes
      queue_length
      use_the_queue
      num_crops
    ```
  - Loss
    ```
      temperature
    ```

  </details>

You can find the description of Trainer class and function using help built in fuction in python.

---

## 📊 Benchmarks

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
