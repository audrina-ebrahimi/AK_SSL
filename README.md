<h1>
<br>AK_SSL: A Self-Supervised Learning Library
</h1>

---

## 📒 Table of Contents
- [📒 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [📂 Project Structure](#-project-structure)
- [✍️ Self Supervised Learning](#-self-supervised-learning)
- [🔎 Supported Methods](#-supported-methods)
- [🚀 Getting Started](#-getting-started)
- [🤝 Collaborators](#-collaborators)


---
## 📍 Overview
Welcome to the Self-Supervised Learning Library! This repository hosts a collection of tools and implementations for self-supervised learning. Self-supervised learning is a powerful paradigm that leverages unlabeled data to pre-train models, which can then be fine-tuned on specific tasks with smaller labeled datasets. This library aims to provide researchers and practitioners with a comprehensive set of tools to experiment, learn, and apply self-supervised learning techniques effectively.
This project was our assignment during the summer apprenticeship in the newly established Intelligent and Learning System laboratory at the University of Isfahan.

---


## 📂 Project Structure

 * [README.md](./README.md)
 * [AK_SSL](./AK_SSL)
   * [Trainer.py](./AK_SSL/Trainer.py)
   
  
---

## ✍️ Self Supervised Learning

Self-supervised learning is a subfield of machine learning where models are trained to predict certain aspects of the input data without relying on manual labeling. This approach has gained significant attention due to its ability to leverage large amounts of unlabeled data, which is often easier to obtain than fully annotated datasets. This library provides implementations of various self-supervised techniques, allowing you to experiment with and apply these methods in your own projects.

---

## 🔎 Supported Methods

### [BarlowTwins](./AK_SSL/models/barlowtwins.py)
Barlow Twins is a self-supervised learning method that aims to learn embeddings invariant to distortions of the input sample. It achieves this by applying two distinct sets of augmentations to the same input sample, resulting in two distorted views of the same image. The objective function measures the cross-correlation matrix between the outputs of two identical networks fed with these distorted sample versions, striving to make it as close to the identity matrix as possible. This causes the embedding vectors of the distorted sample versions to become similar while minimizing redundancy among the components of these vectors. Barlow Twins particularly benefits from utilizing high-dimensional output vectors.

### [BYOL](./AK_SSL/models/byol.py)
BYOL (Bootstrap Your Own Latent) is one of the new approaches to self-supervised learning. Like other methods, BYOL aims to learn a representation that can be utilized for downstream tasks. It employs two neural networks for learning: the online and target networks. The online network is trained to predict the target network's representation of the same image from a different augmented view. Simultaneously, the target network is updated with a slow-moving average of the online network's parameters. While state-of-the-art methods rely on negative pairs, BYOL achieves a new state of the art without them. It directly minimizes the similarity between the representations of the same image from different augmented views (positive pair).

### [DINO](./AK_SSL/models/dino.py)
DINO (self-distillation with no labels) is a self-supervised learning method that directly predicts the output of a teacher network—constructed with a momentum encoder—by utilizing a standard cross-entropy loss. It is an innovative self-supervised learning algorithm developed by Facebook AI. Through the utilization of self-supervised learning with Transformers, DINO paves the way for creating machines that can comprehend images and videos at a much deeper level.

### [MoCos](./AK_SSL/models/moco.py)
MoCo, short for Momentum Contrast, is a self-supervised learning algorithm that employs a contrastive loss. MoCo v2 represents an enhanced iteration of the original Momentum Contrast self-supervised learning algorithm. Motivated by the findings outlined in the SimCLR paper, the authors introduced several modifications in MoCo v1, which included replacing the 1-layer fully connected layer with a 2-layer MLP head featuring ReLU activation for the unsupervised training stage. Additionally, they incorporated blur augmentation and adopted a cosine learning rate schedule. These adjustments enabled MoCo to outperform the state-of-the-art SimCLR, even when utilizing a smaller batch size and fewer epochs.

MoCo v3, introduced in the paper "An Empirical Study of Training Self-Supervised Vision Transformers," represents another advancement in self-supervised learning. It builds upon the foundation of MoCo v1 / MoCo v2 and addresses the instability issue observed when employing ViT for self-supervised learning.

In contrast to MoCo v2, MoCo v3 adopts a different approach where the keys naturally coexist within the same batch. The memory queue (memory bank) is discarded, resulting in a setting similar to that of SimCLR. The encoder fq comprises a backbone (e.g., ResNet, ViT), a projection head, and an additional prediction head.

### [SimCLRs](./AK_SSL/models/simclr.py)
SimCLR (Simple Framework for Contrastive Learning of Representations) is a self-supervised technique used to learn image representations. The fundamental building blocks of contrastive self-supervised methods, such as SimCLR, are image transformations. Each image is transformed into multiple new images through randomly applied augmentations. The goal of the self-supervised model is to identify images that originate from the same original source among a set of negative examples. SimCLR operates on the principle of maximizing the similarity between positive pairs of augmented images while minimizing the similarity with negative pairs. The training process can be summarized as follows: Data Augmentation - SimCLR employs robust data augmentation techniques to generate multiple augmented versions of each input image.

### [SimSiam](./AK_SSL/models/simsiam.py)
SimSiam is a self-supervised representation learning model that was proposed by Facebook AI Research (FAIR). It is a simple Siamese network designed to learn meaningful representations without requiring negative sample pairs, large batches, or momentum encoders.

### [SwAV](./AK_SSL/models/swav.py)
SwAV, or Swapping Assignments Between Views, is a self-supervised learning approach that takes advantage of contrastive methods without requiring to compute pairwise comparisons. Specifically, it simultaneously clusters the data while enforcing consistency between cluster assignments produced for different augmentations (or views) of the same image, instead of comparing features directly as in contrastive learning. Simply put, SwAV uses a swapped prediction mechanism where we predict the cluster assignment of a view from the representation of another view.

---
## 🚀 Getting Started

### ✔️ Requirements

Before you begin, ensure that you have the packages in `requirements.txt` installed.

### 📦 Installation

1. Clone the AK_SSL repository:
```sh
git clone https://github.com/audrina-ebrahimi/AK_SSL.git
```

2. Change to the project directory:
```sh
cd AK_SSL
```

3. Install the dependencies:
```sh
pip install -r ./Codes/requirements.txt
```

### 🎮 Using AK_SSL



---
## 🤝 Collaborators
[Kian Majlessi](https://github.com/kianmajl) and [Audrina Ebrahimi](https://github.com/audrina-ebrahimi)
