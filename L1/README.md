# L1: Introduction

## What is ML

"ask machine to help us to find a *function*"

<img src="../img/image-20240521151343106.png" alt="image-20240521151343106" style="zoom:50%;" />

DL: use neural network to find *the function*

<img src="../img/image-20240521211515732.png" alt="image-20240521211515732" style="zoom:50%;" />

Output: regression, classification, rich content

## How to "teach" machine

- Supervised Learning 监督学习

​	Collect data -> Tag data(labeles) -> Train

- Self-supervised Learning 无监督学习

  Why? Collect data is not efficient for every work.

  <img src="../img/image-20240521212308892.png" alt="image-20240521212308892" style="zoom:50%;" />

  Pre-Train("learn basic skill") -> Downstream Tasks

## PyTorch

Training - Validation - Testing

### Dataset and Dataloader

```python
DataLoader(dataset, batch_size, _shuffle_)
```

Training: shuffle=True
Testing: shuffle=False

<img src="../img/image-20240521214915711.png" alt="image-20240521214915711" style="zoom:50%;" />

### Tensor

high-dimensional matrices

<img src="../img/image-20240521215323124.png" alt="image-20240521215323124" style="zoom:50%;" />

```x.transpose(0, 1)``` exchange the dim

```x.squeeze(0)```  remove a dim

```x.unsqueeze(1)``` expand a dim

```torch.cat([x,y,z], dim=1)``` concatenate tensors with specific dim

