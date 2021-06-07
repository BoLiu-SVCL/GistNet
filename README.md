# GistNet: a Geometric Structure Transfer Network for Long-Tailed Recognition

## Overview
This is the author's pytorch implementation for the paper "GistNet: a Geometric Structure Transfer Network for Long-Tailed Recognition". This will reproduce the GistNet performance on ImageNet-LT with ResNet10.

The model is designed to train on two (2) Titan Xp (12GB memory each). Please adjust the batch size (or even learning rate) accordingly, if the GPU setting is different.

Dataloader and sampler inherit from OTLR.

## Requirements
* [Python](https://python.org/) (version 3.7.6 tested)
* [PyTorch](https://pytorch.org/) (version 1.4.0 tested)

## Data Preparation
- First, please download the [ImageNet_2014](http://image-net.org/index).

- Next, change the `data_root` in `pretrain.py`, `train.py`, and `eval.py` accordingly.

- The data splits are provided in the codes.

## Getting Started (Training & Testing)
- pre-training:
```
sh pretrain.sh
```
- training:
```
sh train.sh
```
- testing:
```
sh eval.sh
```
