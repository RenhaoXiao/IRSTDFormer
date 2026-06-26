# IRSTDFormer

IRSTDFormer is an infrared small target detection project built on a real-time
DETR-style detection pipeline. The repository contains training, evaluation,
inference, and deployment utilities for COCO-format infrared small target
datasets.

## Features

- Single-class infrared small target detection with COCO-style annotations.
- Config-driven training and evaluation.
- Lightweight HGNetv2 / LiteEncoder based model configuration.
- PyTorch inference plus ONNX, TensorRT, and OpenVINO export or inference tools.
- Visualization and COCO prediction export helpers.

## Repository Structure

```text
configs/
  IRSTDFormer/          Main IRSTDFormer experiment config
  dataset/              Infrared small target dataset templates
  base/                 Shared model, optimizer, and dataloader configs
engine/                 Model, data, solver, optimizer, and utility code
tools/
  inference/            PyTorch, ONNX, TensorRT, and OpenVINO inference
  deployment/           Export helpers
  eval/                 Evaluation format conversion helpers
  visualization/        Visualization utilities
train.py                Training and evaluation entry point
```

## Installation

```bash
pip install -r requirements.txt
```

Install the correct PyTorch build for your CUDA environment if the pinned wheel
in `requirements.txt` does not match your system.

## Dataset Setup

Datasets should be converted to COCO detection format. By default, the configs
look for datasets under `./datasets/`. Update the paths in:

- `configs/dataset/IRSTD-1k_detection.yml`
- `configs/dataset/NUDT-SIRST.yml`

Each dataset config expects image folders and COCO JSON annotation files for
training and validation.

## Training

```bash
python train.py -c configs/IRSTDFormer/IRSTDFormer_IRSTD-1k.yml -d cuda:0
```

To resume from a checkpoint:

```bash
python train.py -c configs/IRSTDFormer/IRSTDFormer_IRSTD-1k.yml -r path/to/checkpoint.pth -d cuda:0
```

## Evaluation

```bash
python train.py -c configs/IRSTDFormer/IRSTDFormer_IRSTD-1k.yml -r path/to/checkpoint.pth --test-only -d cuda:0
```

## Inference

PyTorch inference scripts are available under `tools/inference/`. Export helpers
for ONNX and deployment workflows are available under `tools/deployment/`.

## License

This project is released under the Apache License 2.0. See `LICENSE`.

IRSTDFormer is modified from DEIMv2 (Intellindust-AI-Lab/DEIMv2). Copyright
(c) 2026 The IRSTDFormer Authors. All Rights Reserved.

This repository is modified from DEIMv2 (Intellindust-AI-Lab/DEIMv2). See
`NOTICE` and source file headers for attribution details.
