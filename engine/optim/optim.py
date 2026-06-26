"""
IRSTDFormer: A Layer-wise Feature Aggregation and BBox Correction Network for Infrared Small Target Detection
Copyright (c) 2026 The IRSTDFormer Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIMv2 (Intellindust-AI-Lab/DEIMv2)
Copyright (c) 2025 The DEIM Authors. All Rights Reserved.
"""


import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ..core import register


__all__ = ['AdamW', 'SGD', 'Adam', 'MultiStepLR', 'CosineAnnealingLR', 'OneCycleLR', 'LambdaLR']



SGD = register()(optim.SGD)
Adam = register()(optim.Adam)
AdamW = register()(optim.AdamW)


MultiStepLR = register()(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register()(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register()(lr_scheduler.OneCycleLR)
LambdaLR = register()(lr_scheduler.LambdaLR)
