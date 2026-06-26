"""
IRSTDFormer: A Layer-wise Feature Aggregation and BBox Correction Network for Infrared Small Target Detection
Copyright (c) 2026 The IRSTDFormer Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIMv2 (Intellindust-AI-Lab/DEIMv2)
Copyright (c) 2025 The DEIM Authors. All Rights Reserved.
"""


import torch.cuda.amp as amp

from ..core import register


__all__ = ['GradScaler']

GradScaler = register()(amp.grad_scaler.GradScaler)
