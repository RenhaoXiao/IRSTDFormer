"""
IRSTDFormer: A Layer-wise Feature Aggregation and BBox Correction Network for Infrared Small Target Detection
Copyright (c) 2026 The IRSTDFormer Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIMv2 (Intellindust-AI-Lab/DEIMv2)
Copyright (c) 2025 The DEIM Authors. All Rights Reserved.
"""

                                  
from .coco_dataset import CocoDetection
from .coco_dataset import (
    mscoco_category2name,
    mscoco_category2label,
    mscoco_label2category,
)
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
from .voc_detection import VOCDetection
from .voc_eval import VOCEvaluator
