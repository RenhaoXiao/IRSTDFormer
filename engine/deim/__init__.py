"""
IRSTDFormer: A Layer-wise Feature Aggregation and BBox Correction Network for Infrared Small Target Detection
Copyright (c) 2026 The IRSTDFormer Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIMv2 (Intellindust-AI-Lab/DEIMv2)
Copyright (c) 2025 The DEIM Authors. All Rights Reserved.
"""


from .deim import DEIM

from .matcher import HungarianMatcher

from .hybrid_encoder import HybridEncoder
from .lite_encoder import LiteEncoder


from .dfine_decoder import DFINETransformer
from .rtdetrv2_decoder import RTDETRTransformerv2

from .postprocessor import PostProcessor
from .deim_criterion import DEIMCriterion
from .deim_decoder import DEIMTransformer