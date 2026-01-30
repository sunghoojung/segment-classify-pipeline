"""
Inclusion Pipeline - MicroSAM Segmentation + ResNet101 Classification

A pipeline for microscopy image analysis that segments objects using MicroSAM
and classifies them using a fine-tuned ResNet101 model.

Installation:
    conda install -c conda-forge micro_sam
    pip install git+https://github.com/sunghoojung/segment-classify-pipeline.git

Usage:
    from inclusion_pipeline import SegmentClassifyPipeline
    
    pipeline = SegmentClassifyPipeline()
    result = pipeline(image)
    
    # With debugging visualization
    result = pipeline(image, verbose=True)
    
    # Access results
    print(f"Swiss cheese found: {result.num_swiss_cheese}")
    print(f"Solid found: {result.num_solid}")
    
    # Get masks
    swiss_cheese_mask = result.swiss_cheese_mask
    solid_mask = result.solid_mask
    combined_mask = result.combined_mask  # 0=bg, 1=swiss cheese, 2=solid
"""

from .pipeline import SegmentClassifyPipeline, PipelineOutput
from .segmentation import MicroSAMSegmenter
from .classification import ResNetClassifier
from .preprocessing import preprocess_for_segmentation, normalize_image
from .utils import overlay_mask_on_image, colorize_mask

__version__ = "0.1.0"
__author__ = "sunghoojung"

__all__ = [
    # Main pipeline
    "SegmentClassifyPipeline",
    "PipelineOutput",
    # Components (for advanced users)
    "MicroSAMSegmenter",
    "ResNetClassifier",
    # Utilities
    "preprocess_for_segmentation",
    "normalize_image",
    "overlay_mask_on_image",
    "colorize_mask",
]
