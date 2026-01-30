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
    print(f"Inclusions found: {result.num_inclusions}")
    print(f"Non-inclusions found: {result.num_non_inclusions}")
    
    # Get masks
    inclusion_mask = result.inclusion_mask
    non_inclusion_mask = result.non_inclusion_mask
    combined_mask = result.combined_mask  # 0=bg, 1=non-inclusion, 2=inclusion
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
