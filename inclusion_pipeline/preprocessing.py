"""
Image preprocessing and normalization utilities for microscopy images.
"""

import numpy as np
from skimage.filters import gaussian
from skimage import exposure


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize the image to the range [0, 1].
    
    Args:
        image: Input image
        
    Returns:
        Normalized image in [0, 1]
    """
    img_min, img_max = np.min(image), np.max(image)
    if img_max > img_min:
        return (image - img_min) / (img_max - img_min)
    return np.zeros_like(image, dtype=np.float32)


def preprocess_for_segmentation(
    image: np.ndarray,
    sigma: float = 2.0,
    sigmoid_cutoff: float = 0.25,
) -> np.ndarray:
    """
    Preprocess image for MicroSAM segmentation.
    
    This matches the preprocessing used in training:
    - Applies Gaussian blur to reduce noise
    - Enhances contrast using sigmoid adjustment
    - Normalizes intensities to [0, 1]
    
    Args:
        image: Input image (grayscale or single channel)
        sigma: Gaussian blur sigma (default: 2.0)
        sigmoid_cutoff: Cutoff for sigmoid contrast adjustment (default: 0.25)
        
    Returns:
        Preprocessed image normalized to [0, 1]
        
    Example:
        >>> from inclusion_pipeline import preprocess_for_segmentation
        >>> preprocessed = preprocess_for_segmentation(image, sigma=2, sigmoid_cutoff=0.25)
    """
    # Apply Gaussian blur
    blurred = gaussian(image, sigma=sigma)
    
    # Apply sigmoid contrast adjustment
    enhanced = exposure.adjust_sigmoid(image, cutoff=sigmoid_cutoff)
    
    # Normalize to [0, 1]
    normalized = normalize_image(enhanced)
    
    return normalized.astype(np.float32)
