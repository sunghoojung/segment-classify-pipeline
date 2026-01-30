"""
Utility functions for the segmentation-classification pipeline.
"""

import numpy as np
from typing import Tuple


def colorize_mask(
    mask: np.ndarray,
    swiss_cheese_color: Tuple[int, int, int] = (0, 255, 0),   # Green
    solid_color: Tuple[int, int, int] = (255, 0, 0),       # Red
    background_color: Tuple[int, int, int] = (0, 0, 0),        # Black
) -> np.ndarray:
    """
    Convert a combined mask (0=bg, 1=non-inclusion, 2=inclusion) to a colored RGB image.
    
    Args:
        mask: Combined mask with values 0, 1, 2
        swiss_cheese_color: RGB color for swiss cheese inclusions (default: green)
        solid_color: RGB color for solid inclusions (default: red)
        background_color: RGB color for background (default: black)
        
    Returns:
        RGB image (H, W, 3) with colored regions
    """
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    colored[mask == 0] = background_color
    colored[mask == 1] = swiss_cheese_color
    colored[mask == 2] = solid_color
    
    return colored


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    swiss_cheese_color: Tuple[int, int, int] = (0, 255, 0),
    solid_color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay the classification mask on the original image.
    
    Args:
        image: Original RGB image (H, W, 3). If grayscale, will be converted.
        mask: Combined mask with values 0, 1, 2
        swiss_cheese_color: RGB color for swiss cheese inclusions (default: green)
        solid_color: RGB color for solid inclusions (default: red)
        alpha: Transparency of the overlay (0=invisible, 1=opaque)
        
    Returns:
        Image with semi-transparent mask overlay
        
    Example:
        >>> from inclusion_pipeline import overlay_mask_on_image
        >>> overlay = overlay_mask_on_image(image_rgb, result.combined_mask)
    """
    # Convert grayscale to RGB if needed
    if image.ndim == 2:
        img_normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
        img_8bit = (img_normalized * 255).astype(np.uint8)
        image = np.stack([img_8bit, img_8bit, img_8bit], axis=-1)
    
    # Create colored mask
    colored_mask = colorize_mask(mask, swiss_cheese_color, solid_color, (0, 0, 0))
    
    # Create overlay
    overlay = image.copy().astype(np.float32)
    
    # Only blend where mask is non-zero
    mask_regions = mask > 0
    overlay[mask_regions] = (
        (1 - alpha) * overlay[mask_regions] + 
        alpha * colored_mask[mask_regions].astype(np.float32)
    )
    
    return overlay.astype(np.uint8)
