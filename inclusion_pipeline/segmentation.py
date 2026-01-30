"""
MicroSAM segmentation wrapper using automatic instance segmentation.
"""

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from micro_sam.automatic_segmentation import (
    get_predictor_and_segmenter,
    automatic_instance_segmentation,
)


class MicroSAMSegmenter:
    """
    Wrapper for MicroSAM automatic instance segmentation.
    
    Example:
        >>> segmenter = MicroSAMSegmenter()
        >>> mask = segmenter.segment(image)
    """
    
    def __init__(
        self,
        repo_id: str = "sunny17347/machine_learning_models",
        filename: str = "inclusion_segmentation_12_3_25.pt",
        model_type: str = "vit_b_lm",
        device: torch.device = None,
        is_tiled: bool = False,
    ):
        """
        Initialize MicroSAM segmenter.
        
        Args:
            repo_id: HuggingFace repo ID containing the checkpoint
            filename: Name of the checkpoint file
            model_type: MicroSAM model type (e.g., "vit_b_lm", "vit_l_lm")
            device: Torch device
            is_tiled: Whether to use tiled prediction for large images
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model_type = model_type
        
        # Download checkpoint from HuggingFace
        print(f"Downloading SAM checkpoint from {repo_id}/{filename}...")
        self.checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )
        print(f"Checkpoint downloaded to: {self.checkpoint_path}")
        
        # Initialize predictor and segmenter
        print("Initializing MicroSAM predictor and segmenter...")
        self.predictor, self.segmenter = get_predictor_and_segmenter(
            model_type=self.model_type,
            checkpoint=self.checkpoint_path,
            device=self.device,
            is_tiled=is_tiled,
        )
        print("MicroSAM initialized!")
    
    def segment(self, image: np.ndarray, ndim: int = 2) -> np.ndarray:
        """
        Run automatic instance segmentation on an image.
        
        Args:
            image: Input image as numpy array. For 2D images, shape should be (H, W)
                   or (H, W, C). The image should be preprocessed/normalized.
            ndim: Number of spatial dimensions (2 for 2D images)
            
        Returns:
            Instance segmentation mask where each unique value represents
            a different object instance. 0 is background.
        """
        instance_segmentation = automatic_instance_segmentation(
            predictor=self.predictor,
            segmenter=self.segmenter,
            input_path=image,
            ndim=ndim,
        )
        
        return instance_segmentation
