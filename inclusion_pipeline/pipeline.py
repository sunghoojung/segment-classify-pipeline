"""
Main pipeline: Preprocess → MicroSAM Segmentation → ResNet Classification
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, List
from skimage.measure import label, regionprops

from .preprocessing import preprocess_for_segmentation
from .segmentation import MicroSAMSegmenter
from .classification import ResNetClassifier


@dataclass
class PipelineOutput:
    """
    Output from the segmentation and classification pipeline.
    
    Attributes:
        swiss_cheese_mask: Binary mask of swiss cheese objects (H, W)
        solid_mask: Binary mask of solid objects (H, W)
        combined_mask: Combined mask where 0=background, 1=swiss cheese, 2=solid
        num_swiss_cheese: Count of detected swiss cheese
        num_solid: Count of detected solid
        instance_masks: List of individual binary masks for each object
        instance_labels: List of class labels (0=solid, 1=swiss cheese) for each object
    """
    swiss_cheese_mask: np.ndarray
    solid_mask: np.ndarray
    combined_mask: np.ndarray
    num_swiss_cheese: int
    num_solid: int
    instance_masks: List[np.ndarray]
    instance_labels: List[int]
    
    # Aliases for backward compatibility
    @property
    def non_inclusion_mask(self) -> np.ndarray:
        """Alias for solid_mask."""
        return self.solid_mask

    @property
    def inclusion_mask(self) -> np.ndarray:
        """Alias for swiss_cheese_mask."""
        return self.swiss_cheese_mask

    @property
    def num_non_inclusions(self) -> int:
        """Alias for num_solid."""
        return self.num_solid

    @property
    def num_inclusions(self) -> int:
        """Alias for num_swiss_cheese."""
        return self.num_swiss_cheese

    @property
    def class1_mask(self) -> np.ndarray:
        """Alias for solid_mask (class 0)."""
        return self.solid_mask
    
    @property
    def class2_mask(self) -> np.ndarray:
        """Alias for swiss_cheese_mask (class 1)."""
        return self.swiss_cheese_mask
    
    @property
    def num_class1(self) -> int:
        """Alias for num_solid (class 0)."""
        return self.num_solid
    
    @property
    def num_class2(self) -> int:
        """Alias for num_swiss_cheese (class 1)."""
        return self.num_swiss_cheese


def _display_two_images(image1, image2, title1, title2):
    """Display two images side-by-side for debugging."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(image1, cmap='gray' if image1.ndim == 2 else None)
    axes[0].set_title(title1, fontsize=10)
    axes[0].axis('off')
    
    axes[1].imshow(image2, cmap='gray' if image2.ndim == 2 else None)
    axes[1].set_title(title2, fontsize=10)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def _display_single_image(image, title):
    """Display a single image for debugging."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def _display_crop(crop, label_name, obj_idx, area, prob=None):
    """Display a single crop being classified."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(crop, cmap='gray' if crop.ndim == 2 else None)
    title = f'Object {obj_idx}: {label_name}\n(area={area}, dtype={crop.dtype})'
    if prob is not None:
        title += f'\nconf={prob:.2f}'
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def _display_classification_result(original, mask, swiss_cheese_mask, solid_mask, 
                                   num_swiss_cheese, num_solid):
    """Display final classification results."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original
    axes[0].imshow(original, cmap='gray' if original.ndim == 2 else None)
    axes[0].set_title('Original Image', fontsize=10)
    axes[0].axis('off')
    
    # Instance segmentation
    axes[1].imshow(mask, cmap='nipy_spectral')
    axes[1].set_title(f'Instance Segmentation ({mask.max()} objects)', fontsize=10)
    axes[1].axis('off')
    
    # Swiss cheese inclusions
    axes[2].imshow(swiss_cheese_mask, cmap='Reds')
    axes[2].set_title(f'Swiss Cheese ({num_swiss_cheese})', fontsize=10)
    axes[2].axis('off')
    
    # Solid inclusions
    axes[3].imshow(solid_mask, cmap='Greens')
    axes[3].set_title(f'Solid ({num_solid})', fontsize=10)
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()


class SegmentClassifyPipeline:
    """
    Pipeline that segments objects using MicroSAM and classifies each
    object as swiss cheese or solid using a ResNet101 model.
    
    Uses regionprops to extract bounding boxes and crops from the original image.
    
    Example:
        >>> from inclusion_pipeline import SegmentClassifyPipeline
        >>> import numpy as np
        >>> from PIL import Image
        >>> 
        >>> # Initialize (downloads weights automatically on first run)
        >>> pipeline = SegmentClassifyPipeline()
        >>> 
        >>> # Load your microscopy image
        >>> image = np.array(Image.open("cell_image.tif"))
        >>> 
        >>> # Run the pipeline
        >>> result = pipeline(image)
        >>> 
        >>> # Run with visualization for debugging
        >>> result = pipeline(image, verbose=True)
    """
    
    def __init__(
        self,
        sam_repo_id: str = "sunny17347/machine_learning_models",
        sam_filename: str = "inclusion_segmentation_12_3_25.pt",
        resnet_repo_id: str = "sunny17347/machine_learning_models",
        resnet_filename: str = "inclusion_classifier_v1.pth",
        sam_model_type: str = "vit_b_lm",
        resnet_variant: str = "resnet101",
        device: Optional[str] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            sam_repo_id: HuggingFace repo ID for SAM weights
            sam_filename: Filename of SAM checkpoint in the repo
            resnet_repo_id: HuggingFace repo ID for ResNet weights
            resnet_filename: Filename of ResNet checkpoint in the repo
            sam_model_type: MicroSAM model type (default: "vit_b_lm")
            resnet_variant: ResNet architecture (default: "resnet101")
            device: Device to run on ("cuda" or "cpu"). Auto-detected if None.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        print(f"Initializing pipeline on {self.device}...")
        
        # Initialize segmenter
        self.segmenter = MicroSAMSegmenter(
            repo_id=sam_repo_id,
            filename=sam_filename,
            model_type=sam_model_type,
            device=self.device,
        )
        
        # Initialize classifier
        self.classifier = ResNetClassifier(
            repo_id=resnet_repo_id,
            filename=resnet_filename,
            device=self.device,
            resnet_variant=resnet_variant,
        )
        
        print("Pipeline ready!")
    
    @torch.no_grad()
    def __call__(
        self,
        image: np.ndarray,
        preprocess: bool = True,
        sigma: float = 2.0,
        sigmoid_cutoff: float = 0.25,
        verbose: bool = False,
    ) -> PipelineOutput:
        """
        Run the full pipeline on an image.
        
        Args:
            image: Input image as numpy array. Can be:
                   - Grayscale: (H, W)
                   - RGB: (H, W, 3)
                   - 16-bit or 8-bit
            preprocess: Whether to apply preprocessing (gaussian + sigmoid + normalize)
            sigma: Gaussian blur sigma for preprocessing
            sigmoid_cutoff: Cutoff for sigmoid contrast adjustment
            verbose: If True, display images at each step for debugging
            show_crops: If True (and verbose=True), show each crop being classified
            
        Returns:
            PipelineOutput with masks and classification results
        """
        # Get image dimensions
        if image.ndim == 2:
            h, w = image.shape
        else:
            h, w = image.shape[:2]
        
        if verbose:
            _display_single_image(image, 'Step 1: Original Image')
        
        # 1. Preprocess for segmentation
        if preprocess:
            preprocessed = preprocess_for_segmentation(
                image, 
                sigma=sigma, 
                sigmoid_cutoff=sigmoid_cutoff
            )
            if verbose:
                _display_two_images(
                    image, preprocessed,
                    'Original', 'Step 2: After Preprocessing (gaussian + sigmoid + normalize)'
                )
        else:
            preprocessed = image
            if verbose:
                print("Step 2: Preprocessing skipped")
        
        # 2. Segment with MicroSAM
        if verbose:
            print("Step 3: Running MicroSAM segmentation...")
        
        instance_segmentation = self.segmenter.segment(preprocessed, ndim=2)
        
        if verbose:
            _display_two_images(
                preprocessed, instance_segmentation,
                'Preprocessed Image', f'Step 3: Instance Segmentation ({instance_segmentation.max()} objects)'
            )
        
        # 3. Initialize output masks
        solid_mask = np.zeros((h, w), dtype=np.uint8)
        swiss_cheese_mask = np.zeros((h, w), dtype=np.uint8)
        instance_masks = []
        instance_labels = []
        
        # 4. Label the segmentation and use regionprops
        labeled_segmentation = label(instance_segmentation > 0)
        regions = regionprops(labeled_segmentation)
        
        if verbose:
            print(f"Step 4: Classifying {len(regions)} objects using regionprops...")
        
        # 5. Classify each object using regionprops
        for i, region in enumerate(regions):
            
            # Get bounding box: (min_row, min_col, max_row, max_col)
            min_row, min_col, max_row, max_col = region.bbox
            
            # Add a small buffer around the crop 
            buffer = 5
            min_row = max(0, min_row - buffer)
            min_col = max(0, min_col - buffer)
            max_row = min(h, max_row + buffer)
            max_col = min(w, max_col + buffer)
            
            # Extract crop from ORIGINAL image using bounding box
            crop = image[min_row:max_row, min_col:max_col].copy()
            
            # Get the object mask
            obj_mask = (labeled_segmentation == region.label).astype(np.uint8)
            
            # Classify: {'solid': 0, 'swiss_cheese': 1}
            pred_class = self.classifier.predict(crop)
            probs = self.classifier.predict_proba(crop)
            confidence = probs[pred_class]
            label_name = "swiss_cheese" if pred_class == 1 else "solid"
            
            if verbose:
                print(f"  Object {i+1}: {label_name} (area={region.area}, "
                      f"crop_shape={crop.shape}, crop_dtype={crop.dtype}, "
                      f"crop_min={crop.min():.1f}, crop_max={crop.max():.1f}, "
                      f"conf={confidence:.3f})")
                _display_crop(crop, label_name, i+1, region.area, confidence)
            
            # Add to appropriate mask
            if pred_class == 0:
                solid_mask = np.maximum(solid_mask, obj_mask)
            else:
                swiss_cheese_mask = np.maximum(swiss_cheese_mask, obj_mask)
            
            instance_masks.append(obj_mask)
            instance_labels.append(pred_class)
        
        # 6. Create combined mask: 0=bg, 1=swiss cheese, 2=solid
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        combined_mask[swiss_cheese_mask > 0] = 1
        combined_mask[solid_mask > 0] = 2
        
        num_swiss_cheese = int((np.array(instance_labels) == 1).sum()) if instance_labels else 0
        num_solid = int((np.array(instance_labels) == 0).sum()) if instance_labels else 0
        
        if verbose:
            print(f"\nStep 5: Classification complete!")
            print(f"  Total objects classified: {len(instance_labels)}")
            print(f"  Swiss cheese: {num_swiss_cheese}")
            print(f"  Solid: {num_solid}")
            
            _display_classification_result(
                image, instance_segmentation,
                swiss_cheese_mask, solid_mask,
                num_swiss_cheese, num_solid
            )
        
        return PipelineOutput(
            swiss_cheese_mask=swiss_cheese_mask,
            solid_mask=solid_mask,
            combined_mask=combined_mask,
            num_swiss_cheese=num_swiss_cheese,
            num_solid=num_solid,
            instance_masks=instance_masks,
            instance_labels=instance_labels,
        )
    
    def segment_only(self, image: np.ndarray, preprocess: bool = True, 
                     sigma: float = 2.0, sigmoid_cutoff: float = 0.25,
                     verbose: bool = False) -> np.ndarray:
        """
        Run only the segmentation step (no classification).
        
        Args:
            image: Input image
            preprocess: Whether to apply preprocessing
            sigma: Gaussian blur sigma
            sigmoid_cutoff: Sigmoid cutoff
            verbose: If True, display images for debugging
            
        Returns:
            Instance segmentation mask
        """
        if preprocess:
            preprocessed = preprocess_for_segmentation(image, sigma=sigma, sigmoid_cutoff=sigmoid_cutoff)
            if verbose:
                _display_two_images(image, preprocessed, 'Original', 'Preprocessed')
        else:
            preprocessed = image
        
        instance_mask = self.segmenter.segment(preprocessed, ndim=2)
        
        if verbose:
            _display_two_images(preprocessed, instance_mask, 'Preprocessed', f'Segmentation ({instance_mask.max()} objects)')
        
        return instance_mask
