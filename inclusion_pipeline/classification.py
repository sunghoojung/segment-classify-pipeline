"""
ResNet classifier for binary classification of segmented objects.

Designed for 16-bit microscopy images (inclusion classification).
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from huggingface_hub import hf_hub_download
from PIL import Image
from typing import Union


class ResNetClassifier:
    """
    ResNet101-based binary classifier for segmented objects.
    
    Trained on 16-bit microscopy images for inclusion classification.
    
    Example:
        >>> classifier = ResNetClassifier()
        >>> pred = classifier.predict(crop)  # 0 = non-inclusion, 1 = inclusion
    """
    
    def __init__(
        self,
        repo_id: str = "sunny17347/machine_learning_models",
        filename: str = "inclusion_classifier_v1.pth",
        device: torch.device = None,
        num_classes: int = 2,
        resnet_variant: str = "resnet101",
        input_size: int = 224,
    ):
        """
        Initialize the ResNet classifier.
        
        Args:
            repo_id: HuggingFace repo ID containing the checkpoint
            filename: Name of the checkpoint file
            device: Torch device
            num_classes: Number of output classes
            resnet_variant: Which ResNet architecture (default: resnet101)
            input_size: Input image size for the model
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.input_size = input_size
        
        # Download checkpoint from HuggingFace
        print(f"Downloading ResNet checkpoint from {repo_id}/{filename}...")
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )
        print(f"Checkpoint downloaded to: {checkpoint_path}")
        
        # Initialize model
        print(f"Initializing {resnet_variant}...")
        self.model = self._build_model(resnet_variant, num_classes)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        print("ResNet classifier initialized!")
        
        # Transform matching training setup: normalized to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ),
        ])
    
    def _build_model(self, variant: str, num_classes: int) -> nn.Module:
        """Build the ResNet model architecture."""
        resnet_builders = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }
        
        if variant not in resnet_builders:
            raise ValueError(f"Unknown ResNet variant: {variant}. Choose from {list(resnet_builders.keys())}")
        
        model = resnet_builders[variant](weights=None)
        
        # Replace the final fully connected layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
        return model
    
    def _preprocess_crop(self, image: np.ndarray) -> Image.Image:
        """
        Preprocess a crop for classification.
        Handles both 8-bit and 16-bit images.
        """
        # Normalize to 0-1 then to 0-255
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            normalized = (image - img_min) / (img_max - img_min)
        else:
            normalized = np.zeros_like(image, dtype=np.float32)
        
        img_8bit = (normalized * 255).astype(np.uint8)
        
        # Handle grayscale vs RGB
        if img_8bit.ndim == 2:
            img_rgb = np.stack([img_8bit, img_8bit, img_8bit], axis=-1)
        elif img_8bit.shape[-1] == 1:
            img_rgb = np.concatenate([img_8bit, img_8bit, img_8bit], axis=-1)
        else:
            img_rgb = img_8bit
        
        return Image.fromarray(img_rgb)
    
    def predict(self, image: Union[np.ndarray, Image.Image]) -> int:
        """
        Predict class for a single image crop.
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, C) or PIL Image
            
        Returns:
            Predicted class index {'solid': 0, 'swiss_cheese': 1}
        """
        # Preprocess
        if isinstance(image, np.ndarray):
            pil_image = self._preprocess_crop(image)
        else:
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            if image.mode == "L":
                image = image.convert("RGB")
            pil_image = image
        
        # Transform and add batch dimension
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.item()
    
    def predict_proba(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Predict class probabilities for a single image.
        
        Args:
            image: Input image as numpy array or PIL Image
            
        Returns:
            Array of class probabilities [non_inclusion_prob, inclusion_prob]
        """
        # Preprocess
        if isinstance(image, np.ndarray):
            pil_image = self._preprocess_crop(image)
        else:
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            if image.mode == "L":
                image = image.convert("RGB")
            pil_image = image
        
        # Transform and add batch dimension
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()[0]
