# Inclusion Pipeline

A Python package for microscopy image analysis that segments objects using **MicroSAM** and classifies them as swiss cheese or solid using a fine-tuned **ResNet101** model.

## Installation

**Step 1: Install MicroSAM via conda (required)**

```bash
conda install -c conda-forge micro_sam
```

**Step 2: Install this package**

```bash
pip install git+https://github.com/sunghoojung/segment-classify-pipeline.git
```

> **Note:** MicroSAM must be installed via conda before installing this package. A GPU is recommended for reasonable performance.

## Quick Start

```python
from inclusion_pipeline import SegmentClassifyPipeline
import numpy as np
from PIL import Image

# Initialize the pipeline (downloads model weights automatically)
pipeline = SegmentClassifyPipeline()

# Load your microscopy image
image = np.array(Image.open("cell_image.tif"))

# Run the pipeline
result = pipeline(image)

# Print results
print(f"Swiss Cheese found: {result.num_swiss_cheese}")
print(f"Solid found: {result.num_solid}")
```

## Usage in Jupyter Notebooks

### Basic Usage

```python
from inclusion_pipeline import SegmentClassifyPipeline
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Initialize once (downloads weights on first run)
pipeline = SegmentClassifyPipeline()

# Load image
image = np.array(Image.open("your_image.czi"))

# Run pipeline
result = pipeline(image)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(result.swiss_cheese_mask, cmap='Reds')
axes[1].set_title(f'Swiss Cheese ({result.num_swiss_cheese})')
axes[2].imshow(result.solid_mask, cmap='Greens')
axes[2].set_title(f'Solid ({result.num_solid})')
plt.show()
```

### With Overlay Visualization

```python
from inclusion_pipeline import SegmentClassifyPipeline, overlay_mask_on_image

pipeline = SegmentClassifyPipeline()
result = pipeline(image)

# Create colored overlay
overlay = overlay_mask_on_image(
    image,
    result.combined_mask,
    swiss_cheese_color=(255, 0, 0),      # Red for Swiss Cheese
    solid_color=(0, 255, 0),  # Green for Solid
    alpha=0.4
)

plt.imshow(overlay)
plt.title(f'Swiss Cheese Inclusions: {result.num_swiss_cheese}, Solid Inclusions: {result.num_solid}')
plt.show()
```

### Processing Multiple Images

```python
from inclusion_pipeline import SegmentClassifyPipeline
import os
from pathlib import Path

pipeline = SegmentClassifyPipeline()

image_folder = Path("./images")
results = []

for img_path in image_folder.glob("*.czi"):
    image = np.array(Image.open(img_path))
    result = pipeline(image)
    
    results.append({
        'filename': img_path.name,
        'Swiss Cheese Num': result.num_swiss_cheese,
        'Solid Num': result.num_solid,
    })
    
# Convert to DataFrame
import pandas as pd
df = pd.DataFrame(results)
print(df)
```

### Custom Preprocessing Parameters

```python
# Adjust preprocessing if needed
result = pipeline(
    image,
    preprocess=True,       # Apply gaussian + sigmoid preprocessing
    sigma=2.0,             # Gaussian blur sigma
    sigmoid_cutoff=0.25,   # Sigmoid contrast cutoff
)
```

### Access Individual Object Masks

```python
result = pipeline(image)

# Loop through each detected object
for i, (mask, label) in enumerate(zip(result.instance_masks, result.instance_labels)):
    label_name = "Swiss Cheese" if label == 1 else "Solid"
    area = mask.sum()
    print(f"Object {i}: {label_name}, area = {area} pixels")
```

## Output Structure

The pipeline returns a `PipelineOutput` object with:

| Attribute | Type | Description |
|-----------|------|-------------|
| `swiss_cheese_mask` | `np.ndarray` | Binary mask of swiss cheese |
| `solid_mask` | `np.ndarray` | Binary mask of solid |
| `combined_mask` | `np.ndarray` | 0=background, 1=swiss cheese, 2=solid |
| `num_swiss_cheese` | `int` | Count of detected swiss cheese |
| `num_solid` | `int` | Count of detected solid |
| `instance_masks` | `List[np.ndarray]` | Individual masks for each object |
| `instance_labels` | `List[int]` | Class labels (0 or 1) for each object |

## API Reference

### Main Functions

```python
from inclusion_pipeline import (
    SegmentClassifyPipeline,  # Main pipeline class
    overlay_mask_on_image,    # Create visualization overlay
    colorize_mask,            # Convert mask to colored image
    preprocess_for_segmentation,  # Manual preprocessing
)
```

### Advanced: Using Components Separately

```python
from inclusion_pipeline import MicroSAMSegmenter, ResNetClassifier

# Use just the segmenter
segmenter = MicroSAMSegmenter()
instance_mask = segmenter.segment(preprocessed_image)

# Use just the classifier
classifier = ResNetClassifier()
prediction = classifier.predict(crop)  # 0 or 1
probabilities = classifier.predict_proba(crop)  # [p_solid, p_swiss_cheese]
```

## Requirements

- Python 3.9+
- **MicroSAM** (install via `conda install -c conda-forge micro_sam`)
- PyTorch 2.0+
- CUDA recommended for GPU acceleration

## Model Details

- **Segmentation**: MicroSAM (vit_b_lm) fine-tuned on microscopy data
- **Classification**: ResNet101 fine-tuned with lr=1e-5 for 20 epochs
- **Weights hosted on**: HuggingFace (`sunny17347/machine_learning_models`)
