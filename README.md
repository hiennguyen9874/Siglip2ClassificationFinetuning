# Siglip2 Image Classification Fine-tuning

A comprehensive fine-tuning framework for image classification using Google's Siglip2 vision model with advanced training strategies including progressive unfreezing, weighted sampling for imbalanced datasets, and extensive configuration options.

## Features

-   **Siglip2 Model Support**: Fine-tune Google's state-of-the-art Siglip2 vision models for custom image classification tasks
-   **Progressive Unfreezing**: Gradually unfreeze backbone layers during training for better transfer learning
-   **Imbalanced Dataset Handling**: Weighted random sampling to handle class imbalance
-   **Comprehensive Data Augmentation**: Random cropping, rotation, color jitter, sharpness adjustment, and horizontal flips
-   **Mixed Precision Training**: Support for FP16/BF16 automatic mixed precision with auto-detection
-   **Flexible Configuration**: YAML-based configuration with CLI argument overrides
-   **Detailed Evaluation**: Accuracy, macro F1-score, and per-class classification reports
-   **Hugging Face Integration**: Seamless model saving and optional Hub deployment

## Requirements

### System Requirements

-   CUDA-capable GPU (single GPU recommended)
-   Linux/Unix environment (Docker supported)

### Python Dependencies

```bash
pip install torch torchvision transformers datasets pillow scikit-learn evaluate pyyaml numpy
```

Or install from a requirements file:

```txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.14.0
Pillow>=9.5.0
scikit-learn>=1.3.0
evaluate>=0.4.0
pyyaml>=6.0
numpy>=1.24.0
```

## Dataset Preparation

The training script expects datasets in the **ImageFolder** format, which is a standard structure for image classification:

```
data/
└── your-dataset-name/
    ├── class_1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class_2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── class_n/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

### Example Dataset Structure

```
data/
└── helmet-classification-annotated-v2/
    ├── helmet/
    │   ├── img_001.jpg
    │   ├── img_002.jpg
    │   └── ...
    └── no_helmet/
        ├── img_001.jpg
        ├── img_002.jpg
        └── ...
```

**Notes:**

-   Each subdirectory name becomes a class label
-   Supports common image formats: JPG, PNG, BMP, etc.
-   Images are automatically resized to the model's expected input size (typically 256x256)
-   The script automatically handles train/validation splitting if no validation set exists

## Configuration

The project uses a YAML configuration file for managing training parameters. All settings can be overridden via command-line arguments.

### Configuration Parameters

#### Model Configuration

| Parameter                | Type | Default                             | Description                                                |
| ------------------------ | ---- | ----------------------------------- | ---------------------------------------------------------- |
| `model_name`             | str  | `google/siglip2-so400m-patch16-256` | Pretrained Siglip2 model identifier or local path          |
| `freeze_vision_backbone` | bool | `false`                             | Freeze all vision backbone weights (only train classifier) |

#### Data Configuration

| Parameter    | Type  | Default                                     | Description                                               |
| ------------ | ----- | ------------------------------------------- | --------------------------------------------------------- |
| `data_dir`   | str   | `./data/helmet-classification-annotated-v2` | Path to dataset directory in ImageFolder format           |
| `val_split`  | float | `0.1`                                       | Validation split ratio (0.1 = 10% of data for validation) |
| `oversample` | bool  | `true`                                      | Use weighted random sampling for imbalanced datasets      |

#### Training Hyperparameters

| Parameter      | Type  | Default | Description                              |
| -------------- | ----- | ------- | ---------------------------------------- |
| `train_bs`     | int   | `16`    | Training batch size per device           |
| `eval_bs`      | int   | `16`    | Evaluation batch size per device         |
| `epochs`       | int   | `10`    | Number of training epochs                |
| `lr`           | float | `2e-5`  | Learning rate                            |
| `weight_decay` | float | `0.02`  | Weight decay (L2 regularization)         |
| `warmup_ratio` | float | `0.05`  | Warmup ratio for learning rate scheduler |
| `grad_accum`   | int   | `2`     | Gradient accumulation steps              |
| `seed`         | int   | `42`    | Random seed for reproducibility          |

#### Progressive Unfreezing Configuration

| Parameter                    | Type | Default | Description                                |
| ---------------------------- | ---- | ------- | ------------------------------------------ |
| `progressive_unfreezing`     | bool | `true`  | Enable progressive unfreezing strategy     |
| `unfreeze_top_layer_epoch`   | int  | `1`     | Epoch to unfreeze top N backbone layers    |
| `unfreeze_all_epoch`         | int  | `2`     | Epoch to unfreeze all backbone layers      |
| `num_top_layers_to_unfreeze` | int  | `3`     | Number of top layers to unfreeze initially |

#### Mixed Precision Configuration

| Parameter | Type | Default | Description                                          |
| --------- | ---- | ------- | ---------------------------------------------------- |
| `fp16`    | int  | `-1`    | FP16 mixed precision: -1=disabled, 0=auto, 1=enabled |
| `bf16`    | int  | `1`     | BF16 mixed precision: -1=disabled, 0=auto, 1=enabled |

**Note:** BF16 is preferred on Ampere GPUs (A100, RTX 3090+) for better stability. The script auto-detects GPU capability when set to `0`.

#### Logging and Checkpointing

| Parameter          | Type | Default                      | Description                                         |
| ------------------ | ---- | ---------------------------- | --------------------------------------------------- |
| `output_dir`       | str  | `./runs/siglip2-finetune-v2` | Directory for model checkpoints and logs            |
| `logging_steps`    | int  | `50`                         | Log training metrics every N steps                  |
| `eval_strategy`    | str  | `epoch`                      | Evaluation strategy: `no`, `steps`, or `epoch`      |
| `save_strategy`    | str  | `epoch`                      | Checkpoint save strategy: `no`, `steps`, or `epoch` |
| `save_total_limit` | int  | `3`                          | Maximum number of checkpoints to keep               |

#### Hugging Face Hub Configuration

| Parameter      | Type | Default | Description                                           |
| -------------- | ---- | ------- | ----------------------------------------------------- |
| `push_to_hub`  | bool | `false` | Push final model to Hugging Face Hub                  |
| `hub_model_id` | str  | `null`  | Hub model repository ID (e.g., `username/model-name`) |

## Usage

### Basic Training with Config File

```bash
python train.py --config config.yaml
```

### Training with CLI Overrides

Override specific parameters from the command line:

```bash
python train.py --config config.yaml --epochs 20 --lr 1e-5 --train_bs 32
```

### Training without Config File

Specify all parameters via CLI:

```bash
python train.py \
  --data_dir ./data/my-dataset \
  --output_dir ./runs/my-experiment \
  --epochs 15 \
  --train_bs 16 \
  --lr 2e-5 \
  --progressive_unfreezing true
```

### Common Training Scenarios

#### 1. Quick Fine-tuning (Small Dataset, < 1000 images)

```bash
python train.py \
  --config config.yaml \
  --freeze_vision_backbone true \
  --epochs 20 \
  --lr 5e-5
```

#### 2. Full Fine-tuning (Large Dataset, > 10,000 images)

```bash
python train.py \
  --config config.yaml \
  --progressive_unfreezing false \
  --freeze_vision_backbone false \
  --epochs 10 \
  --lr 1e-5
```

#### 3. Progressive Unfreezing (Medium Dataset)

```bash
python train.py \
  --config config.yaml \
  --progressive_unfreezing true \
  --unfreeze_top_layer_epoch 2 \
  --unfreeze_all_epoch 5 \
  --epochs 10
```

#### 4. Imbalanced Dataset Training

```bash
python train.py \
  --config config.yaml \
  --oversample true \
  --weight_decay 0.03
```

## Training Strategies

### Progressive Unfreezing

Progressive unfreezing is a transfer learning technique that gradually unfreezes layers during training:

1. **Phase 1 (Epochs 0-1)**: Only train the classification head
2. **Phase 2 (Epochs 1-2)**: Unfreeze top N layers of the backbone
3. **Phase 3 (Epochs 2+)**: Unfreeze all backbone layers

**Benefits:**

-   Prevents catastrophic forgetting of pretrained features
-   More stable training, especially on small datasets
-   Often achieves better final performance

**When to use:**

-   Small to medium datasets (< 10,000 images)
-   When you want more stable training
-   When pretrained features are highly relevant to your task

**When not to use:**

-   Very large datasets where full fine-tuning works well
-   When your domain is very different from the pretraining data

### Frozen Backbone Training

Train only the classification head while keeping the backbone frozen:

```yaml
freeze_vision_backbone: true
progressive_unfreezing: false
```

**When to use:**

-   Very small datasets (< 500 images)
-   As a baseline before trying progressive unfreezing
-   When you need fast training

### Full Fine-tuning

Unfreeze all layers from the start:

```yaml
freeze_vision_backbone: false
progressive_unfreezing: false
```

**When to use:**

-   Large datasets (> 10,000 images)
-   When your task domain differs significantly from pretraining
-   When you have sufficient compute resources

### Handling Imbalanced Datasets

Enable weighted sampling to handle class imbalance:

```yaml
oversample: true
```

This creates a `WeightedRandomSampler` that oversamples minority classes during training, ensuring the model sees balanced batches.

## Output Files

After training, the following files are saved to `output_dir`:

```
runs/siglip2-finetune-v2/
├── checkpoint-{step}/          # Model checkpoints (kept up to save_total_limit)
│   ├── config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   └── training_args.bin
├── config.json                 # Final model configuration
├── model.safetensors          # Final model weights
├── preprocessor_config.json   # Image processor configuration
├── labels.json                # Label mappings (id2label, label2id)
├── final_metrics.json         # Final evaluation metrics
└── training_args.bin          # Training arguments
```

### labels.json Format

```json
{
    "id2label": {
        "0": "helmet",
        "1": "no_helmet"
    },
    "label2id": {
        "helmet": 0,
        "no_helmet": 1
    }
}
```

### final_metrics.json Format

```json
{
    "eval_loss": 0.1234,
    "eval_accuracy": 0.9567,
    "eval_macro_f1": 0.9523,
    "eval_runtime": 12.34,
    "eval_samples_per_second": 45.67
}
```

## Evaluation Metrics

The training script computes three main metrics:

1. **Accuracy**: Overall classification accuracy across all classes
2. **Macro F1-score**: Average F1-score across classes (gives equal weight to each class, useful for imbalanced datasets)
3. **Classification Report**: Per-class precision, recall, and F1-score

### Understanding the Classification Report

After training, you'll see a detailed classification report:

```
Classification Report:
              precision    recall  f1-score   support

      helmet     0.9645    0.9726    0.9685       219
   no_helmet     0.9718    0.9633    0.9675       218

    accuracy                         0.9680       437
   macro avg     0.9682    0.9679    0.9680       437
weighted avg     0.9682    0.9680    0.9680       437
```

-   **Precision**: Of all predictions for this class, how many were correct?
-   **Recall**: Of all actual instances of this class, how many did we find?
-   **F1-score**: Harmonic mean of precision and recall
-   **Support**: Number of samples in each class

## Inference

### Loading the Trained Model

```python
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_path = "./runs/siglip2-finetune-v2"
model = SiglipForImageClassification.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)

# Load and preprocess image
image = Image.open("path/to/image.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

# Get class label
predicted_label = model.config.id2label[predicted_class_idx]
print(f"Predicted class: {predicted_label}")
```

### Batch Inference

```python
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

# Create dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = ImageFolder("path/to/test/images", transform=transform)
dataloader = DataLoader(dataset, batch_size=32)

# Batch prediction
model.eval()
predictions = []

for batch in dataloader:
    images, labels = batch
    inputs = processor(images=images, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits.argmax(-1)
        predictions.extend(preds.cpu().numpy())
```

## Hyperparameter Tuning Tips

### Learning Rate

-   **Small datasets** (< 1000): Start with `5e-5` to `1e-4`
-   **Medium datasets** (1000-10000): Use `2e-5` to `5e-5`
-   **Large datasets** (> 10000): Try `1e-5` to `2e-5`

### Batch Size

-   Larger batch sizes (32-64) are more stable but may generalize worse
-   Smaller batch sizes (8-16) can generalize better but are noisier
-   Use gradient accumulation to simulate larger batches: `effective_batch_size = train_bs × grad_accum`

### Weight Decay

-   Increases regularization, useful for preventing overfitting
-   Start with `0.01` to `0.03`
-   Increase if you see overfitting (large gap between train and validation accuracy)

### Epochs

-   Monitor validation metrics to detect when to stop
-   Typical range: 10-30 epochs
-   Use early stopping if validation metrics plateau

## Troubleshooting

### Out of Memory (OOM) Errors

1. Reduce batch size: `--train_bs 8`
2. Increase gradient accumulation: `--grad_accum 4`
3. Enable automatic batch size finding: Already enabled via `auto_find_batch_size=True`
4. Use smaller model variant

### Poor Performance on Imbalanced Data

1. Enable weighted sampling: `--oversample true`
2. Monitor macro F1 instead of accuracy: `metric_for_best_model="macro_f1"`
3. Collect more data for minority classes
4. Use data augmentation more aggressively

### Model Not Learning

1. Check learning rate (might be too low or too high)
2. Ensure data is properly preprocessed
3. Verify labels are correct
4. Try unfreezing backbone earlier
5. Check for data leakage or issues

### Overfitting

1. Increase weight decay: `--weight_decay 0.03`
2. Use more aggressive data augmentation
3. Train for fewer epochs
4. Keep backbone frozen or use progressive unfreezing
5. Collect more training data

## Advanced Configuration

### Custom Augmentation Parameters

Edit these constants in `train.py` to customize augmentation:

```python
AUGMENTATION_SCALE = (0.7, 1.0)  # Random crop scale range
ROTATION_DEGREES = 15             # Max rotation angle
COLOR_JITTER_PARAMS = {
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2
}
SHARPNESS_FACTOR = 1.5
SHARPNESS_PROBABILITY = 0.3
HORIZONTAL_FLIP_PROBABILITY = 0.5
```

### Multiple GPU Training

The current implementation is designed for single GPU training. For multi-GPU training, consider:

1. Remove the assertion checking GPU count
2. Use `torch.nn.DataParallel` or `torch.nn.DistributedDataParallel`
3. Adjust learning rate proportionally to the number of GPUs

## Acknowledgments

-   Google Research for the Siglip2 model
-   Hugging Face for the Transformers library
-   PyTorch team for the deep learning framework
