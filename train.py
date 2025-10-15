import argparse
import json
import math
import os
import random
import warnings
from dataclasses import dataclass, fields
from typing import Dict, Optional, Tuple

import evaluate
import numpy as np
import torch
import yaml
from datasets import load_dataset
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomAdjustSharpness,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    Resize,
    ToTensor,
)
from transformers import (
    AutoImageProcessor,
    SiglipForImageClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

assert (
    torch.cuda.device_count() <= 1
), "Multiple GPUs visible. Set CUDA_VISIBLE_DEVICES to a single index."


AUGMENTATION_SCALE = (0.7, 1.0)
ROTATION_DEGREES = 15
COLOR_JITTER_PARAMS = {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2}
SHARPNESS_FACTOR = 1.5
SHARPNESS_PROBABILITY = 0.3
HORIZONTAL_FLIP_PROBABILITY = 0.5


@dataclass
class TrainingConfig:
    model_name: str = "google/siglip2-so400m-patch16-256"
    data_dir: str = "./data/helmet-classification-annotated-v2"
    output_dir: str = "./runs/siglip2-finetune-v2"
    train_bs: int = 16
    eval_bs: int = 16
    val_split: float = 0.1
    epochs: int = 10
    lr: float = 2e-5
    weight_decay: float = 0.02
    warmup_ratio: float = 0.05
    grad_accum: int = 2
    seed: int = 42
    oversample: bool = True
    fp16: int = -1
    bf16: int = 1
    freeze_vision_backbone: bool = False
    progressive_unfreezing: bool = True
    unfreeze_top_layer_epoch: int = 1
    unfreeze_all_epoch: int = 2
    num_top_layers_to_unfreeze: int = 3
    save_total_limit: int = 3
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 50
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def debug_model_architecture(model):
    print("\n=== Siglip2 Model Architecture ===")
    print(f"Model type: {type(model).__name__}")
    print("\nLayer structure:")
    for name, param in model.named_parameters():
        print(
            f"  {name}: shape={list(param.shape)}, "
            f"requires_grad={param.requires_grad}, "
            f"dtype={param.dtype}"
        )
    print("\n" + "=" * 50 + "\n")


def count_trainable_parameters(model) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


class ProgressiveUnfreezingCallback(TrainerCallback):
    def __init__(
        self,
        unfreeze_top_layer_epoch: int = 1,
        unfreeze_all_epoch: int = 2,
        num_top_layers: int = 3,
    ):
        self.unfreeze_top_layer_epoch = unfreeze_top_layer_epoch
        self.unfreeze_all_epoch = unfreeze_all_epoch
        self.num_top_layers = num_top_layers
        self.top_layers_unfrozen = False
        self.all_layers_unfrozen = False

    def on_epoch_begin(self, args, state, control, model, **kwargs):
        current_epoch = int(state.epoch)

        if current_epoch >= self.unfreeze_all_epoch and not self.all_layers_unfrozen:
            print(f"\n[Epoch {current_epoch}] Unfreezing ALL backbone layers...")
            self._unfreeze_all_backbone(model)
            self.all_layers_unfrozen = True
            trainable, total = count_trainable_parameters(model)
            print(
                f"Trainable parameters: {trainable/1e6:.2f}M / {total/1e6:.2f}M "
                f"({100*trainable/total:.1f}%)\n"
            )

        elif (
            current_epoch >= self.unfreeze_top_layer_epoch
            and not self.top_layers_unfrozen
            and not self.all_layers_unfrozen
        ):
            print(
                f"\n[Epoch {current_epoch}] Unfreezing TOP {self.num_top_layers} "
                f"backbone layers..."
            )
            self._unfreeze_top_layers(model, self.num_top_layers)
            self.top_layers_unfrozen = True
            trainable, total = count_trainable_parameters(model)
            print(
                f"Trainable parameters: {trainable/1e6:.2f}M / {total/1e6:.2f}M "
                f"({100*trainable/total:.1f}%)\n"
            )

    def _unfreeze_top_layers(self, model, num_layers: int):
        vision_model = None

        if hasattr(model, "vision_model"):
            vision_model = model.vision_model
        elif hasattr(model, "siglip") and hasattr(model.siglip, "vision_model"):
            vision_model = model.siglip.vision_model

        if (
            vision_model
            and hasattr(vision_model, "encoder")
            and hasattr(vision_model.encoder, "layers")
        ):
            total_layers = len(vision_model.encoder.layers)
            print(f"Total encoder layers: {total_layers}")
            for i in range(max(0, total_layers - num_layers), total_layers):
                layer_idx = i
                print(f"  Unfreezing layer {layer_idx}")
                for param in vision_model.encoder.layers[layer_idx].parameters():
                    param.requires_grad = True

    def _unfreeze_all_backbone(self, model):
        for name, param in model.named_parameters():
            if "classifier" not in name and "score" not in name and "logit_scale" not in name:
                param.requires_grad = True


def detect_amp_dtype() -> str:
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8:
            return "bf16"
        return "fp16"
    return "no"


def build_transforms(processor, size: int) -> Tuple:
    mean, std = processor.image_mean, processor.image_std

    train_transform = Compose(
        [
            RandomResizedCrop(size, scale=AUGMENTATION_SCALE),
            RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROBABILITY),
            RandomRotation(ROTATION_DEGREES),
            ColorJitter(**COLOR_JITTER_PARAMS),
            RandomAdjustSharpness(SHARPNESS_FACTOR, p=SHARPNESS_PROBABILITY),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )

    eval_transform = Compose(
        [
            Resize((size, size)),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )

    return train_transform, eval_transform


def make_set_transform(transform):
    def apply_transform(examples):
        images = []
        for img in examples["image"]:
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")
            else:
                img = img.convert("RGB")
            images.append(transform(img))
        examples["pixel_values"] = images
        return examples

    return apply_transform


def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    labels = torch.tensor([e["label"] for e in examples], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    acc_metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def setup_dataset(config: TrainingConfig) -> Tuple:
    dataset = load_dataset("imagefolder", data_dir=config.data_dir)

    split_name = "train"
    if "train" not in dataset:
        available_splits = list(dataset.keys())
        split_name = available_splits[0]
        dataset = dataset.rename_column(split_name, "train")

    if config.val_split > 0 and "validation" not in dataset:
        split = dataset["train"].train_test_split(
            test_size=config.val_split, stratify_by_column="label", seed=config.seed
        )
        train_ds, val_ds = split["train"], split["test"]
    else:
        train_ds = dataset["train"]
        val_ds = dataset.get("validation", dataset.get("test"))

    features = train_ds.features
    class_names = features["label"].names
    num_labels = len(class_names)
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in enumerate(class_names)}

    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "labels.json"), "w") as f:
        json.dump(
            {"id2label": id2label, "label2id": label2id},
            f,
            indent=2,
            ensure_ascii=False,
        )

    return train_ds, val_ds, num_labels, id2label, label2id, class_names


def setup_model(config: TrainingConfig, num_labels: int, id2label: Dict, label2id: Dict):
    model = SiglipForImageClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    if config.progressive_unfreezing:
        print("Progressive unfreezing enabled - freezing backbone initially...")
        for name, param in model.named_parameters():
            if "classifier" not in name and "score" not in name and "logit_scale" not in name:
                param.requires_grad = False
    elif config.freeze_vision_backbone:
        print("Freezing vision backbone...")
        for name, param in model.named_parameters():
            if "classifier" not in name and "score" not in name and "logit_scale" not in name:
                param.requires_grad = False

    debug_model_architecture(model)

    trainable, total = count_trainable_parameters(model)
    print(
        f"Initial trainable parameters: {trainable/1e6:.2f}M / {total/1e6:.2f}M "
        f"({100*trainable/total:.1f}%)\n"
    )

    return model


def create_weighted_sampler(train_ds, num_labels: int) -> Optional[WeightedRandomSampler]:
    labels = [int(l) for l in train_ds["label"]]
    class_count = np.bincount(labels, minlength=num_labels)
    class_weight = 1.0 / (class_count + 1e-6)
    sample_weight = [class_weight[label] for label in labels]

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weight, dtype=torch.double),
        num_samples=len(sample_weight),
        replacement=True,
    )


def get_mixed_precision_config(config: TrainingConfig) -> Tuple[bool, bool]:
    use_fp16 = False
    use_bf16 = False

    if config.fp16 == 1:
        use_fp16 = True
    elif config.bf16 == 1:
        use_bf16 = True
    elif config.fp16 != 0 and config.bf16 != 0:
        amp = detect_amp_dtype()
        if amp == "bf16":
            use_bf16 = True
        elif amp == "fp16":
            use_fp16 = True

    return use_fp16, use_bf16


def setup_trainer(
    config: TrainingConfig,
    model,
    train_ds,
    eval_ds,
    processor,
    sampler: Optional[WeightedRandomSampler] = None,
) -> Trainer:
    use_fp16, use_bf16 = get_mixed_precision_config(config)

    total_steps_per_epoch = math.ceil(
        len(train_ds) / (config.train_bs * max(1, torch.cuda.device_count()))
    ) // max(config.grad_accum, 1)
    warmup_steps = int(config.warmup_ratio * config.epochs * max(total_steps_per_epoch, 1))

    training_args = TrainingArguments(
        auto_find_batch_size=True,
        output_dir=config.output_dir,
        per_device_train_batch_size=config.train_bs,
        per_device_eval_batch_size=config.eval_bs,
        gradient_accumulation_steps=config.grad_accum,
        num_train_epochs=config.epochs,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=(config.eval_strategy != "no" and config.save_strategy != "no"),
        metric_for_best_model="macro_f1",
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        seed=config.seed,
        eval_accumulation_steps=1,
    )

    callbacks = []
    if config.progressive_unfreezing:
        print(
            f"Adding ProgressiveUnfreezingCallback: "
            f"unfreeze_top_layer_epoch={config.unfreeze_top_layer_epoch}, "
            f"unfreeze_all_epoch={config.unfreeze_all_epoch}, "
            f"num_top_layers={config.num_top_layers_to_unfreeze}"
        )
        callbacks.append(
            ProgressiveUnfreezingCallback(
                unfreeze_top_layer_epoch=config.unfreeze_top_layer_epoch,
                unfreeze_all_epoch=config.unfreeze_all_epoch,
                num_top_layers=config.num_top_layers_to_unfreeze,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
        callbacks=callbacks,
    )

    if sampler is not None:
        original_get_train_dataloader = trainer.get_train_dataloader

        def get_train_dataloader_with_sampler():
            dataloader = original_get_train_dataloader()
            dataloader.sampler = sampler
            return dataloader

        trainer.get_train_dataloader = get_train_dataloader_with_sampler

    return trainer


def print_classification_report(trainer: Trainer, eval_ds, class_names):
    predictions = trainer.predict(eval_ds)
    y_true = predictions.label_ids
    y_pred = predictions.predictions.argmax(1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Fine-tune Siglip2 for image classification")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--model_name",
        "--model-name",
        type=str,
        default=None,
        help="Pretrained model name or path",
    )
    parser.add_argument(
        "--data_dir",
        "--data-dir",
        type=str,
        default=None,
        help="Path to image dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for model and logs",
    )
    parser.add_argument(
        "--train_bs",
        "--train-bs",
        type=int,
        default=None,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--eval_bs",
        "--eval-bs",
        type=int,
        default=None,
        help="Evaluation batch size per device",
    )
    parser.add_argument(
        "--val_split",
        "--val-split",
        type=float,
        default=None,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_ratio",
        "--warmup-ratio",
        type=float,
        default=None,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--grad_accum",
        "--grad-accum",
        type=int,
        default=None,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--oversample",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help="Use weighted random sampler for imbalanced datasets",
    )
    parser.add_argument(
        "--fp16",
        type=int,
        default=None,
        help="Use FP16 mixed precision (-1: disabled, 0: auto, 1: enabled)",
    )
    parser.add_argument(
        "--bf16",
        type=int,
        default=None,
        help="Use BF16 mixed precision (-1: disabled, 0: auto, 1: enabled)",
    )
    parser.add_argument(
        "--freeze_vision_backbone",
        "--freeze-vision-backbone",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help="Freeze vision backbone weights",
    )
    parser.add_argument(
        "--progressive_unfreezing",
        "--progressive-unfreezing",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help="Enable progressive unfreezing of backbone layers",
    )
    parser.add_argument(
        "--unfreeze_top_layer_epoch",
        "--unfreeze-top-layer-epoch",
        type=int,
        default=None,
        help="Epoch at which to unfreeze top layers of backbone",
    )
    parser.add_argument(
        "--unfreeze_all_epoch",
        "--unfreeze-all-epoch",
        type=int,
        default=None,
        help="Epoch at which to unfreeze all backbone layers",
    )
    parser.add_argument(
        "--num_top_layers_to_unfreeze",
        "--num-top-layers-to-unfreeze",
        type=int,
        default=None,
        help="Number of top layers to unfreeze in progressive unfreezing",
    )
    parser.add_argument(
        "--save_total_limit",
        "--save-total-limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--eval_strategy",
        "--eval-strategy",
        type=str,
        default=None,
        help="Evaluation strategy (no, steps, epoch)",
    )
    parser.add_argument(
        "--save_strategy",
        "--save-strategy",
        type=str,
        default=None,
        help="Save strategy (no, steps, epoch)",
    )
    parser.add_argument(
        "--logging_steps",
        "--logging-steps",
        type=int,
        default=None,
        help="Logging frequency in steps",
    )
    parser.add_argument(
        "--push_to_hub",
        "--push-to-hub",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help="Push model to Hugging Face Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        "--hub-model-id",
        type=str,
        default=None,
        help="Hugging Face Hub model ID",
    )

    args = parser.parse_args()

    config_dict = {}

    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f) or {}
        print(f"Loaded config from {args.config}")

    for field in fields(TrainingConfig):
        arg_value = getattr(args, field.name, None)
        if arg_value is not None:
            config_dict[field.name] = arg_value
        elif field.name not in config_dict:
            if field.default is not field.default_factory:
                config_dict[field.name] = field.default
            else:
                config_dict[field.name] = field.default_factory()

    return TrainingConfig(**config_dict)


def main():
    config = parse_args()
    seed_everything(config.seed)

    print(f"Loading dataset from {config.data_dir}...")
    train_ds, val_ds, num_labels, id2label, label2id, class_names = setup_dataset(config)

    print(f"Setting up model: {config.model_name}...")
    model = setup_model(config, num_labels, id2label, label2id)

    processor = AutoImageProcessor.from_pretrained(config.model_name)
    size = processor.size.get("shortest_edge", processor.size.get("height", 256))

    sampler = None
    if config.oversample:
        print("Creating weighted sampler for imbalanced dataset...")
        sampler = create_weighted_sampler(train_ds, num_labels)

    train_transform, eval_transform = build_transforms(processor, size)
    train_ds = train_ds.with_transform(make_set_transform(train_transform))
    val_ds = val_ds.with_transform(make_set_transform(eval_transform))

    print("Setting up trainer...")
    trainer = setup_trainer(config, model, train_ds, val_ds, processor, sampler)

    if config.eval_strategy != "no":
        print("Pre-training evaluation:")
        print(trainer.evaluate())

    print("Starting training...")
    trainer.train()

    print("Post-training evaluation:")
    metrics = trainer.evaluate()
    print(metrics)

    print("Saving model and processor...")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)

    with open(os.path.join(config.output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print_classification_report(trainer, val_ds, class_names)

    if config.push_to_hub:
        print("Pushing model to Hugging Face Hub...")
        trainer.push_to_hub()

    print(f"Training completed! Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
