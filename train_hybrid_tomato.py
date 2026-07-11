from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from hybrid_model import HybridTomatoClassifier, build_eval_transform, build_train_transform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class EpochResult:
    loss: float
    accuracy: float


class ValidImageFolder(datasets.ImageFolder):
    def __init__(self, root: Path, transform=None) -> None:
        super().__init__(root=root, transform=transform)
        valid_samples = []
        for path, target in self.samples:
            try:
                with Image.open(path) as img:
                    img.verify()
                valid_samples.append((path, target))
            except Exception:
                continue
        self.samples = valid_samples
        self.imgs = valid_samples
        self.targets = [target for _, target in valid_samples]


def build_loaders(dataset_root: Path, batch_size: int) -> tuple[dict[str, DataLoader], list[str]]:
    train_transform = build_train_transform()
    eval_transform = build_eval_transform()

    train_dataset = ValidImageFolder(dataset_root / "train", transform=train_transform)
    val_dataset = ValidImageFolder(dataset_root / "val", transform=eval_transform)
    test_dataset = ValidImageFolder(dataset_root / "test", transform=eval_transform)

    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    }
    return loaders, train_dataset.classes


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += labels.size(0)

    return EpochResult(loss=total_loss / total_count, accuracy=total_correct / total_count)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    preds: list[int] = []
    labels_all: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        labels_all.extend(labels.tolist())

    return {
        "accuracy": accuracy_score(labels_all, preds),
        "precision": precision_score(labels_all, preds, average="binary"),
        "recall": recall_score(labels_all, preds, average="binary"),
        "f1_score": f1_score(labels_all, preds, average="binary"),
        "confusion_matrix": confusion_matrix(labels_all, preds).tolist(),
        "predictions": preds,
        "labels": labels_all,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hybrid CNN-Transformer tomato classifier.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hybrid_tomato"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    loaders, class_names = build_loaders(args.data_root, args.batch_size)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = HybridTomatoClassifier(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    history: list[dict[str, float | int]] = []
    best_model_path = output_dir / "best_hybrid_tomato.pt"

    print(f"Using device: {device}")
    print(f"Classes: {class_names}")
    print(f"Train/Val/Test sizes: {[len(loaders[k].dataset) for k in ['train', 'val', 'test']]}")

    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_result = run_epoch(model, loaders["val"], criterion, device, optimizer=None)

        record = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "train_accuracy": train_result.accuracy,
            "val_loss": val_result.loss,
            "val_accuracy": val_result.accuracy,
        }
        history.append(record)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_result.loss:.4f} train_acc={train_result.accuracy:.4f} | "
            f"val_loss={val_result.loss:.4f} val_acc={val_result.accuracy:.4f}"
        )

        if val_result.accuracy > best_val_acc:
            best_val_acc = val_result.accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "history": history,
                },
                best_model_path,
            )

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, loaders["test"], device)

    metrics_payload = {
        "device": str(device),
        "class_names": class_names,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "history": history,
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1_score": test_metrics["f1_score"],
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    cm = np.array(test_metrics["confusion_matrix"], dtype=int)
    np.savetxt(output_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    print("Test metrics:")
    print(json.dumps(metrics_payload["test_metrics"], indent=2))
    print(f"Saved model to: {best_model_path}")
    print(f"Saved metrics to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
