from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from hybrid_model import (
    CNNOnlyClassifier,
    HybridTomatoClassifier,
    TransformerOnlyClassifier,
    build_eval_transform,
    build_train_transform,
)


PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("TOMATO_DATASET_ROOT", PROJECT_ROOT / "data/Tomato_Plant_Stages_Dataset"))
OUTPUT_DIR = PROJECT_ROOT / "outputs/research_analysis"
HYBRID_CKPT = PROJECT_ROOT / "outputs/hybrid_tomato_run1/best_hybrid_tomato.pt"


def set_seed(seed: int = 42) -> None:
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


@dataclass
class EpochResult:
    loss: float
    accuracy: float


def build_loaders(batch_size: int = 8) -> tuple[dict[str, DataLoader], list[str], dict[str, ValidImageFolder]]:
    train_dataset = ValidImageFolder(ROOT / "train", transform=build_train_transform())
    val_dataset = ValidImageFolder(ROOT / "val", transform=build_eval_transform())
    test_dataset = ValidImageFolder(ROOT / "test", transform=build_eval_transform())

    datasets_map = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    loaders = {
        name: DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"), num_workers=0)
        for name, ds in datasets_map.items()
    }
    return loaders, train_dataset.classes, datasets_map


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
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    test_dataset: ValidImageFolder,
) -> dict[str, object]:
    model.eval()
    preds: list[int] = []
    labels_all: list[int] = []
    confidences: list[float] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds.extend(probs.argmax(dim=1).cpu().tolist())
        labels_all.extend(labels.tolist())
        confidences.extend(probs.max(dim=1).values.cpu().tolist())

    report = classification_report(labels_all, preds, target_names=class_names, output_dict=True, zero_division=0)
    misclassified = []
    for idx, (true_idx, pred_idx, conf) in enumerate(zip(labels_all, preds, confidences)):
        if true_idx != pred_idx:
            sample_path = test_dataset.samples[idx][0]
            misclassified.append(
                {
                    "path": sample_path,
                    "filename": Path(sample_path).name,
                    "true_label": class_names[true_idx].replace("_", " ").title(),
                    "predicted_label": class_names[pred_idx].replace("_", " ").title(),
                    "confidence": round(conf * 100, 2),
                }
            )

    return {
        "accuracy": accuracy_score(labels_all, preds),
        "precision": precision_score(labels_all, preds, average="binary", zero_division=0),
        "recall": recall_score(labels_all, preds, average="binary", zero_division=0),
        "f1_score": f1_score(labels_all, preds, average="binary", zero_division=0),
        "confusion_matrix": confusion_matrix(labels_all, preds).tolist(),
        "classification_report": report,
        "misclassified_samples": misclassified,
        "test_size": len(labels_all),
    }


def train_baseline(
    name: str,
    model: nn.Module,
    loaders: dict[str, DataLoader],
    class_names: list[str],
    test_dataset: ValidImageFolder,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
) -> dict[str, object]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    history = []
    best_state = None
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        train_result = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_result = run_epoch(model, loaders["val"], criterion, device, optimizer=None)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "train_accuracy": train_result.accuracy,
                "val_loss": val_result.loss,
                "val_accuracy": val_result.accuracy,
            }
        )
        print(
            f"[{name}] Epoch {epoch}/{epochs} | "
            f"train_acc={train_result.accuracy:.4f} val_acc={val_result.accuracy:.4f}"
        )
        if val_result.accuracy > best_val_acc:
            best_val_acc = val_result.accuracy
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate_model(model, loaders["test"], device, class_names, test_dataset)
    metrics["history"] = history
    metrics["best_val_accuracy"] = best_val_acc
    return metrics


def analyze_hybrid(
    loaders: dict[str, DataLoader],
    class_names: list[str],
    test_dataset: ValidImageFolder,
    device: torch.device,
) -> dict[str, object]:
    ckpt = torch.load(HYBRID_CKPT, map_location=device)
    model = HybridTomatoClassifier(num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    metrics = evaluate_model(model, loaders["test"], device, class_names, test_dataset)
    metrics["history"] = ckpt.get("history", [])
    metrics["best_val_accuracy"] = max((item["val_accuracy"] for item in metrics["history"]), default=None)
    return metrics


def slim_report(report: dict[str, object], class_names: list[str]) -> dict[str, dict[str, float]]:
    result = {}
    for name in class_names:
        row = report[name]
        result[name] = {
            "precision": round(row["precision"], 4),
            "recall": round(row["recall"], 4),
            "f1_score": round(row["f1-score"], 4),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=["cnn", "transformer", "hybrid"])
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "baseline_results.json"

    loaders, class_names, datasets_map = build_loaders(batch_size=8)
    print(f"Using device: {device}")
    print(f"Classes: {class_names}")

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["device"] = str(device)
        summary["class_names"] = class_names
        summary["dataset_summary"] = {
            "train": len(datasets_map["train"]),
            "val": len(datasets_map["val"]),
            "test": len(datasets_map["test"]),
        }
        summary.setdefault("models", {})
    else:
        summary = {
            "device": str(device),
            "class_names": class_names,
            "dataset_summary": {
                "train": len(datasets_map["train"]),
                "val": len(datasets_map["val"]),
                "test": len(datasets_map["test"]),
            },
            "models": {},
        }

    if "cnn" in args.models:
        cnn_metrics = train_baseline(
            "cnn_baseline",
            CNNOnlyClassifier(num_classes=len(class_names)),
            loaders,
            class_names,
            datasets_map["test"],
            device,
            epochs=args.epochs,
        )
        summary["models"]["cnn"] = {
            "display_name": "CNN",
            **{k: cnn_metrics[k] for k in ["accuracy", "precision", "recall", "f1_score", "confusion_matrix", "best_val_accuracy", "misclassified_samples", "history"]},
            "per_class": slim_report(cnn_metrics["classification_report"], class_names),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "transformer" in args.models:
        transformer_metrics = train_baseline(
            "transformer_baseline",
            TransformerOnlyClassifier(num_classes=len(class_names)),
            loaders,
            class_names,
            datasets_map["test"],
            device,
            epochs=args.epochs,
        )
        summary["models"]["transformer"] = {
            "display_name": "Transformer",
            **{k: transformer_metrics[k] for k in ["accuracy", "precision", "recall", "f1_score", "confusion_matrix", "best_val_accuracy", "misclassified_samples", "history"]},
            "per_class": slim_report(transformer_metrics["classification_report"], class_names),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "hybrid" in args.models:
        hybrid_metrics = analyze_hybrid(loaders, class_names, datasets_map["test"], device)
        summary["models"]["hybrid"] = {
            "display_name": "Hybrid CNN-Transformer",
            **{k: hybrid_metrics[k] for k in ["accuracy", "precision", "recall", "f1_score", "confusion_matrix", "best_val_accuracy", "misclassified_samples", "history"]},
            "per_class": slim_report(hybrid_metrics["classification_report"], class_names),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["models"], indent=2))
    print(f"Saved analysis to {summary_path}")


if __name__ == "__main__":
    main()
