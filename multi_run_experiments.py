from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from baseline_and_analysis import build_loaders, get_device, set_seed, train_baseline
from hybrid_model import CNNOnlyClassifier, HybridTomatoClassifier, TransformerOnlyClassifier


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_ROOT / "outputs/research_analysis/multi_run_results.json"
SEEDS = [42, 123, 2026]
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 1e-3


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def main() -> None:
    device = get_device()
    model_builders = {
        "cnn": ("CNN", CNNOnlyClassifier),
        "transformer": ("Transformer", TransformerOnlyClassifier),
        "hybrid": ("Hybrid CNN-Transformer", HybridTomatoClassifier),
    }
    results: dict[str, object] = {
        "protocol": {
            "seeds": SEEDS,
            "epochs_per_run": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "loss_function": "Cross-Entropy Loss",
            "device": str(device),
        },
        "models": {},
    }

    for key, (display_name, model_class) in model_builders.items():
        runs = []
        for seed in SEEDS:
            print(f"\n[{display_name}] seed={seed}")
            set_seed(seed)
            loaders, class_names, datasets_map = build_loaders(batch_size=BATCH_SIZE)
            model = model_class(num_classes=len(class_names))
            metrics = train_baseline(
                f"{key}_seed_{seed}",
                model,
                loaders,
                class_names,
                datasets_map["test"],
                device,
                epochs=EPOCHS,
                lr=LEARNING_RATE,
            )
            runs.append(
                {
                    "seed": seed,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "best_val_accuracy": metrics["best_val_accuracy"],
                }
            )
            print(f"test_accuracy={metrics['accuracy']:.4f} f1={metrics['f1_score']:.4f}")

        results["models"][key] = {
            "display_name": display_name,
            "runs": runs,
            "accuracy": summarize([run["accuracy"] for run in runs]),
            "precision": summarize([run["precision"] for run in runs]),
            "recall": summarize([run["recall"] for run in runs]),
            "f1_score": summarize([run["f1_score"] for run in runs]),
        }
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nSaved multi-run results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
