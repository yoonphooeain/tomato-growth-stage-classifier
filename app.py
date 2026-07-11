from __future__ import annotations

import base64
import csv
import io
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from flask import Flask, Response, abort, redirect, render_template_string, request, send_file, url_for
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from hybrid_model import CNNOnlyClassifier, HybridTomatoClassifier, TransformerOnlyClassifier, build_eval_transform


APP_TITLE = "Tomato Growth Stage Classifier"
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "outputs/hybrid_tomato_run1/best_hybrid_tomato.pt"
METRICS_PATH = PROJECT_ROOT / "outputs/hybrid_tomato_run1/metrics.json"
RESEARCH_PATH = PROJECT_ROOT / "outputs/research_analysis/baseline_results.json"
MULTI_RUN_PATH = PROJECT_ROOT / "outputs/research_analysis/multi_run_results.json"
DATASET_ROOT = Path(os.environ.get("TOMATO_DATASET_ROOT", PROJECT_ROOT / "demo_data"))
LOGO_PATH = PROJECT_ROOT / "assets/ucsy_logo.png"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["class_names"]
model = HybridTomatoClassifier(num_classes=len(class_names)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
transform = build_eval_transform()
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
research_data = json.loads(RESEARCH_PATH.read_text(encoding="utf-8"))


def count_parameters(model_instance: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model_instance.parameters() if parameter.requires_grad)


def measure_inference_latency() -> float:
    sample = torch.zeros(1, 3, 224, 224, device=device)
    timings = []
    with torch.inference_mode():
        model(sample)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(5):
            started = time.perf_counter()
            model(sample)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1000)
    return sum(timings) / len(timings)


model_parameter_counts = {
    "cnn": count_parameters(CNNOnlyClassifier(num_classes=len(class_names))),
    "transformer": count_parameters(TransformerOnlyClassifier(num_classes=len(class_names))),
    "hybrid": count_parameters(model),
}
inference_latency_ms = measure_inference_latency()
model_info = {
    "name": "Hybrid CNN-Transformer",
    "classes": ", ".join(label.replace("_", " ").title() for label in class_names),
    "test_accuracy": f"{metrics['test_metrics']['accuracy'] * 100:.2f}%",
    "f1_score": f"{metrics['test_metrics']['f1_score']:.2f}",
}
model_details = [
    {"label": "Compute Device", "value": str(device).upper(), "note": "Runtime inference device"},
    {"label": "Trainable Parameters", "value": f"{model_parameter_counts['hybrid']:,}", "note": "Hybrid model complexity"},
    {"label": "Checkpoint Size", "value": f"{MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB", "note": "Saved model file"},
    {"label": "Mean Inference Time", "value": f"{inference_latency_ms:.1f} ms", "note": "Batch size 1, five timed runs"},
    {"label": "Input Tensor", "value": "3 x 224 x 224", "note": "RGB normalized image"},
    {"label": "Transformer Tokens", "value": "196 + CLS", "note": "14 x 14 image patches"},
    {"label": "Fusion Vector", "value": "256-D", "note": "128 CNN + 128 Transformer"},
    {"label": "Output Classes", "value": str(len(class_names)), "note": "Early Vegetative / Flowering Initiation"},
]
reproducibility_items = [
    {"label": "Python", "value": platform.python_version(), "note": "Runtime language version"},
    {"label": "PyTorch", "value": torch.__version__, "note": "Deep-learning framework"},
    {"label": "NumPy", "value": np.__version__, "note": "Numerical processing library"},
    {"label": "Device", "value": str(device).upper(), "note": "Training and inference backend"},
    {"label": "Random Seeds", "value": "42, 123, 2026", "note": "Used for repeated experiments"},
    {"label": "Input Size", "value": "224 x 224 RGB", "note": "Model input resolution"},
    {"label": "Normalization", "value": "ImageNet statistics", "note": "Mean and standard deviation"},
    {"label": "Augmentation", "value": "Flip + 10-degree rotation", "note": "Training images only"},
]
reference_items = [
    {
        "authors": "Vaswani et al.",
        "year": "2017",
        "title": "Attention Is All You Need",
        "source": "NeurIPS",
        "url": "https://arxiv.org/abs/1706.03762",
        "relevance": "Introduced the Transformer architecture and self-attention mechanism.",
    },
    {
        "authors": "Dosovitskiy et al.",
        "year": "2020",
        "title": "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "source": "ICLR 2021",
        "url": "https://arxiv.org/abs/2010.11929",
        "relevance": "Established patch-based Transformer learning for image classification.",
    },
    {
        "authors": "He, Zhang, Ren, and Sun",
        "year": "2016",
        "title": "Deep Residual Learning for Image Recognition",
        "source": "CVPR",
        "url": "https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html",
        "relevance": "A foundational CNN study demonstrating deep residual visual feature learning.",
    },
    {
        "authors": "Selvaraju et al.",
        "year": "2017",
        "title": "Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization",
        "source": "ICCV",
        "url": "https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html",
        "relevance": "Supports visual interpretation of influential image regions in CNN predictions.",
    },
    {
        "authors": "Mohanty, Hughes, and Salathe",
        "year": "2016",
        "title": "Using Deep Learning for Image-Based Plant Disease Detection",
        "source": "Frontiers in Plant Science",
        "url": "https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full",
        "relevance": "Demonstrated the practical value and limitations of deep learning on plant images.",
    },
    {
        "authors": "Lee et al.",
        "year": "2018",
        "title": "An Automated, High-Throughput Plant Phenotyping System Using Machine Learning-Based Plant Segmentation and Image Analysis",
        "source": "PLOS ONE",
        "url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196615",
        "relevance": "Connects automated image analysis with plant growth and phenotyping workflows.",
    },
    {
        "authors": "Shorten and Khoshgoftaar",
        "year": "2019",
        "title": "A Survey on Image Data Augmentation for Deep Learning",
        "source": "Journal of Big Data",
        "url": "https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0",
        "relevance": "Provides the methodological basis for augmentation in limited image datasets.",
    },
]
comparison_rows = [
    {
        "model": "CNN",
        "strength": "Strong local feature extraction",
        "limitation": "Weaker global contextual modeling",
        "role": "Baseline for local pattern learning",
    },
    {
        "model": "Transformer",
        "strength": "Strong global contextual learning",
        "limitation": "Can miss fine local visual details",
        "role": "Baseline for global relationship modeling",
    },
    {
        "model": "Hybrid CNN-Transformer",
        "strength": "Combines local and global feature learning",
        "limitation": "More complex than single-model baselines",
        "role": "Proposed model for balanced feature representation",
    },
]
dataset_info = {
    "total_images": "446",
    "train_split": "310 valid images",
    "val_split": "67 images",
    "test_split": "65 valid images",
}
experiment_summary = [
    {"label": "Dataset split", "value": "Train / Validation / Test"},
    {"label": "Image size", "value": "224 x 224"},
    {"label": "Batch size", "value": "8"},
    {"label": "Epochs", "value": str(metrics["epochs"])},
    {"label": "Learning rate", "value": str(metrics["learning_rate"])},
    {"label": "Loss function", "value": "Cross-Entropy Loss"},
    {"label": "Optimizer", "value": "Adam"},
    {"label": "Evaluation", "value": "Accuracy, Precision, Recall, F1-score, Confusion Matrix"},
]
dataset_counts = research_data["dataset_summary"]
dataset_distribution = [
    {"label": "Train", "count": dataset_counts["train"], "color": "#0f6c78"},
    {"label": "Validation", "count": dataset_counts["val"], "color": "#d96d3a"},
    {"label": "Test", "count": dataset_counts["test"], "color": "#88b7bf"},
]
class_descriptions = [
    {
        "name": "Early Vegetative",
        "description": "This stage is characterized by active leaf development, stronger vegetative growth, and the absence of visible flowering structures.",
    },
    {
        "name": "Flowering Initiation",
        "description": "This stage marks the early transition toward flowering, where the plant begins to show visual cues related to reproductive development.",
    },
]
baseline_models = [
    {
        "name": value["display_name"],
        "accuracy": f"{value['accuracy'] * 100:.2f}%",
        "precision": f"{value['precision']:.3f}",
        "recall": f"{value['recall']:.3f}",
        "f1_score": f"{value['f1_score']:.3f}",
        "best_val_accuracy": f"{(value['best_val_accuracy'] or 0) * 100:.2f}%" if value.get("best_val_accuracy") is not None else "-",
        "misclassified_count": len(value["misclassified_samples"]),
    }
    for value in research_data["models"].values()
]
experiment_comparison = []
experiment_architectures = {
    "cnn": "3 convolution blocks",
    "transformer": "2 encoder layers",
    "hybrid": "CNN + Transformer fusion",
}
for key in ["cnn", "transformer", "hybrid"]:
    value = research_data["models"][key]
    experiment_comparison.append(
        {
            "name": value["display_name"],
            "architecture": experiment_architectures[key],
            "epochs": len(value["history"]),
            "parameters": f"{model_parameter_counts[key]:,}",
            "best_val_accuracy": f"{value['best_val_accuracy'] * 100:.2f}%",
            "test_accuracy": f"{value['accuracy'] * 100:.2f}%",
            "f1_score": f"{value['f1_score'] * 100:.2f}%",
            "proposed": key == "hybrid",
        }
    )
hybrid_accuracy = research_data["models"]["hybrid"]["accuracy"]
ablation_rows = []
ablation_components = {
    "cnn": ("CNN branch only", "Local shape, texture, and edge features"),
    "transformer": ("Transformer branch only", "Global relationships between image patches"),
    "hybrid": ("CNN + Transformer", "Fused local and global representations"),
}
for key in ["cnn", "transformer", "hybrid"]:
    value = research_data["models"][key]
    configuration, learned_features = ablation_components[key]
    delta = (value["accuracy"] - hybrid_accuracy) * 100
    ablation_rows.append(
        {
            "configuration": configuration,
            "learned_features": learned_features,
            "epochs": len(value["history"]),
            "accuracy": f"{value['accuracy'] * 100:.2f}%",
            "f1_score": f"{value['f1_score'] * 100:.2f}%",
            "delta": "Reference" if key == "hybrid" else f"{delta:.2f} pp",
            "is_full": key == "hybrid",
        }
    )
baseline_chart_rows = []
per_class_panels = []
for key in ["cnn", "transformer", "hybrid"]:
    value = research_data["models"][key]
    baseline_chart_rows.append(
        {
            "name": value["display_name"],
            "accuracy": f"{value['accuracy'] * 100:.2f}%",
            "accuracy_width": value["accuracy"] * 100,
            "precision": f"{value['precision'] * 100:.2f}%",
            "precision_width": value["precision"] * 100,
            "recall": f"{value['recall'] * 100:.2f}%",
            "recall_width": value["recall"] * 100,
            "f1_score": f"{value['f1_score'] * 100:.2f}%",
            "f1_width": value["f1_score"] * 100,
        }
    )
    per_class_panels.append(
        {
            "name": value["display_name"],
            "classes": [
                {
                    "label": class_name.replace("_", " ").title(),
                    "precision": f"{class_value['precision'] * 100:.2f}%",
                    "precision_width": class_value["precision"] * 100,
                    "recall": f"{class_value['recall'] * 100:.2f}%",
                    "recall_width": class_value["recall"] * 100,
                }
                for class_name, class_value in value["per_class"].items()
            ],
        }
    )
training_history_panels = []
for key in ["cnn", "transformer", "hybrid"]:
    value = research_data["models"][key]
    points = [
        {
            "epoch": item["epoch"],
            "train_accuracy": item["train_accuracy"] * 100,
            "val_accuracy": item["val_accuracy"] * 100,
        }
        for item in value["history"]
    ]
    training_history_panels.append(
        {
            "name": value["display_name"],
            "points": points,
        }
    )
sample_images = [
    {
        "label": "Sample Early Vegetative",
        "path": DATASET_ROOT / "test" / "early_vegetative" / "1709272861381.jpg",
    },
    {
        "label": "Sample Flowering Initiation",
        "path": DATASET_ROOT / "test" / "flowering_initiation" / "1710594999059.jpg",
    },
]


def collect_dataset_gallery() -> list[dict[str, str]]:
    gallery = []
    for split in ["train", "val", "test"]:
        for class_name in ["early_vegetative", "flowering_initiation"]:
            class_dir = DATASET_ROOT / split / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception:
                    continue
                gallery.append(
                    {
                        "split": split,
                        "split_label": "Validation" if split == "val" else split.title(),
                        "class_name": class_name,
                        "class_label": class_name.replace("_", " ").title(),
                        "filename": path.name,
                    }
                )
    return gallery


dataset_gallery = collect_dataset_gallery()
dataset_is_full = all((DATASET_ROOT / split).is_dir() for split in ["train", "val", "test"])
hero_sample = next(
    item for item in dataset_gallery
    if item["split"] == "test" and item["class_name"] == "flowering_initiation"
)
stage_previews = [
    next(
        item for item in dataset_gallery
        if item["split"] == "test"
        and item["class_name"] == class_name
        and (
            class_name == "early_vegetative"
            or item["filename"] == Path(sample_images[1]["path"]).name
        )
    )
    for class_name in ["early_vegetative", "flowering_initiation"]
]
prediction_history: list[dict[str, str]] = []
last_prediction: dict[str, object] | None = None


def load_multi_run_summary() -> dict[str, object]:
    if not MULTI_RUN_PATH.exists():
        return {"rows": [], "protocol": {}, "interpretation": ""}
    data = json.loads(MULTI_RUN_PATH.read_text(encoding="utf-8"))
    rows = []
    for key in ["cnn", "transformer", "hybrid"]:
        value = data.get("models", {}).get(key)
        if not value:
            continue
        rows.append(
            {
                "name": value["display_name"],
                "runs": len(value["runs"]),
                "accuracy": f"{value['accuracy']['mean'] * 100:.2f}% +/- {value['accuracy']['std'] * 100:.2f}",
                "precision": f"{value['precision']['mean'] * 100:.2f}% +/- {value['precision']['std'] * 100:.2f}",
                "recall": f"{value['recall']['mean'] * 100:.2f}% +/- {value['recall']['std'] * 100:.2f}",
                "f1_score": f"{value['f1_score']['mean'] * 100:.2f}% +/- {value['f1_score']['std'] * 100:.2f}",
                "accuracy_mean": value["accuracy"]["mean"],
                "accuracy_std": value["accuracy"]["std"],
            }
        )
    highest_mean = max(rows, key=lambda row: row["accuracy_mean"])["name"] if rows else "-"
    most_stable = min(rows, key=lambda row: row["accuracy_std"])["name"] if rows else "-"
    interpretation = (
        f"{highest_mean} has the highest mean accuracy under the equal three-epoch protocol, "
        f"while {most_stable} has the lowest run-to-run accuracy variation."
    )
    return {"rows": rows, "protocol": data.get("protocol", {}), "interpretation": interpretation}

PAGE_META = {
    "home": {
        "title": "Tomato Growth Stage Classifier",
        "description": "A master thesis portal for hybrid CNN-Transformer research and browser-based tomato growth-stage classification.",
    },
    "research": {
        "title": "Research Methodology",
        "description": "Explore the system workflow, hybrid architecture, experimental setup, ablation evidence, implementation profile, and research limitations.",
    },
    "dataset": {
        "title": "Dataset Explorer",
        "description": "Review the train, validation, and test distribution and inspect representative tomato images by growth-stage class.",
    },
    "results": {
        "title": "Experimental Results",
        "description": "Compare measured baseline performance, learning curves, per-class metrics, confusion patterns, and misclassified samples.",
    },
    "predictor": {
        "title": "Live Growth Stage Predictor",
        "description": "Upload a tomato plant image to obtain its predicted stage, confidence level, uncertainty measurements, and explainable AI heatmap.",
    },
    "reports": {
        "title": "Prediction Reports",
        "description": "Review session history, inspect the hybrid model confusion matrix, and export prediction evidence for thesis demonstration.",
    },
    "references": {
        "title": "Academic References",
        "description": "Review the foundational computer vision, Transformer, explainable AI, and agricultural imaging studies supporting this thesis.",
    },
}

app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <style>
    :root {
      --bg: #f3efe5;
      --bg-soft: #f8f5ee;
      --panel: rgba(255, 255, 255, 0.84);
      --panel-solid: #fffdf8;
      --ink: #1a3551;
      --accent: #0f6c78;
      --accent-2: #d96d3a;
      --accent-3: #88b7bf;
      --soft: #d8e3e3;
      --line: rgba(24, 53, 81, 0.08);
      --shadow: 0 18px 45px rgba(22, 41, 61, 0.10);
      --hero-start: #16344e;
      --hero-mid: #275971;
      --hero-end: #0f6c78;
      --footer-start: #19344f;
      --footer-end: #0f5d69;
      --text-soft: #516e86;
      --reveal-distance: 26px;
    }
    body[data-theme="forest"] {
      --bg: #eef4eb;
      --bg-soft: #f8fbf5;
      --panel: rgba(255, 255, 255, 0.86);
      --panel-solid: #fdfefb;
      --ink: #163c31;
      --accent: #2e7d53;
      --accent-2: #b76e2f;
      --accent-3: #9fc1a8;
      --soft: #d8e5d9;
      --line: rgba(22, 60, 49, 0.08);
      --shadow: 0 18px 45px rgba(24, 56, 42, 0.10);
      --hero-start: #163c31;
      --hero-mid: #2d6a4f;
      --hero-end: #547d5d;
      --footer-start: #14352c;
      --footer-end: #2d6a4f;
      --text-soft: #537162;
    }
    body[data-theme="sunset"] {
      --bg: #f7efe7;
      --bg-soft: #fcf6ef;
      --panel: rgba(255, 255, 255, 0.86);
      --panel-solid: #fffdfa;
      --ink: #4a2b2e;
      --accent: #b5543a;
      --accent-2: #d08a2f;
      --accent-3: #d8b28a;
      --soft: #ead9cd;
      --line: rgba(74, 43, 46, 0.08);
      --shadow: 0 18px 45px rgba(91, 47, 35, 0.10);
      --hero-start: #4d2d36;
      --hero-mid: #9a4f39;
      --hero-end: #cb7a42;
      --footer-start: #4d2d36;
      --footer-end: #8a4837;
      --text-soft: #7a5d57;
    }
    body[data-mode="dark"] {
      --bg: #0d1720;
      --bg-soft: #14222d;
      --panel: rgba(17, 31, 42, 0.82);
      --panel-solid: #152531;
      --ink: #edf7ff;
      --accent: #49a6c8;
      --accent-2: #f09a5a;
      --accent-3: #7dd3c7;
      --soft: rgba(125, 166, 188, 0.18);
      --line: rgba(255, 255, 255, 0.08);
      --shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
      --hero-start: #132838;
      --hero-mid: #16405a;
      --hero-end: #0e6b73;
      --footer-start: #11202d;
      --footer-end: #14394f;
      --text-soft: #c0d7e7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Avenir Next", "Gill Sans", "Trebuchet MS", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(217,109,58,0.10), transparent 26%),
        radial-gradient(circle at top right, rgba(15,108,120,0.12), transparent 30%),
        linear-gradient(180deg, var(--bg-soft), var(--bg));
      color: var(--ink);
    }
    .wrap {
      max-width: 1180px;
      margin: 22px auto 46px;
      padding: 0 22px;
    }
    .topnav {
      display: flex;
      flex-direction: row;
      justify-content: space-between;
      align-items: center;
      gap: 18px;
      padding: 12px 10px 24px;
      margin-bottom: 8px;
      background: rgba(255,255,255,0.55);
      border: 1px solid rgba(255,255,255,0.65);
      border-radius: 24px;
      backdrop-filter: blur(12px);
      box-shadow: 0 10px 28px rgba(22, 41, 61, 0.06);
    }
    html.slideshow-mode,
    body.slideshow-mode {
      overflow: hidden !important;
      height: 100dvh !important;
      max-height: 100dvh !important;
      overscroll-behavior: none !important;
    }
    body.slideshow-mode {
      position: fixed;
      inset: 0;
      width: 100%;
      left: 0;
      right: 0;
    }
    body[data-mode="dark"] .topnav {
      background: rgba(17, 31, 42, 0.74);
      border-color: rgba(255,255,255,0.06);
    }
    body.slideshow-mode .wrap {
      max-width: 100vw;
      margin: 0;
      padding: 0;
      height: 100dvh !important;
      width: 100vw;
      max-height: 100dvh;
      overflow: hidden !important;
      position: fixed;
      inset: 0;
    }
    body.slideshow-mode .topnav {
      position: fixed;
      top: 16px;
      left: 20px;
      right: 20px;
      z-index: 20;
      margin-bottom: 0;
    }
    body.slideshow-mode .reveal,
    body.slideshow-mode .reveal.is-visible {
      opacity: 1;
      transform: none;
    }
    body.slideshow-mode .slide-section,
    body.slideshow-mode .hero,
    body.slideshow-mode footer {
      min-height: 100dvh;
      border-radius: 0;
      margin-top: 0;
      padding: 110px 38px 36px;
      width: 100%;
      overflow: hidden;
    }
    body.slideshow-mode .hero {
      padding-top: 128px;
      border-radius: 0;
    }
    body.slideshow-mode footer {
      padding-top: 128px;
    }
    body.slideshow-mode .slide-section {
      display: none !important;
    }
    body.slideshow-mode .slide-section.active-slide {
      display: block !important;
      position: fixed;
      inset: 0;
      z-index: 5;
      overflow: hidden !important;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 1rem;
      font-weight: 700;
      color: var(--ink);
      letter-spacing: 0.02em;
    }
    .brand img {
      width: 56px;
      height: 56px;
      object-fit: contain;
      border-radius: 16px;
      background: linear-gradient(180deg, #ffffff, #f8f2e8);
      border: 1px solid rgba(26, 53, 81, 0.08);
      padding: 5px;
      box-shadow: 0 8px 16px rgba(26, 53, 81, 0.08);
    }
    .navlinks {
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      font-size: 0.95rem;
      align-items: center;
    }
    .navlinks a {
      color: #516e86;
      text-decoration: none;
      font-weight: 600;
      padding: 9px 12px;
      border-radius: 999px;
      transition: background 180ms ease, color 180ms ease;
    }
    .navlinks a:hover,
    .navlinks a.active {
      color: white;
      background: var(--accent);
    }
    .navlinks button {
      margin-top: 0;
      width: auto;
      padding: 11px 15px;
      font-size: 0.92rem;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--accent), #174f76);
      color: white;
      box-shadow: 0 10px 24px rgba(15,108,120,0.20);
    }
    .mobile-menu-button {
      display: none;
      width: auto;
      margin: 0;
      padding: 9px 13px;
      border-radius: 999px;
      font-size: 0.84rem;
      box-shadow: none;
    }
    .toolbar {
      display: flex;
      justify-content: flex-end;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .theme-switcher {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px;
      border-radius: 999px;
      background: rgba(255,255,255,0.62);
      border: 1px solid rgba(24,53,81,0.08);
    }
    body[data-mode="dark"] .theme-switcher {
      background: rgba(255,255,255,0.05);
      border-color: rgba(255,255,255,0.06);
    }
    .theme-switcher button {
      margin-top: 0;
      width: auto;
      min-width: 88px;
      padding: 10px 12px;
      border-radius: 999px;
      box-shadow: none;
      font-size: 0.88rem;
    }
    .theme-switcher .is-plain {
      background: rgba(255,255,255,0.92);
      color: var(--ink);
      border: 1px solid rgba(24,53,81,0.08);
    }
    body[data-mode="dark"] .theme-switcher .is-plain {
      background: rgba(255,255,255,0.08);
      color: var(--ink);
      border-color: rgba(255,255,255,0.08);
    }
    .hero {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 22px;
      background:
        radial-gradient(circle at top right, rgba(255,255,255,0.16), transparent 28%),
        linear-gradient(135deg, var(--hero-start), var(--hero-mid) 42%, var(--hero-end) 100%);
      color: white;
      border-radius: 34px;
      padding: 26px;
      box-shadow: 0 24px 50px rgba(18, 56, 93, 0.20);
    }
    .page-banner {
      position: relative;
      overflow: hidden;
      margin-bottom: 22px;
      padding: 30px 34px;
      border-radius: 28px;
      background:
        radial-gradient(circle at 85% 20%, rgba(255,255,255,0.17), transparent 24%),
        linear-gradient(130deg, var(--hero-start), var(--hero-mid), var(--hero-end));
      color: white;
      box-shadow: 0 18px 38px rgba(18,56,93,0.17);
    }
    .page-banner span {
      display: block;
      margin-bottom: 8px;
      color: #ccecff;
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }
    .page-banner h1 {
      margin: 0 0 8px;
      font-family: "Palatino Linotype", Georgia, serif;
      font-size: 2.05rem;
    }
    .page-banner p { margin: 0; max-width: 760px; color: #e1f1fb; line-height: 1.65; }
    .slide-section[data-page]:not([data-page="{{ active_page }}"]) { display: none; }
    .hero h1 {
      margin: 0 0 12px;
      font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
      font-size: 2.2rem;
      line-height: 1.08;
      letter-spacing: -0.02em;
    }
    .hero p {
      margin: 0;
      line-height: 1.75;
      max-width: 720px;
      color: #ddecfb;
      font-size: 0.94rem;
    }
    .hero-actions {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-top: 16px;
    }
    .hero-actions a {
      text-decoration: none;
    }
    .hero-button {
      display: inline-block;
      padding: 12px 17px;
      border-radius: 999px;
      font-weight: 700;
      min-width: 148px;
      text-align: center;
    }
    .hero-button.primary {
      background: linear-gradient(180deg, #fffaf1, #ffffff);
      color: #17344e;
      box-shadow: 0 12px 24px rgba(10, 21, 37, 0.14);
    }
    .hero-button.secondary {
      background: rgba(255,255,255,0.10);
      color: white;
      border: 1px solid rgba(255,255,255,0.18);
    }
    .hero-panel {
      position: relative;
      overflow: hidden;
      background: linear-gradient(180deg, rgba(255,255,255,0.13), rgba(255,255,255,0.08));
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 28px;
      padding: 16px;
      align-self: stretch;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.10);
    }
    .hero-plant-image {
      width: 100%;
      height: 100%;
      min-height: 230px;
      max-height: 260px;
      display: block;
      object-fit: cover;
      border-radius: 20px;
      filter: saturate(1.08) contrast(1.02);
    }
    .hero-image-shade {
      position: absolute;
      inset: 16px;
      border-radius: 20px;
      background: linear-gradient(180deg, transparent 38%, rgba(10,31,40,0.82) 100%);
      pointer-events: none;
    }
    .hero-image-caption {
      position: absolute;
      left: 32px;
      right: 32px;
      bottom: 28px;
      color: white;
    }
    .hero-image-caption span {
      display: inline-flex;
      margin-bottom: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.16);
      border: 1px solid rgba(255,255,255,0.18);
      font-size: 0.72rem;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      backdrop-filter: blur(8px);
    }
    .hero-image-caption strong { display: block; font-size: 0.98rem; line-height: 1.35; }
    .hero-badge {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.12);
      color: #e8f5ff;
      font-size: 0.86rem;
      font-weight: 700;
      margin-bottom: 14px;
      border: 1px solid rgba(255,255,255,0.12);
    }
    .hero-badge::before {
      content: "";
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: #8fd1ff;
      box-shadow: 0 0 0 4px rgba(143, 209, 255, 0.18);
    }
    .hero-panel h3 {
      margin: 0 0 14px;
      font-size: 1.05rem;
    }
    .hero-panel ul {
      margin: 0;
      padding-left: 18px;
      line-height: 1.8;
      color: #edf7ff;
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 18px;
      margin-top: 22px;
    }
    .quick-links-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
      margin-top: 22px;
    }
    .quick-link-card {
      position: relative;
      overflow: hidden;
      display: grid;
      grid-template-columns: 44px 1fr auto;
      gap: 12px;
      align-items: center;
      min-height: 92px;
      padding: 15px;
      color: var(--ink);
      text-decoration: none;
      border-radius: 24px;
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.75);
      box-shadow: var(--shadow);
      transition: transform 180ms ease, box-shadow 180ms ease;
    }
    .quick-link-card:hover { transform: translateY(-4px); box-shadow: 0 22px 44px rgba(22,41,61,0.14); }
    .quick-link-icon {
      width: 44px;
      height: 44px;
      display: grid;
      place-items: center;
      border-radius: 18px;
      background: linear-gradient(145deg, var(--accent), #174f76);
      color: white;
      font-family: Georgia, serif;
      font-size: 1.05rem;
      font-weight: 700;
      box-shadow: 0 10px 20px rgba(15,108,120,0.20);
    }
    .quick-link-card:nth-child(2) .quick-link-icon { background: linear-gradient(145deg, var(--accent-2), #b8562c); }
    .quick-link-card:nth-child(3) .quick-link-icon { background: linear-gradient(145deg, #4f87aa, var(--accent)); }
    .quick-link-copy strong { display: block; margin-bottom: 5px; font-size: 1.02rem; }
    .quick-link-copy span { color: var(--text-soft); font-size: 0.83rem; line-height: 1.45; }
    .quick-link-arrow { color: var(--accent); font-size: 1.55rem; font-weight: 800; }
    .stage-preview-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 17px;
    }
    .stage-preview-section { margin-top: 22px; }
    .stage-preview-card {
      position: relative;
      overflow: hidden;
      min-height: 220px;
      border-radius: 24px;
      background: var(--soft);
      border: 1px solid rgba(15,108,120,0.11);
    }
    .stage-preview-card img { width: 100%; height: 220px; object-fit: cover; display: block; }
    .stage-preview-overlay {
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      justify-content: flex-end;
      padding: 17px;
      color: white;
      background: linear-gradient(180deg, transparent 35%, rgba(12,34,43,0.88) 100%);
    }
    .stage-preview-overlay span {
      align-self: flex-start;
      margin-bottom: 8px;
      padding: 6px 9px;
      border-radius: 999px;
      background: rgba(255,255,255,0.16);
      font-size: 0.72rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .stage-preview-overlay h3 { margin: 0 0 5px; font-family: Georgia, serif; font-size: 1.08rem; }
    .stage-preview-overlay p { margin: 0; color: #e4f2f4; line-height: 1.45; font-size: 0.8rem; }
    .section-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 22px;
      margin-top: 22px;
    }
    .overview-grid {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 22px;
      margin-top: 22px;
    }
    .research-grid {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 22px;
      margin-top: 22px;
    }
    .thesis-meta {
      display: grid;
      gap: 10px;
      margin-top: 16px;
    }
    .thesis-meta div {
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.12);
    }
    .thesis-meta span {
      display: block;
      color: #cae8ff;
      font-size: 0.8rem;
      margin-bottom: 5px;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 22px;
      margin-top: 24px;
    }
    .card {
      position: relative;
      overflow: hidden;
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.75);
      border-radius: 26px;
      padding: 24px;
      backdrop-filter: blur(12px);
      box-shadow: var(--shadow);
    }
    .card::before {
      content: "";
      position: absolute;
      inset: 0 auto auto 0;
      width: 100%;
      height: 5px;
      background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent-3));
      opacity: 0.75;
    }
    .reveal {
      opacity: 0;
      transform: translateY(var(--reveal-distance));
      transition: opacity 700ms ease, transform 700ms ease;
      will-change: opacity, transform;
    }
    .reveal.is-visible {
      opacity: 1;
      transform: translateY(0);
    }
    .card h2 {
      margin-top: 0;
      margin-bottom: 14px;
      font-family: "Palatino Linotype", Georgia, serif;
      font-size: 1.38rem;
      letter-spacing: -0.01em;
    }
    .eyebrow {
      display: inline-block;
      margin-bottom: 10px;
      font-size: 0.8rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #6b8f96;
      font-weight: 700;
    }
    .upload-box {
      border: 2px dashed rgba(15,108,120,0.22);
      border-radius: 22px;
      padding: 22px;
      background: linear-gradient(180deg, #fcfdfa, #f5fbfb);
    }
    input[type=file] {
      width: 100%;
      padding: 14px;
      border: 1px solid rgba(15,108,120,0.14);
      border-radius: 14px;
      background: rgba(255,255,255,0.92);
      color: var(--ink);
    }
    button {
      margin-top: 16px;
      border: 0;
      border-radius: 16px;
      padding: 14px 18px;
      width: 100%;
      background: linear-gradient(135deg, var(--accent), #1e6072 50%, #174f76);
      color: white;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 14px 24px rgba(15,108,120,0.18);
    }
    .preview {
      width: 100%;
      border-radius: 22px;
      border: 1px solid rgba(15,108,120,0.14);
      margin-top: 16px;
      background: white;
      box-shadow: inset 0 0 0 8px rgba(250, 252, 248, 0.85);
    }
    .stack {
      display: grid;
      gap: 22px;
    }
    .wide-stack {
      display: grid;
      gap: 22px;
      margin-top: 22px;
    }
    .mini-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 14px;
    }
    .metric {
      padding: 14px;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.92), #f5faf8);
      border: 1px solid rgba(15,108,120,0.10);
    }
    .metric span {
      display: block;
      color: var(--text-soft);
      font-size: 0.85rem;
      margin-bottom: 6px;
    }
    .metric strong {
      font-size: 1.1rem;
    }
    .workflow {
      display: grid;
      gap: 12px;
      margin-top: 16px;
    }
    .workflow-step {
      display: grid;
      grid-template-columns: 38px 1fr;
      gap: 14px;
      align-items: start;
      padding: 14px 16px;
      border-radius: 18px;
      background: linear-gradient(180deg, #fbfdfa, #f5faf9);
      border: 1px solid rgba(15,108,120,0.10);
    }
    .workflow-step b {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 38px;
      height: 38px;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--accent-2), #b8562c);
      color: white;
      font-size: 0.95rem;
      box-shadow: 0 10px 16px rgba(217,109,58,0.20);
    }
    .workflow-step strong {
      display: block;
      margin-bottom: 4px;
    }
    .bullet-list {
      margin: 0;
      padding-left: 18px;
      color: var(--text-soft);
      line-height: 1.8;
    }
    .class-grid {
      display: grid;
      gap: 14px;
      margin-top: 14px;
    }
    .class-card {
      padding: 16px 18px;
      border-radius: 20px;
      background: linear-gradient(180deg, #fffdf8, #f7faf7);
      border: 1px solid rgba(15,108,120,0.10);
    }
    .class-card strong {
      display: block;
      margin-bottom: 6px;
      color: #12385d;
    }
    .compare-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      font-size: 0.95rem;
      overflow: hidden;
      border-radius: 18px;
    }
    .baseline-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      font-size: 0.95rem;
    }
    .baseline-table th,
    .baseline-table td {
      text-align: left;
      padding: 12px 10px;
      border-bottom: 1px solid rgba(15,108,120,0.10);
    }
    .baseline-table th {
      color: var(--text-soft);
      font-size: 0.84rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      background: rgba(15,108,120,0.05);
    }
    .chart-stack {
      display: grid;
      gap: 14px;
      margin-top: 16px;
    }
    .chart-row {
      padding: 16px 18px;
      border-radius: 20px;
      background: linear-gradient(180deg, #fffefb, #f6faf9);
      border: 1px solid rgba(15,108,120,0.10);
    }
    .chart-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
    }
    .chart-head strong {
      font-size: 1rem;
      color: #12385d;
    }
    .metric-bars {
      display: grid;
      gap: 10px;
    }
    .metric-bar-row {
      display: grid;
      grid-template-columns: 92px 1fr 64px;
      gap: 10px;
      align-items: center;
    }
    .metric-bar-row span {
      font-size: 0.88rem;
      color: var(--text-soft);
    }
    .metric-track {
      height: 12px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(15,108,120,0.10);
    }
    .metric-fill {
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--accent), #2d8d96, var(--accent-2));
      box-shadow: 0 8px 16px rgba(15,108,120,0.16);
    }
    .metric-fill.alt {
      background: linear-gradient(90deg, var(--accent-2), #e19d52, #efc372);
      box-shadow: 0 8px 16px rgba(217,109,58,0.16);
    }
    .metric-fill.soft {
      background: linear-gradient(90deg, #5f97bf, #75b7ca, var(--accent-3));
      box-shadow: 0 8px 16px rgba(95,151,191,0.16);
    }
    .perclass-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
      margin-top: 16px;
    }
    .perclass-card {
      padding: 16px;
      border-radius: 20px;
      background: linear-gradient(180deg, #fffefb, #f6faf9);
      border: 1px solid rgba(15,108,120,0.10);
    }
    .perclass-card h3 {
      margin: 0 0 12px;
      font-size: 1.02rem;
      font-family: "Palatino Linotype", Georgia, serif;
    }
    .perclass-item + .perclass-item {
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid rgba(15,108,120,0.08);
    }
    .perclass-item strong {
      display: block;
      margin-bottom: 10px;
      color: #12385d;
    }
    .compare-table th,
    .compare-table td {
      text-align: left;
      vertical-align: top;
      padding: 12px 10px;
      border-bottom: 1px solid rgba(15,108,120,0.10);
    }
    .compare-table th {
      color: var(--text-soft);
      font-size: 0.84rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      background: rgba(15,108,120,0.05);
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }
    .summary-item {
      padding: 14px 16px;
      border-radius: 18px;
      background: linear-gradient(180deg, #fffefb, #f6faf8);
      border: 1px solid rgba(15,108,120,0.10);
    }
    .summary-item span {
      display: block;
      color: var(--text-soft);
      font-size: 0.82rem;
      margin-bottom: 6px;
    }
    .architecture-shell {
      margin-top: 18px;
      padding: 24px;
      border-radius: 24px;
      background:
        radial-gradient(circle at 15% 0%, rgba(217,109,58,0.10), transparent 30%),
        linear-gradient(135deg, rgba(15,108,120,0.06), rgba(255,255,255,0.62));
      border: 1px solid rgba(15,108,120,0.12);
    }
    .architecture-flow {
      display: grid;
      grid-template-columns: 0.8fr 42px 1.5fr 42px 0.9fr 42px 0.9fr 42px 0.9fr;
      align-items: center;
      gap: 8px;
    }
    .architecture-node {
      min-height: 112px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      padding: 16px;
      border-radius: 20px;
      background: var(--panel-solid);
      border: 1px solid rgba(15,108,120,0.14);
      box-shadow: 0 12px 24px rgba(22,41,61,0.08);
      text-align: center;
    }
    .architecture-node strong {
      display: block;
      color: var(--ink);
      font-size: 0.96rem;
      margin-bottom: 7px;
    }
    .architecture-node span {
      color: var(--text-soft);
      font-size: 0.8rem;
      line-height: 1.45;
    }
    .architecture-node.input-node,
    .architecture-node.output-node {
      color: white;
      background: linear-gradient(145deg, var(--hero-start), var(--accent));
      border-color: transparent;
    }
    .architecture-node.input-node strong,
    .architecture-node.input-node span,
    .architecture-node.output-node strong,
    .architecture-node.output-node span {
      color: white;
    }
    .architecture-node.fusion-node {
      background: linear-gradient(145deg, rgba(217,109,58,0.14), var(--panel-solid));
      border-color: rgba(217,109,58,0.24);
    }
    .branch-stack {
      display: grid;
      gap: 12px;
    }
    .branch-stack .architecture-node {
      min-height: 104px;
      text-align: left;
      border-left: 5px solid var(--accent);
    }
    .branch-stack .architecture-node:last-child {
      border-left-color: var(--accent-2);
    }
    .flow-arrow {
      text-align: center;
      color: var(--accent);
      font-size: 1.8rem;
      font-weight: 800;
    }
    .architecture-legend {
      display: flex;
      justify-content: center;
      gap: 22px;
      flex-wrap: wrap;
      margin-top: 18px;
      color: var(--text-soft);
      font-size: 0.84rem;
    }
    .architecture-legend span::before {
      content: "";
      display: inline-block;
      width: 9px;
      height: 9px;
      border-radius: 999px;
      margin-right: 7px;
      background: var(--accent);
    }
    .architecture-legend span:nth-child(2)::before { background: var(--accent-2); }
    .architecture-legend span:nth-child(3)::before { background: var(--accent-3); }
    .table-scroll {
      overflow-x: auto;
      border-radius: 18px;
      border: 1px solid rgba(15,108,120,0.10);
      margin-top: 16px;
    }
    .experiment-table {
      width: 100%;
      min-width: 860px;
      border-collapse: collapse;
      font-size: 0.92rem;
      background: var(--panel-solid);
    }
    .experiment-table th,
    .experiment-table td {
      padding: 14px 13px;
      text-align: left;
      border-bottom: 1px solid rgba(15,108,120,0.10);
    }
    .experiment-table th {
      color: var(--text-soft);
      background: rgba(15,108,120,0.06);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.045em;
    }
    .experiment-table tr:last-child td { border-bottom: 0; }
    .experiment-table .proposed-row {
      background: linear-gradient(90deg, rgba(15,108,120,0.10), rgba(217,109,58,0.07));
    }
    .model-chip {
      display: inline-flex;
      padding: 5px 9px;
      border-radius: 999px;
      margin-left: 7px;
      background: var(--accent);
      color: white;
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .protocol-strip {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }
    .protocol-strip div {
      padding: 11px 13px;
      border-radius: 14px;
      background: rgba(15,108,120,0.06);
      color: var(--text-soft);
      font-size: 0.82rem;
    }
    .protocol-strip strong { color: var(--ink); }
    .details-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }
    .detail-card {
      position: relative;
      min-height: 132px;
      padding: 17px;
      border-radius: 20px;
      background: linear-gradient(155deg, var(--panel-solid), rgba(15,108,120,0.06));
      border: 1px solid rgba(15,108,120,0.11);
    }
    .detail-card::after {
      content: "";
      position: absolute;
      width: 34px;
      height: 4px;
      top: 0;
      left: 17px;
      border-radius: 0 0 4px 4px;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }
    .detail-card span,
    .detail-card small {
      display: block;
      color: var(--text-soft);
    }
    .detail-card span { font-size: 0.8rem; }
    .detail-card strong {
      display: block;
      margin: 9px 0 7px;
      color: var(--ink);
      font-size: 1.18rem;
    }
    .detail-card small { font-size: 0.76rem; line-height: 1.4; }
    .ablation-summary {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }
    .ablation-stat {
      padding: 17px;
      border-radius: 19px;
      background: linear-gradient(155deg, var(--panel-solid), rgba(15,108,120,0.06));
      border: 1px solid rgba(15,108,120,0.11);
    }
    .ablation-stat span,
    .ablation-stat small { display: block; color: var(--text-soft); }
    .ablation-stat strong {
      display: block;
      margin: 7px 0;
      color: var(--ink);
      font-size: 1.35rem;
    }
    .ablation-stat.full {
      background: linear-gradient(145deg, rgba(15,108,120,0.13), rgba(217,109,58,0.08));
      border-color: rgba(15,108,120,0.22);
    }
    .dataset-controls {
      display: grid;
      grid-template-columns: 1fr 1fr auto;
      gap: 12px;
      align-items: end;
      margin-top: 18px;
      padding: 16px;
      border-radius: 20px;
      background: rgba(15,108,120,0.06);
      border: 1px solid rgba(15,108,120,0.10);
    }
    .filter-field label {
      display: block;
      margin-bottom: 7px;
      color: var(--text-soft);
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .filter-field select {
      width: 100%;
      min-height: 44px;
      padding: 10px 12px;
      border-radius: 13px;
      border: 1px solid rgba(15,108,120,0.16);
      background: var(--panel-solid);
      color: var(--ink);
      font: inherit;
    }
    .dataset-count {
      min-width: 130px;
      padding: 12px 14px;
      border-radius: 14px;
      background: var(--accent);
      color: white;
      text-align: center;
      font-weight: 700;
    }
    .dataset-gallery {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: 16px;
    }
    .dataset-pagination {
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 12px;
      margin-top: 16px;
    }
    .dataset-pagination button {
      width: auto;
      min-width: 110px;
      margin: 0;
      padding: 10px 15px;
      border-radius: 13px;
      font-size: 0.86rem;
      box-shadow: none;
    }
    .dataset-pagination button:disabled { opacity: 0.42; cursor: not-allowed; }
    .dataset-page-info {
      min-width: 110px;
      color: var(--text-soft);
      text-align: center;
      font-size: 0.84rem;
      font-weight: 700;
    }
    .dataset-sample {
      overflow: hidden;
      border-radius: 19px;
      background: var(--panel-solid);
      border: 1px solid rgba(15,108,120,0.11);
      box-shadow: 0 10px 22px rgba(22,41,61,0.07);
    }
    .dataset-sample[hidden] { display: none; }
    .dataset-sample img {
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      display: block;
      background: var(--soft);
    }
    .dataset-sample-info { padding: 11px 12px 13px; }
    .dataset-sample-info strong,
    .dataset-sample-info span { display: block; }
    .dataset-sample-info strong { font-size: 0.88rem; margin-bottom: 4px; }
    .dataset-sample-info span { color: var(--text-soft); font-size: 0.76rem; }
    .confidence-panel {
      margin: 16px 0;
      padding: 17px;
      border-radius: 20px;
      border: 1px solid rgba(15,108,120,0.12);
      background: linear-gradient(145deg, rgba(15,108,120,0.08), var(--panel-solid));
    }
    .confidence-panel.medium {
      border-color: rgba(217,109,58,0.28);
      background: linear-gradient(145deg, rgba(217,109,58,0.11), var(--panel-solid));
    }
    .confidence-panel.low {
      border-color: rgba(180,35,24,0.26);
      background: linear-gradient(145deg, rgba(180,35,24,0.09), var(--panel-solid));
    }
    .confidence-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }
    .confidence-badge {
      display: inline-flex;
      padding: 7px 11px;
      border-radius: 999px;
      background: var(--accent);
      color: white;
      font-size: 0.76rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .confidence-panel.medium .confidence-badge { background: var(--accent-2); }
    .confidence-panel.low .confidence-badge { background: #b42318; }
    .uncertainty-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .uncertainty-grid div {
      padding: 11px 12px;
      border-radius: 14px;
      background: rgba(255,255,255,0.48);
      border: 1px solid rgba(15,108,120,0.08);
    }
    .uncertainty-grid span,
    .uncertainty-grid strong { display: block; }
    .uncertainty-grid span { color: var(--text-soft); font-size: 0.74rem; margin-bottom: 4px; }
    .uncertainty-note { margin-top: 11px; color: var(--text-soft); line-height: 1.55; font-size: 0.86rem; }
    .quality-panel {
      margin-top: 14px;
      padding: 14px;
      border-radius: 17px;
      background: rgba(15,108,120,0.07);
      border: 1px solid rgba(15,108,120,0.13);
    }
    .quality-panel.review { background: rgba(217,109,58,0.09); border-color: rgba(217,109,58,0.22); }
    .quality-panel.reject { background: rgba(180,35,24,0.08); border-color: rgba(180,35,24,0.20); }
    .quality-head { display: flex; justify-content: space-between; gap: 12px; align-items: center; }
    .quality-status {
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--accent);
      color: white;
      font-size: 0.72rem;
      font-weight: 800;
      text-transform: uppercase;
    }
    .quality-panel.review .quality-status { background: var(--accent-2); }
    .quality-panel.reject .quality-status { background: #b42318; }
    .quality-checks { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 8px; margin-top: 11px; }
    .quality-checks div { padding: 8px 10px; border-radius: 12px; background: rgba(255,255,255,0.52); }
    .quality-checks span,.quality-checks strong { display:block; }
    .quality-checks span { color:var(--text-soft); font-size:0.72rem; }
    .quality-checks strong { margin-top:3px; font-size:0.86rem; }
    .limitation-alert {
      margin-top: 12px;
      padding: 13px 15px;
      border-radius: 16px;
      background: rgba(217,109,58,0.09);
      border-left: 4px solid var(--accent-2);
      color: var(--text-soft);
      line-height: 1.55;
      font-size: 0.84rem;
    }
    .download-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; margin-top:14px; }
    .download-grid a { text-decoration:none; }
    .download-card { display:block; padding:14px; border-radius:16px; background:var(--panel-solid); border:1px solid rgba(15,108,120,0.12); color:var(--ink); }
    .download-card strong,.download-card span { display:block; }
    .download-card span { margin-top:5px; color:var(--text-soft); font-size:0.8rem; }
    .reference-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin-top:14px; }
    .reference-card { padding:16px; border-radius:18px; background:var(--panel-solid); border:1px solid rgba(15,108,120,0.11); }
    .reference-card span { color:var(--accent); font-size:0.76rem; font-weight:800; text-transform:uppercase; letter-spacing:0.04em; }
    .reference-card h3 { margin:7px 0; font-family:Georgia,serif; font-size:1.02rem; }
    .reference-card p { margin:0 0 10px; color:var(--text-soft); font-size:0.82rem; line-height:1.5; }
    .reference-card a { color:var(--accent); font-size:0.8rem; font-weight:700; }
    .viz-grid {
      display: grid;
      grid-template-columns: 0.9fr 1.1fr;
      gap: 22px;
      margin-top: 22px;
    }
    .dataset-viz {
      display: grid;
      justify-items: center;
      gap: 14px;
    }
    .dataset-viz img {
      width: 100%;
      max-width: 420px;
      background: white;
      border-radius: 22px;
      border: 1px solid rgba(15,108,120,0.10);
      box-shadow: 0 14px 30px rgba(22, 41, 61, 0.08);
    }
    .dataset-legend {
      display: grid;
      gap: 10px;
      width: 100%;
    }
    .dataset-legend-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 12px 14px;
      border-radius: 16px;
      background: linear-gradient(180deg, #fffefb, #f6faf9);
      border: 1px solid rgba(15,108,120,0.10);
    }
    .dataset-legend-label {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      color: #12385d;
      font-weight: 700;
    }
    .dataset-dot {
      width: 14px;
      height: 14px;
      border-radius: 999px;
    }
    .history-grid {
      display: grid;
      gap: 16px;
      margin-top: 16px;
    }
    .history-card {
      padding: 14px;
      border-radius: 20px;
      background: linear-gradient(180deg, #fffefb, #f6faf9);
      border: 1px solid rgba(15,108,120,0.10);
    }
    .history-card img {
      width: 100%;
      border-radius: 18px;
      background: white;
      border: 1px solid rgba(15,108,120,0.10);
    }
    .slideshow-tip {
      margin-top: 14px;
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(15,108,120,0.06);
      border: 1px solid rgba(15,108,120,0.10);
      color: var(--text-soft);
      line-height: 1.6;
    }
    .slide-badge {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 9px 14px;
      border-radius: 999px;
      background: rgba(15,108,120,0.08);
      color: var(--accent);
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 12px;
    }
    .limit-card {
      padding: 16px 18px;
      border-radius: 20px;
      background: linear-gradient(180deg, #fffdf8, #f8faf9);
      border: 1px solid rgba(15,108,120,0.10);
      margin-top: 14px;
    }
    .limit-card strong {
      display: block;
      color: #12385d;
      margin-bottom: 8px;
    }
    .history-list {
      margin: 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 12px;
    }
    .history-actions {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 16px;
    }
    .history-actions form {
      margin: 0;
    }
    .action-button {
      margin-top: 0;
      width: auto;
      min-width: 210px;
    }
    .action-button.secondary {
      background: white;
      color: var(--ink);
      border: 1px solid #bfd7ea;
    }
    .action-button.ghost {
      background: #eaf5ff;
      color: var(--ink);
    }
    .history-item {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 16px;
      border-radius: 18px;
      background: linear-gradient(180deg, #fffefb, #f6faf9);
      border: 1px solid rgba(15,108,120,0.10);
    }
    .history-item strong {
      display: block;
      margin-bottom: 4px;
    }
    .history-item span {
      color: var(--text-soft);
      font-size: 0.92rem;
    }
    .mis-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
      margin-top: 14px;
    }
    .mis-card {
      padding: 16px;
      border-radius: 20px;
      background: linear-gradient(180deg, #fffefb, #f6faf9);
      border: 1px solid rgba(15,108,120,0.10);
    }
    .mis-card h3 {
      margin: 0 0 8px;
      font-size: 1.02rem;
      font-family: "Palatino Linotype", Georgia, serif;
    }
    .mis-card img {
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      border-radius: 16px;
      border: 1px solid rgba(15,108,120,0.10);
      margin-top: 10px;
    }
    .mis-meta {
      margin-top: 8px;
      font-size: 0.9rem;
      color: var(--text-soft);
      line-height: 1.5;
    }
    .sample-buttons {
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }
    .sample-buttons button {
      margin-top: 0;
      background: linear-gradient(180deg, #ffffff, #f7faf8);
      color: var(--ink);
      border: 1px solid rgba(15,108,120,0.12);
      box-shadow: none;
    }
    .result {
      padding: 22px;
      border-radius: 24px;
      background:
        radial-gradient(circle at top right, rgba(217,109,58,0.08), transparent 25%),
        linear-gradient(180deg, #fffef9, #f3faf8);
      border: 1px solid rgba(15,108,120,0.12);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.65);
    }
    .result strong {
      display: block;
      color: var(--accent);
      font-size: 1.55rem;
      margin-top: 8px;
      letter-spacing: -0.01em;
    }
    .prob-list {
      margin: 14px 0 0;
      padding: 0;
      list-style: none;
    }
    .prob-list li {
      padding: 12px 0;
      border-bottom: 1px solid rgba(15,108,120,0.10);
    }
    .prob-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }
    .bar {
      width: 100%;
      height: 12px;
      background: #e6efef;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), #2d8d96, var(--accent-2));
      border-radius: 999px;
    }
    .explain {
      margin-top: 16px;
      padding: 14px 16px;
      background: rgba(255,255,255,0.78);
      border-radius: 18px;
      border: 1px solid rgba(15,108,120,0.10);
    }
    .heatmap {
      margin-top: 18px;
      padding: 18px;
      border-radius: 20px;
      background:
        radial-gradient(circle at top right, rgba(15,108,120,0.08), transparent 28%),
        linear-gradient(180deg, rgba(255,255,255,0.90), rgba(246,250,249,0.96));
      border: 1px solid rgba(15,108,120,0.10);
    }
    .heatmap-header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 10px;
    }
    .heatmap-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(15,108,120,0.08);
      color: var(--accent);
      font-size: 0.8rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .heatmap-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-top: 14px;
    }
    .heatmap-figure {
      padding: 12px;
      border-radius: 18px;
      background: rgba(255,255,255,0.78);
      border: 1px solid rgba(15,108,120,0.10);
    }
    .heatmap-figure span {
      display: block;
      margin-bottom: 8px;
      color: var(--text-soft);
      font-size: 0.84rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .heatmap img {
      width: 100%;
      border-radius: 18px;
      border: 1px solid rgba(15,108,120,0.10);
      margin-top: 0;
      background: white;
    }
    .presentation-note {
      margin-top: 12px;
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(15,108,120,0.06);
      border: 1px solid rgba(15,108,120,0.10);
      color: var(--text-soft);
      line-height: 1.6;
    }
    .matrix-panel {
      text-align: center;
    }
    .matrix-panel img {
      width: 100%;
      max-width: 360px;
      border-radius: 20px;
      border: 1px solid rgba(15,108,120,0.12);
      background: white;
      margin-top: 10px;
      box-shadow: 0 12px 24px rgba(22, 41, 61, 0.08);
    }
    footer {
      margin-top: 24px;
      padding: 20px 24px;
      border-radius: 26px;
      background: linear-gradient(135deg, #19344f, #0f5d69);
      color: #e7f4ff;
      box-shadow: 0 18px 34px rgba(18, 56, 93, 0.16);
    }
    .footer-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 16px;
    }
    .footer-grid span {
      display: block;
      color: #a9d2f3;
      font-size: 0.82rem;
      margin-bottom: 6px;
    }
    .muted {
      color: var(--text-soft);
      line-height: 1.6;
    }
    .error {
      color: #b42318;
      background: #fff3f2;
      border: 1px solid #f3c4bf;
      padding: 12px 14px;
      border-radius: 12px;
      margin-top: 14px;
    }
    body:not([data-page="home"]) .wrap {
      max-width: 1120px;
      margin-top: 14px;
      margin-bottom: 28px;
    }
    body:not([data-page="home"]) .topnav {
      padding: 9px 10px 14px;
      border-radius: 20px;
    }
    body:not([data-page="home"]) .brand img {
      width: 46px;
      height: 46px;
      border-radius: 13px;
    }
    body:not([data-page="home"]) .page-banner {
      margin-bottom: 14px;
      padding: 20px 26px;
      border-radius: 22px;
      height: 142px;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }
    body:not([data-page="home"]) .page-banner h1 {
      font-size: 1.65rem;
      margin-bottom: 5px;
    }
    body:not([data-page="home"]) .page-banner p {
      font-size: 0.88rem;
      line-height: 1.5;
    }
    body:not([data-page="home"]) .card {
      padding: 17px 18px;
      border-radius: 20px;
    }
    body:not([data-page="home"]) .card h2 {
      margin-bottom: 9px;
      font-size: 1.18rem;
    }
    body:not([data-page="home"]) .eyebrow {
      margin-bottom: 7px;
      font-size: 0.7rem;
    }
    body:not([data-page="home"]) .section-grid,
    body:not([data-page="home"]) .overview-grid,
    body:not([data-page="home"]) .research-grid,
    body:not([data-page="home"]) .viz-grid,
    body:not([data-page="home"]) .grid,
    body:not([data-page="home"]) .wide-stack {
      gap: 14px;
      margin-top: 14px;
    }
    body:not([data-page="home"]) .workflow { gap: 8px; margin-top: 10px; }
    body:not([data-page="home"]) .workflow-step {
      grid-template-columns: 32px 1fr;
      gap: 10px;
      padding: 10px 12px;
      border-radius: 14px;
    }
    body:not([data-page="home"]) .workflow-step b { width: 32px; height: 32px; }
    body:not([data-page="home"]) .mini-grid,
    body:not([data-page="home"]) .summary-grid,
    body:not([data-page="home"]) .details-grid,
    body:not([data-page="home"]) .ablation-summary,
    body:not([data-page="home"]) .perclass-grid,
    body:not([data-page="home"]) .mis-grid {
      gap: 9px;
      margin-top: 10px;
    }
    body:not([data-page="home"]) .metric,
    body:not([data-page="home"]) .summary-item,
    body:not([data-page="home"]) .detail-card,
    body:not([data-page="home"]) .ablation-stat,
    body:not([data-page="home"]) .perclass-card,
    body:not([data-page="home"]) .mis-card {
      padding: 11px 12px;
      border-radius: 15px;
      min-height: auto;
    }
    body:not([data-page="home"]) .architecture-shell {
      margin-top: 12px;
      padding: 15px;
      border-radius: 18px;
    }
    body:not([data-page="home"]) .architecture-node {
      min-height: 82px;
      padding: 11px;
      border-radius: 15px;
    }
    body:not([data-page="home"]) .branch-stack .architecture-node { min-height: 76px; }
    body:not([data-page="home"]) .architecture-node span { font-size: 0.72rem; }
    body:not([data-page="home"]) .experiment-table th,
    body:not([data-page="home"]) .experiment-table td,
    body:not([data-page="home"]) .baseline-table th,
    body:not([data-page="home"]) .baseline-table td,
    body:not([data-page="home"]) .compare-table th,
    body:not([data-page="home"]) .compare-table td {
      padding: 9px 8px;
      font-size: 0.82rem;
    }
    body:not([data-page="home"]) .protocol-strip { gap: 7px; margin-top: 9px; }
    body:not([data-page="home"]) .protocol-strip div { padding: 8px 10px; }
    body[data-page="dataset"] .dataset-controls {
      margin-top: 11px;
      padding: 11px;
      border-radius: 15px;
    }
    body[data-page="dataset"] .dataset-gallery { gap: 9px; margin-top: 10px; }
    body[data-page="dataset"] .dataset-sample { border-radius: 14px; }
    body[data-page="dataset"] .dataset-sample-info { padding: 8px 9px 10px; }
    body[data-page="results"] .chart-stack,
    body[data-page="results"] .history-grid { gap: 9px; margin-top: 10px; }
    body[data-page="results"] .chart-row,
    body[data-page="results"] .history-card { padding: 10px 11px; border-radius: 15px; }
    body[data-page="results"] .metric-bar-row { gap: 7px; }
    body[data-page="results"] .matrix-panel img { max-width: 290px; }
    body[data-page="results"] .slide-section[data-slide-title="Baseline Results"] {
      grid-template-columns: 1fr;
    }
    body[data-page="predictor"] .upload-box { padding: 14px; border-radius: 16px; }
    body[data-page="predictor"] .preview { max-height: 300px; object-fit: contain; }
    body[data-page="predictor"] .result { padding: 15px; border-radius: 18px; }
    body[data-page="predictor"] .heatmap { padding: 12px; border-radius: 16px; }
    body[data-page="predictor"] .confidence-panel { margin: 10px 0; padding: 12px; }
    body[data-page="reports"] .history-item { padding: 10px 12px; border-radius: 14px; }
    body[data-page="reports"] .matrix-panel img { max-width: 270px; }
    body:not([data-page="home"]) footer {
      margin-top: 14px;
      padding: 14px 18px;
      border-radius: 20px;
      font-size: 0.84rem;
    }
    @media (max-width: 800px) {
      .topnav { align-items: center; flex-direction: row; flex-wrap: wrap; }
      .brand { flex: 1; min-width: 0; }
      .mobile-menu-button { display: inline-flex; align-items: center; justify-content: center; }
      .toolbar { display: none; width: 100%; justify-content: flex-start; }
      .topnav.menu-open .toolbar { display: flex; }
      .navlinks { width: 100%; padding-top: 8px; border-top: 1px solid var(--line); }
      .table-scroll::before {
        content: "Swipe horizontally to view all columns";
        position: sticky;
        left: 0;
        display: block;
        width: max-content;
        padding: 7px 9px;
        color: var(--text-soft);
        font-size: 0.7rem;
        font-weight: 700;
      }
      .hero, .section-grid, .overview-grid, .research-grid, .stats-grid, .grid, .summary-grid, .mis-grid, .perclass-grid, .heatmap-grid, .viz-grid, .details-grid, .protocol-strip, .ablation-summary, .dataset-controls, .uncertainty-grid, .quick-links-grid, .stage-preview-grid, .quality-checks, .download-grid, .reference-grid { grid-template-columns: 1fr; }
      .dataset-gallery { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .architecture-flow { grid-template-columns: 1fr; }
      .flow-arrow { transform: rotate(90deg); }
      .grid { grid-template-columns: 1fr; }
      .footer-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .hero h1 { font-size: 1.7rem; }
      .metric-bar-row { grid-template-columns: 76px 1fr 56px; }
      body.slideshow-mode .slide-section,
      body.slideshow-mode .hero,
      body.slideshow-mode footer { padding: 106px 18px 26px; }
      body:not([data-page="home"]) .page-banner { height: auto; min-height: 118px; }
    }
    @media (max-width: 520px) {
      .wrap { padding: 0 12px; margin-top: 12px; }
      .navlinks { gap: 6px; }
      .navlinks a { padding: 8px 9px; font-size: 0.86rem; }
      .brand > div > div:last-child { font-size: 0.7rem !important; }
      .theme-switcher { border-radius: 18px; flex-wrap: wrap; }
      .theme-switcher button { min-width: 72px; }
      .dataset-gallery { grid-template-columns: 1fr; }
      .footer-grid { grid-template-columns: 1fr; }
      .page-banner { padding: 24px; }
      .page-banner h1 { font-size: 1.65rem; }
    }
  </style>
</head>
<body data-page="{{ active_page }}">
  <div class="wrap">
    <div class="topnav">
      <div class="brand">
        <img src="data:image/png;base64,{{ logo_image }}" alt="UCSY logo">
        <div>
          <div>Tomato Thesis Demo</div>
          <div style="font-size:0.78rem;color:#6b89a5;font-weight:600;">UCSY Smart Agriculture Research</div>
        </div>
      </div>
      <button class="mobile-menu-button" type="button" aria-expanded="false" aria-controls="primary-navigation" onclick="toggleMobileMenu(this)">Menu</button>
      <div class="toolbar">
        <div class="navlinks" id="primary-navigation">
          <a href="{{ url_for('index') }}"{% if active_page == 'home' %} class="active"{% endif %}>Home</a>
          <a href="{{ url_for('research_page') }}"{% if active_page == 'research' %} class="active"{% endif %}>Research</a>
          <a href="{{ url_for('dataset_page') }}"{% if active_page == 'dataset' %} class="active"{% endif %}>Dataset</a>
          <a href="{{ url_for('results_page') }}"{% if active_page == 'results' %} class="active"{% endif %}>Results</a>
          <a href="{{ url_for('predictor_page') }}"{% if active_page == 'predictor' %} class="active"{% endif %}>Predictor</a>
          <a href="{{ url_for('reports_page') }}"{% if active_page == 'reports' %} class="active"{% endif %}>Reports</a>
          <a href="{{ url_for('references_page') }}"{% if active_page == 'references' %} class="active"{% endif %}>References</a>
        </div>
      </div>
    </div>

    {% if active_page != 'home' %}
      <section class="page-banner reveal">
        <span>Tomato Thesis Research Portal</span>
        <h1>{{ page_meta.title }}</h1>
        <p>{{ page_meta.description }}</p>
      </section>
    {% endif %}

    <section class="hero reveal slide-section" data-page="home" data-slide-title="Home">
      <div>
        <div class="hero-badge">UCSY Master Thesis Presentation</div>
        <div class="eyebrow" style="color:#cde7fb;">Master Thesis Demo</div>
        <h1>{{ title }}</h1>
        <p>This web-based demo presents the proposed hybrid CNN-Transformer system for classifying early tomato growth stages. It is designed to support thesis presentation, live prediction, and result reporting in a simple browser interface.</p>
        <div class="hero-actions">
          <a href="{{ url_for('predictor_page') }}" class="hero-button primary">Start Prediction</a>
          <a href="{{ url_for('results_page') }}" class="hero-button secondary">View Results</a>
        </div>
      </div>
      <div class="hero-panel">
        <img class="hero-plant-image" src="{{ url_for('dataset_image', split=hero_sample.split, class_name=hero_sample.class_name, filename=hero_sample.filename) }}" alt="Tomato plant at flowering initiation stage">
        <div class="hero-image-shade"></div>
        <div class="hero-image-caption">
          <span>Dataset Image</span>
          <strong>Visual intelligence for early tomato growth monitoring</strong>
        </div>
      </div>
    </section>

    <div class="quick-links-grid reveal slide-section" data-page="home" data-slide-title="Explore Research">
      <a class="quick-link-card" href="{{ url_for('dataset_page') }}">
        <div class="quick-link-icon">D</div>
        <div class="quick-link-copy"><strong>Explore Dataset</strong><span>Browse all 442 valid images by split and growth stage.</span></div>
        <div class="quick-link-arrow">&#8594;</div>
      </a>
      <a class="quick-link-card" href="{{ url_for('results_page') }}">
        <div class="quick-link-icon">R</div>
        <div class="quick-link-copy"><strong>View Results</strong><span>Compare baselines, metrics, learning curves, and errors.</span></div>
        <div class="quick-link-arrow">&#8594;</div>
      </a>
      <a class="quick-link-card" href="{{ url_for('predictor_page') }}">
        <div class="quick-link-icon">AI</div>
        <div class="quick-link-copy"><strong>Try Prediction</strong><span>Classify a tomato image with confidence and XAI output.</span></div>
        <div class="quick-link-arrow">&#8594;</div>
      </a>
    </div>

    <div class="stats-grid reveal slide-section" data-page="home" data-slide-title="Dataset Overview">
      <section class="card">
        <span class="eyebrow">Dataset</span>
        <h2>{{ dataset_info.total_images }} Images</h2>
        <p class="muted">Collected across two early tomato growth stages for supervised classification.</p>
      </section>
      <section class="card">
        <span class="eyebrow">Accuracy</span>
        <h2>{{ model_info.test_accuracy }}</h2>
        <p class="muted">Best recorded test accuracy from the current trained hybrid model.</p>
      </section>
      <section class="card">
        <span class="eyebrow">Model</span>
        <h2>{{ model_info.name }}</h2>
        <p class="muted">Combines CNN feature extraction with Transformer contextual learning.</p>
      </section>
      <section class="card">
        <span class="eyebrow">Export</span>
        <h2>PDF + PNG</h2>
        <p class="muted">Supports downloadable reports and prediction snapshot exports for demo use.</p>
      </section>
    </div>

    <section class="card stage-preview-section reveal slide-section" data-page="home" data-slide-title="Growth Stage Preview">
      <span class="eyebrow">Visual Class Preview</span>
      <h2>Two Early Tomato Growth Stages</h2>
      <p class="muted">The study focuses on two visually similar stages. These test-set examples show the subtle developmental differences the model learns to recognize.</p>
      <div class="stage-preview-grid">
        {% for item in stage_previews %}
          <article class="stage-preview-card">
            <img loading="lazy" src="{{ url_for('dataset_image', split=item.split, class_name=item.class_name, filename=item.filename) }}" alt="{{ item.class_label }} tomato plant">
            <div class="stage-preview-overlay">
              <span>Stage {{ loop.index }}</span>
              <h3>{{ item.class_label }}</h3>
              {% if item.class_name == 'early_vegetative' %}
                <p>Active leaf development and vegetative growth without clearly visible flowering structures.</p>
              {% else %}
                <p>The early transition toward flowering, with visual cues linked to reproductive development.</p>
              {% endif %}
            </div>
          </article>
        {% endfor %}
      </div>
    </section>

    <div class="section-grid reveal slide-section" id="overview" data-page="home" data-slide-title="Problem Overview">
      <section class="card">
        <span class="eyebrow">About This System</span>
        <h2>Browser Demo for Thesis Presentation</h2>
        <p class="muted">This application serves as a presentation-friendly front end for the trained hybrid model. It allows image upload, sample prediction, result explanation, confidence visualization, history tracking, and report generation in one place.</p>
        <ul class="bullet-list">
          <li>Supports local browser-based image upload and prediction</li>
          <li>Displays confidence scores for both tomato growth stage classes</li>
          <li>Maintains a lightweight prediction history for presentation use</li>
          <li>Exports results as text report, PDF report, and PNG snapshot</li>
        </ul>
      </section>
      <section class="card">
        <span class="eyebrow">Dataset Summary</span>
        <h2>Prepared for Supervised Learning</h2>
        <div class="mini-grid">
          <div class="metric"><span>Total images</span><strong>{{ dataset_info.total_images }}</strong></div>
          <div class="metric"><span>Train split</span><strong>{{ dataset_info.train_split }}</strong></div>
          <div class="metric"><span>Validation split</span><strong>{{ dataset_info.val_split }}</strong></div>
          <div class="metric"><span>Test split</span><strong>{{ dataset_info.test_split }}</strong></div>
        </div>
        <p class="muted" style="margin-top:14px;">The dataset focuses on two visually similar early tomato stages, making the classification task meaningful for agricultural AI research.</p>
        <div class="class-grid">
          {% for item in class_descriptions %}
            <div class="class-card">
              <strong>{{ item.name }}</strong>
              <div class="muted">{{ item.description }}</div>
            </div>
          {% endfor %}
        </div>
      </section>
    </div>

    <div class="overview-grid reveal slide-section" id="workflow" data-page="research" data-slide-title="System Workflow">
      <section class="card">
        <span class="eyebrow">Model Workflow</span>
        <h2>How the System Works</h2>
        <div class="workflow">
          <div class="workflow-step"><b>1</b><div><strong>Input and Preprocessing</strong><span class="muted">The tomato plant image is uploaded, resized, normalized, and prepared for inference.</span></div></div>
          <div class="workflow-step"><b>2</b><div><strong>CNN Feature Learning</strong><span class="muted">The CNN branch learns local visual patterns such as leaf shape, texture, and edge structure.</span></div></div>
          <div class="workflow-step"><b>3</b><div><strong>Transformer Context Learning</strong><span class="muted">The Transformer branch captures global contextual relationships across the full plant image.</span></div></div>
          <div class="workflow-step"><b>4</b><div><strong>Fusion and Classification</strong><span class="muted">The model fuses local and global features and predicts the final tomato growth stage.</span></div></div>
        </div>
      </section>
      <section class="card">
        <span class="eyebrow">Performance Overview</span>
        <h2>Evaluation Snapshot</h2>
        <div class="mini-grid">
          <div class="metric"><span>Test Accuracy</span><strong>{{ model_info.test_accuracy }}</strong></div>
          <div class="metric"><span>F1-score</span><strong>{{ model_info.f1_score }}</strong></div>
          <div class="metric"><span>Classes</span><strong>2</strong></div>
          <div class="metric"><span>Prediction Mode</span><strong>Browser Demo</strong></div>
        </div>
        <ul class="bullet-list" style="margin-top:14px;">
          <li>Confusion matrix is displayed below for test-set interpretation</li>
          <li>Prediction confidence bars help explain the model output visually</li>
          <li>History and export features support thesis demonstration workflow</li>
        </ul>
      </section>
    </div>

    <div class="research-grid reveal slide-section" data-page="research" data-slide-title="Research Design">
      <section class="card">
        <span class="eyebrow">Model Comparison</span>
        <h2>CNN vs Transformer vs Hybrid</h2>
        <p class="muted">This comparison highlights why the hybrid approach is chosen as the proposed model for this thesis.</p>
        <table class="compare-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Strength</th>
              <th>Limitation</th>
              <th>Role in Study</th>
            </tr>
          </thead>
          <tbody>
            {% for row in comparison_rows %}
              <tr>
                <td><strong>{{ row.model }}</strong></td>
                <td>{{ row.strength }}</td>
                <td>{{ row.limitation }}</td>
                <td>{{ row.role }}</td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </section>

      <section class="card">
        <span class="eyebrow">Experiment Summary</span>
        <h2>Training and Testing Setup</h2>
        <div class="summary-grid">
          {% for item in experiment_summary %}
            <div class="summary-item">
              <span>{{ item.label }}</span>
              <strong>{{ item.value }}</strong>
            </div>
          {% endfor %}
        </div>
      </section>
    </div>

    <section class="card reveal slide-section" data-page="research" data-slide-title="Model Architecture">
      <span class="eyebrow">Proposed Architecture</span>
      <h2>Hybrid CNN-Transformer Feature Learning</h2>
      <p class="muted">The same preprocessed tomato image is analyzed by two complementary branches. Their representations are concatenated before the final growth-stage decision.</p>
      <div class="architecture-shell">
        <div class="architecture-flow">
          <div class="architecture-node input-node">
            <strong>Input Image</strong>
            <span>RGB, 224 x 224</span>
          </div>
          <div class="flow-arrow" aria-hidden="true">&#8594;</div>
          <div class="branch-stack">
            <div class="architecture-node">
              <strong>CNN Branch</strong>
              <span>3 convolution blocks learn local shape, texture, and edge features. Output: 128-D.</span>
            </div>
            <div class="architecture-node">
              <strong>Transformer Branch</strong>
              <span>196 patches, CLS token, and 2 encoder layers learn global context. Output: 128-D.</span>
            </div>
          </div>
          <div class="flow-arrow" aria-hidden="true">&#8594;</div>
          <div class="architecture-node fusion-node">
            <strong>Feature Fusion</strong>
            <span>Concatenate local and global representations into a 256-D vector.</span>
          </div>
          <div class="flow-arrow" aria-hidden="true">&#8594;</div>
          <div class="architecture-node">
            <strong>Classifier</strong>
            <span>Fully connected layer, ReLU, dropout, and two-class output.</span>
          </div>
          <div class="flow-arrow" aria-hidden="true">&#8594;</div>
          <div class="architecture-node output-node">
            <strong>Growth Stage</strong>
            <span>Early Vegetative or Flowering Initiation</span>
          </div>
        </div>
        <div class="architecture-legend">
          <span>Local feature learning</span>
          <span>Global context learning</span>
          <span>Joint feature representation</span>
        </div>
      </div>
    </section>

    <section class="card reveal slide-section" data-page="research" data-slide-title="Experiment Comparison">
      <span class="eyebrow">Controlled Experiments</span>
      <h2>Experiment Comparison Table</h2>
      <p class="muted">All models were evaluated on the same 65-image unseen test set. The table records architecture complexity and measured results from the completed runs.</p>
      <div class="table-scroll">
        <table class="experiment-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Architecture</th>
              <th>Epochs</th>
              <th>Parameters</th>
              <th>Best Val. Acc.</th>
              <th>Test Acc.</th>
              <th>F1-score</th>
            </tr>
          </thead>
          <tbody>
            {% for row in experiment_comparison %}
              <tr{% if row.proposed %} class="proposed-row"{% endif %}>
                <td><strong>{{ row.name }}</strong>{% if row.proposed %}<span class="model-chip">Proposed</span>{% endif %}</td>
                <td>{{ row.architecture }}</td>
                <td>{{ row.epochs }}</td>
                <td>{{ row.parameters }}</td>
                <td>{{ row.best_val_accuracy }}</td>
                <td><strong>{{ row.test_accuracy }}</strong></td>
                <td>{{ row.f1_score }}</td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      <div class="protocol-strip">
        <div><strong>Batch:</strong> 8 images</div>
        <div><strong>Learning rate:</strong> 0.001</div>
        <div><strong>Optimizer:</strong> Adam</div>
        <div><strong>Loss:</strong> Cross-Entropy</div>
      </div>
    </section>

    <section class="card reveal slide-section" data-page="research" data-slide-title="Ablation Study">
      <span class="eyebrow">Branch-Level Evidence</span>
      <h2>Ablation Study: Local vs Global Features</h2>
      <p class="muted">This analysis removes one feature-learning branch at a time and compares the available completed runs with the full hybrid configuration.</p>
      <div class="table-scroll">
        <table class="experiment-table">
          <thead>
            <tr>
              <th>Configuration</th>
              <th>Feature Information</th>
              <th>Epochs</th>
              <th>Test Accuracy</th>
              <th>F1-score</th>
              <th>Accuracy Change</th>
            </tr>
          </thead>
          <tbody>
            {% for row in ablation_rows %}
              <tr{% if row.is_full %} class="proposed-row"{% endif %}>
                <td><strong>{{ row.configuration }}</strong>{% if row.is_full %}<span class="model-chip">Full Model</span>{% endif %}</td>
                <td>{{ row.learned_features }}</td>
                <td>{{ row.epochs }}</td>
                <td><strong>{{ row.accuracy }}</strong></td>
                <td>{{ row.f1_score }}</td>
                <td>{{ row.delta }}</td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      <div class="ablation-summary">
        <div class="ablation-stat">
          <span>Without global branch</span>
          <strong>-1.54 pp</strong>
          <small>CNN-only remains strong, but loses the full model's contextual information.</small>
        </div>
        <div class="ablation-stat">
          <span>Without local branch</span>
          <strong>-23.08 pp</strong>
          <small>Transformer-only struggles to preserve the fine visual cues in this small dataset.</small>
        </div>
        <div class="ablation-stat full">
          <span>Full feature fusion</span>
          <strong>100.00%</strong>
          <small>The combined representation achieves the strongest result in the current experiments.</small>
        </div>
      </div>
      <div class="slideshow-tip">Research note: these are branch-level ablations from the completed runs. A stronger final study should repeat all configurations with equal epochs and multiple random seeds.</div>
    </section>

    <section class="card reveal slide-section" data-page="research" data-slide-title="Model Details">
      <span class="eyebrow">Implementation Profile</span>
      <h2>Hybrid Model Details</h2>
      <p class="muted">These values are read or calculated from the loaded checkpoint and current runtime, making the implementation easier to explain and reproduce.</p>
      <div class="details-grid">
        {% for item in model_details %}
          <div class="detail-card">
            <span>{{ item.label }}</span>
            <strong>{{ item.value }}</strong>
            <small>{{ item.note }}</small>
          </div>
        {% endfor %}
      </div>
      <div class="slideshow-tip">Inference time is a runtime estimate and can vary with hardware load. Parameter count and checkpoint size remain fixed for this trained model.</div>
    </section>

    <div class="viz-grid reveal slide-section" data-page="dataset" data-slide-title="Dataset Distribution">
      <section class="card">
        <div class="slide-badge">Dataset Chart</div>
        <h2>Dataset Distribution Pie Chart</h2>
        <p class="muted">This chart shows the valid image split used in the experiment after excluding problematic files during preprocessing.</p>
        <div class="dataset-viz">
          <img src="data:image/svg+xml;base64,{{ dataset_pie_chart }}" alt="Dataset distribution pie chart">
        </div>
      </section>
      <section class="card">
        <div class="slide-badge">Split Summary</div>
        <h2>Train, Validation, and Test Balance</h2>
        <div class="dataset-legend">
          {% for item in dataset_distribution %}
            <div class="dataset-legend-item">
              <div class="dataset-legend-label">
                <span class="dataset-dot" style="background: {{ item.color }};"></span>
                <span>{{ item.label }}</span>
              </div>
              <strong>{{ item.count }} images</strong>
            </div>
          {% endfor %}
        </div>
        <div class="slideshow-tip">This split supports a clear academic workflow: model training on the largest portion, validation during tuning, and final testing on unseen data.</div>
      </section>
    </div>

    <section class="card reveal slide-section" data-page="dataset" data-slide-title="Dataset Explorer">
      <span class="eyebrow">Visual Data Inspection</span>
      <h2>Dataset Explorer</h2>
      <p class="muted">Browse representative valid images by dataset split and growth-stage class. This view supports visual quality checking and dataset explanation during the thesis presentation.</p>
      <div class="dataset-controls">
        <div class="filter-field">
          <label for="dataset-split-filter">Dataset Split</label>
          <select id="dataset-split-filter" onchange="filterDatasetGallery()">
            <option value="all">All splits</option>
            <option value="train">Train</option>
            <option value="val">Validation</option>
            <option value="test">Test</option>
          </select>
        </div>
        <div class="filter-field">
          <label for="dataset-class-filter">Growth Stage</label>
          <select id="dataset-class-filter" onchange="filterDatasetGallery()">
            <option value="all">All classes</option>
            <option value="early_vegetative">Early Vegetative</option>
            <option value="flowering_initiation">Flowering Initiation</option>
          </select>
        </div>
        <div class="dataset-count" id="dataset-visible-count">{{ dataset_gallery|length }} samples</div>
      </div>
      <div class="dataset-gallery" id="dataset-gallery">
        {% for item in dataset_gallery %}
          <article class="dataset-sample" data-split="{{ item.split }}" data-class="{{ item.class_name }}">
            <img loading="lazy" {% if active_page == 'dataset' %}src="{{ url_for('dataset_image', split=item.split, class_name=item.class_name, filename=item.filename) }}"{% else %}data-src="{{ url_for('dataset_image', split=item.split, class_name=item.class_name, filename=item.filename) }}"{% endif %} alt="{{ item.class_label }} dataset sample">
            <div class="dataset-sample-info">
              <strong>{{ item.class_label }}</strong>
              <span>{{ item.split_label }} / {{ item.filename }}</span>
            </div>
          </article>
        {% endfor %}
      </div>
      <div class="dataset-pagination">
        <button type="button" id="dataset-prev" onclick="changeDatasetPage(-1)">Previous</button>
        <div class="dataset-page-info" id="dataset-page-info">Page 1</div>
        <button type="button" id="dataset-next" onclick="changeDatasetPage(1)">Next</button>
      </div>
      <div class="slideshow-tip">{% if dataset_is_full %}The explorer includes every valid image used across the train, validation, and test folders.{% else %}This public demo includes a small sample gallery. Set <code>TOMATO_DATASET_ROOT</code> to the full prepared dataset to browse all images.{% endif %} Use the filters to inspect a specific split or growth-stage class.</div>
    </section>

    <div class="section-grid reveal slide-section" data-page="results" data-slide-title="Baseline Results">
      <section class="card">
        <span class="eyebrow">Actual Baseline Results</span>
        <h2>Measured Test Metrics</h2>
        <p class="muted">These values come from actual runs on the tomato dataset. They show how the proposed hybrid model compares against CNN-only and Transformer-only baselines.</p>
        <div class="table-scroll"><table class="baseline-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-score</th>
              <th>Misclassified</th>
            </tr>
          </thead>
          <tbody>
            {% for row in baseline_models %}
              <tr>
                <td><strong>{{ row.name }}</strong></td>
                <td>{{ row.accuracy }}</td>
                <td>{{ row.precision }}</td>
                <td>{{ row.recall }}</td>
                <td>{{ row.f1_score }}</td>
                <td>{{ row.misclassified_count }}</td>
              </tr>
            {% endfor %}
          </tbody>
        </table></div>
      </section>
      <section class="card">
        <span class="eyebrow">Baseline Comparison Chart</span>
        <h2>Visual Metric Comparison</h2>
        <p class="muted">This chart makes the performance gap easier to present during viva or seminar discussion. The hybrid model remains strongest across all main evaluation metrics.</p>
        <div class="chart-stack">
          {% for row in baseline_chart_rows %}
            <div class="chart-row">
              <div class="chart-head">
                <strong>{{ row.name }}</strong>
                <span class="muted">Accuracy {{ row.accuracy }}</span>
              </div>
              <div class="metric-bars">
                <div class="metric-bar-row">
                  <span>Accuracy</span>
                  <div class="metric-track"><div class="metric-fill" style="width: {{ row.accuracy_width }}%"></div></div>
                  <strong>{{ row.accuracy }}</strong>
                </div>
                <div class="metric-bar-row">
                  <span>Precision</span>
                  <div class="metric-track"><div class="metric-fill alt" style="width: {{ row.precision_width }}%"></div></div>
                  <strong>{{ row.precision }}</strong>
                </div>
                <div class="metric-bar-row">
                  <span>Recall</span>
                  <div class="metric-track"><div class="metric-fill soft" style="width: {{ row.recall_width }}%"></div></div>
                  <strong>{{ row.recall }}</strong>
                </div>
                <div class="metric-bar-row">
                  <span>F1-score</span>
                  <div class="metric-track"><div class="metric-fill" style="width: {{ row.f1_width }}%"></div></div>
                  <strong>{{ row.f1_score }}</strong>
                </div>
              </div>
            </div>
          {% endfor %}
        </div>
      </section>
    </div>

    {% if multi_run_summary.rows %}
      <section class="card reveal slide-section" data-page="results" data-slide-title="Multiple Run Results">
        <span class="eyebrow">Robustness Across Random Seeds</span>
        <h2>Multiple-Run Performance: Mean +/- Standard Deviation</h2>
        <p class="muted">Each architecture was trained from scratch with the same protocol using seeds 42, 123, and 2026. This table measures result stability rather than relying on one favorable run.</p>
        <div class="table-scroll">
          <table class="experiment-table">
            <thead><tr><th>Model</th><th>Runs</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-score</th></tr></thead>
            <tbody>
              {% for row in multi_run_summary.rows %}
                <tr><td><strong>{{ row.name }}</strong></td><td>{{ row.runs }}</td><td>{{ row.accuracy }}</td><td>{{ row.precision }}</td><td>{{ row.recall }}</td><td>{{ row.f1_score }}</td></tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        <div class="protocol-strip">
          <div><strong>Seeds:</strong> 42, 123, 2026</div>
          <div><strong>Epochs:</strong> {{ multi_run_summary.protocol.epochs_per_run }} per run</div>
          <div><strong>Batch:</strong> {{ multi_run_summary.protocol.batch_size }}</div>
          <div><strong>Learning rate:</strong> {{ multi_run_summary.protocol.learning_rate }}</div>
        </div>
        <div class="slideshow-tip"><strong>Interpretation:</strong> {{ multi_run_summary.interpretation }} The original 100% hybrid score should therefore be reported as the best recorded single run, not as guaranteed general performance.</div>
      </section>
    {% endif %}

    <div class="section-grid reveal slide-section" data-page="results" data-slide-title="Training History">
      <section class="card">
        <div class="slide-badge">Learning Curves</div>
        <h2>Training History Line Charts</h2>
        <p class="muted">These line charts compare training and validation accuracy across epochs, helping explain learning stability and generalization behavior for each model.</p>
        <div class="history-grid">
          {% for panel in training_history_panels %}
            <div class="history-card">
              <img src="data:image/svg+xml;base64,{{ panel.chart }}" alt="{{ panel.name }} training history chart">
            </div>
          {% endfor %}
        </div>
      </section>
      <section class="card">
        <div class="slide-badge">Interpretation</div>
        <h2>What the Curves Show</h2>
        <ul class="bullet-list">
          <li>The CNN baseline learns quickly and remains strong because local leaf features are highly informative.</li>
          <li>The Transformer baseline shows weaker validation behavior, suggesting difficulty with this small two-class dataset.</li>
          <li>The hybrid model reaches the strongest final validation performance by combining local detail extraction with global context learning.</li>
        </ul>
        <div class="slideshow-tip">In slideshow mode, this slide works well for explaining not only the final result but also how each model behaved during training.</div>
      </section>
    </div>

    <div class="section-grid reveal slide-section" data-page="results" data-slide-title="Per-Class Analysis">
      <section class="card">
        <span class="eyebrow">Per-Class Analysis</span>
        <h2>Precision and Recall by Class</h2>
        <p class="muted">These bars show how each model behaves on both classes separately. They are especially useful when explaining why Flowering Initiation is harder for the Transformer baseline.</p>
        <div class="perclass-grid">
          {% for panel in per_class_panels %}
            <div class="perclass-card">
              <h3>{{ panel.name }}</h3>
              {% for item in panel.classes %}
                <div class="perclass-item">
                  <strong>{{ item.label }}</strong>
                  <div class="metric-bar-row">
                    <span>Precision</span>
                    <div class="metric-track"><div class="metric-fill alt" style="width: {{ item.precision_width }}%"></div></div>
                    <strong>{{ item.precision }}</strong>
                  </div>
                  <div class="metric-bar-row">
                    <span>Recall</span>
                    <div class="metric-track"><div class="metric-fill soft" style="width: {{ item.recall_width }}%"></div></div>
                    <strong>{{ item.recall }}</strong>
                  </div>
                </div>
              {% endfor %}
            </div>
          {% endfor %}
        </div>
      </section>
      <section class="card">
        <span class="eyebrow">Error Analysis</span>
        <h2>Misclassified Samples by Model</h2>
        <p class="muted">These samples help explain where a baseline struggles. The hybrid model currently shows no test-set errors, while the Transformer baseline makes the most mistakes.</p>
        <div class="mis-grid">
          {% for panel in misclassified_panels %}
            <div class="mis-card">
              <h3>{{ panel.display_name }}</h3>
              <div class="muted">{{ panel.count }} misclassified sample(s)</div>
              {% if panel.samples %}
                {% for item in panel.samples[:1] %}
                  {% if item.image_data %}
                    <img src="data:image/png;base64,{{ item.image_data }}" alt="{{ item.filename }}">
                  {% endif %}
                  <div class="mis-meta">
                    <strong>{{ item.filename }}</strong><br>
                    True: {{ item.true_label }}<br>
                    Predicted: {{ item.predicted_label }}<br>
                    Confidence: {{ item.confidence }}%
                  </div>
                {% endfor %}
              {% else %}
                <div class="mis-meta" style="margin-top:12px;">No misclassified test samples for this model.</div>
              {% endif %}
            </div>
          {% endfor %}
        </div>
      </section>
    </div>

    <section class="card reveal slide-section" data-page="research" data-slide-title="Reproducibility">
      <span class="eyebrow">Reproducibility Profile</span>
      <h2>Software, Hardware, and Training Controls</h2>
      <p class="muted">These recorded settings make the experimental environment transparent and support repeatable implementation.</p>
      <div class="details-grid">
        {% for item in reproducibility_items %}
          <div class="detail-card"><span>{{ item.label }}</span><strong>{{ item.value }}</strong><small>{{ item.note }}</small></div>
        {% endfor %}
      </div>
    </section>

    <section class="card reveal slide-section" data-page="results" data-slide-title="Experiment Downloads">
      <span class="eyebrow">Open Research Data</span>
      <h2>Download Experiment Results</h2>
      <p class="muted">Export the measured model metrics and experiment configuration for independent analysis or thesis appendices.</p>
      <div class="download-grid">
        <a href="{{ url_for('download_experiments_json') }}"><span class="download-card"><strong>JSON Experiment Package</strong><span>Structured baseline, multi-run, protocol, and dataset results.</span></span></a>
        <a href="{{ url_for('download_experiments_csv') }}"><span class="download-card"><strong>CSV Metrics Table</strong><span>Spreadsheet-ready model performance summary.</span></span></a>
      </div>
    </section>

    <div class="section-grid reveal slide-section" data-page="research" data-slide-title="Reflection">
      <section class="card">
        <span class="eyebrow">Research Reflection</span>
        <h2>Limitations and Future Work</h2>
        <div class="limit-card">
          <strong>Current limitations</strong>
          <div class="muted">The current study focuses on only two early tomato growth stages and uses a relatively small dataset. Some invalid image files were excluded during experimentation, and the current browser demo is intended for local academic presentation rather than large-scale deployment.</div>
        </div>
        <div class="limit-card">
          <strong>Future work</strong>
          <div class="muted">Future work can expand the dataset, include more tomato varieties and growth stages, repeat evaluation across field locations and seasons, and validate the uncertainty and heatmap outputs with agricultural experts.</div>
        </div>
      </section>
      <section class="card">
        <span class="eyebrow">Research Contribution</span>
        <h2>Why This Matters at Master Level</h2>
        <ul class="bullet-list">
          <li>Defines a focused agricultural AI problem with practical value</li>
          <li>Justifies a hybrid deep learning approach through model comparison</li>
          <li>Documents an end-to-end experimental pipeline from dataset to evaluation</li>
          <li>Supports both thesis presentation and practical prediction demonstration</li>
        </ul>
      </section>
    </div>

    <div class="grid reveal slide-section" id="predictor" data-page="predictor" data-slide-title="Live Predictor">
      <div class="stack">
        <section class="card">
          <span class="eyebrow">Interactive Prediction</span>
          <h2>Upload Image</h2>
          <form method="post" enctype="multipart/form-data">
            <div class="upload-box">
              <input type="file" name="image" accept=".jpg,.jpeg,.png" required>
              <button type="submit">Predict Growth Stage</button>
            </div>
          </form>
          <div class="sample-buttons">
            {% for sample in sample_images %}
              <form method="post">
                <input type="hidden" name="sample_path" value="{{ sample.path }}">
                <button type="submit">{{ sample.label }}</button>
              </form>
            {% endfor %}
          </div>
          {% if error %}
            <div class="error">{{ error }}</div>
          {% endif %}
          {% if image_data %}
            <img class="preview" src="data:image/png;base64,{{ image_data }}" alt="Uploaded tomato image preview">
          {% endif %}
          {% if quality_info %}
            <div class="quality-panel {{ quality_info.css_class }}">
              <div class="quality-head"><strong>Image Quality Validation</strong><span class="quality-status">{{ quality_info.status }}</span></div>
              <div class="quality-checks">
                {% for check in quality_info.checks %}<div><span>{{ check.label }}</span><strong>{{ check.value }}</strong></div>{% endfor %}
              </div>
              <div class="uncertainty-note">{{ quality_info.message }}</div>
            </div>
          {% endif %}
        </section>

        <section class="card">
          <span class="eyebrow">Model Details</span>
          <h2>Model Information</h2>
          <div class="mini-grid">
            <div class="metric"><span>Model</span><strong>{{ model_info.name }}</strong></div>
            <div class="metric"><span>Classes</span><strong>{{ model_info.classes }}</strong></div>
            <div class="metric"><span>Test Accuracy</span><strong>{{ model_info.test_accuracy }}</strong></div>
            <div class="metric"><span>F1-score</span><strong>{{ model_info.f1_score }}</strong></div>
          </div>
          <div class="limitation-alert"><strong>Research-use limitation:</strong> This model was trained on two growth stages from a small controlled dataset. A high-confidence result does not establish performance on different tomato varieties, field conditions, cameras, or later growth stages.</div>
        </section>
      </div>

      <section class="card">
        <span class="eyebrow">Live Output</span>
        <h2>Prediction Result</h2>
        {% if prediction %}
          <div class="result">
            Predicted class
            <strong>{{ prediction }}</strong>
            <div class="confidence-panel {{ confidence_info.css_class }}">
              <div class="confidence-head">
                <div>
                  <span class="muted">Prediction Reliability</span>
                  <strong style="font-size:1.08rem;margin-top:4px;">{{ confidence_info.level }} Confidence</strong>
                </div>
                <span class="confidence-badge">{{ confidence_info.level }}</span>
              </div>
              <div class="uncertainty-grid">
                <div><span>Top probability</span><strong>{{ confidence_info.top_probability }}</strong></div>
                <div><span>Class margin</span><strong>{{ confidence_info.margin }}</strong></div>
                <div><span>Uncertainty</span><strong>{{ confidence_info.uncertainty }}</strong></div>
              </div>
              <div class="uncertainty-note">{{ confidence_info.message }}</div>
            </div>
            <div class="muted">Confidence scores</div>
            <ul class="prob-list">
              {% for item in scores %}
                <li>
                  <div class="prob-head">
                    <span>{{ item.label }}</span>
                    <span>{{ item.score_text }}%</span>
                  </div>
                  <div class="bar">
                    <div class="bar-fill" style="width: {{ item.score_text }}%"></div>
                  </div>
                </li>
              {% endfor %}
            </ul>
            <div class="explain">
              <strong>Result explanation</strong>
              <div class="muted">{{ explanation }}</div>
            </div>
            <div class="heatmap">
              <div class="heatmap-header">
                <strong>Explainable AI Heatmap</strong>
                <div class="heatmap-badge">Presentation View</div>
              </div>
              <div class="muted">This Grad-CAM-style visualization highlights the image regions that contribute most strongly to the hybrid model's decision.</div>
              <div class="heatmap-grid">
                <div class="heatmap-figure">
                  <span>Input Image</span>
                  <img src="data:image/png;base64,{{ image_data }}" alt="Original tomato image">
                </div>
                <div class="heatmap-figure">
                  <span>Model Attention Map</span>
                  <img src="data:image/png;base64,{{ heatmap_image }}" alt="Hybrid model heatmap">
                </div>
              </div>
              <div class="presentation-note">Warm highlighted regions indicate where the hybrid CNN-Transformer focuses more strongly while separating Early Vegetative and Flowering Initiation samples.</div>
            </div>
            <div class="history-actions">
              <form method="get" action="{{ url_for('download_pdf_report') }}">
                <button class="action-button" type="submit">Download PDF Report</button>
              </form>
              <form method="get" action="{{ url_for('export_prediction_screenshot') }}">
                <button class="action-button ghost" type="submit">Export Prediction Screenshot</button>
              </form>
            </div>
          </div>
        {% else %}
          <p class="muted">No prediction yet. Upload an image from the tomato dataset or use one of the sample buttons to test the model in your browser.</p>
        {% endif %}
      </section>
    </div>

    <div class="wide-stack reveal slide-section" id="reports" data-page="reports" data-slide-title="Reports">
      <section class="card">
        <span class="eyebrow">Session Tracking</span>
        <h2>Prediction History</h2>
        {% if prediction_history %}
          <ul class="history-list">
            {% for item in prediction_history %}
              <li class="history-item">
                <div>
                  <strong>{{ item.source }}</strong>
                  <span>{{ item.prediction }}</span>
                </div>
                <div>
                  <strong>{{ item.top_score }}%</strong>
                  <span>{{ item.confidence_level }} confidence</span>
                </div>
              </li>
            {% endfor %}
          </ul>
          <div class="history-actions">
            <form method="get" action="{{ url_for('download_report') }}">
              <button class="action-button" type="submit">Download Prediction Report</button>
            </form>
            <form method="post" action="{{ url_for('clear_history') }}">
              <button class="action-button secondary" type="submit">Clear History</button>
            </form>
          </div>
        {% else %}
          <p class="muted">Prediction history will appear here after you test images in the app.</p>
        {% endif %}
      </section>

      <section class="card matrix-panel">
        <span class="eyebrow">Evaluation</span>
        <h2>Confusion Matrix</h2>
        <p class="muted">This panel shows the test-set confusion matrix from the trained hybrid model.</p>
        <img src="data:image/png;base64,{{ confusion_matrix_image }}" alt="Confusion matrix for the trained tomato classifier">
      </section>
    </div>

    <section class="card reveal slide-section" data-page="references" data-slide-title="References">
      <span class="eyebrow">Selected Literature</span>
      <h2>Thesis References and Methodological Foundations</h2>
      <p class="muted">These primary studies support the CNN, Transformer, explainable AI, augmentation, and agricultural image-analysis components of the research.</p>
      <div class="reference-grid">
        {% for item in reference_items %}
          <article class="reference-card">
            <span>{{ item.authors }} ({{ item.year }}) | {{ item.source }}</span>
            <h3>{{ item.title }}</h3>
            <p>{{ item.relevance }}</p>
            <a href="{{ item.url }}" target="_blank" rel="noopener noreferrer">Open publication</a>
          </article>
        {% endfor %}
      </div>
    </section>

    <footer class="reveal slide-section" data-slide-title="Closing">
      <div class="footer-grid">
        <div><span>Student</span><strong>Ma Aye Aye Aung</strong></div>
        <div><span>Supervisor</span><strong>Dr. Yu Yu Than</strong></div>
        <div><span>Thesis Title</span><strong>Hybrid CNN-Transformer Approach for Early Tomato Growth Stage Classification</strong></div>
        <div><span>Institution</span><strong>University of Computer Studies, Yangon</strong></div>
      </div>
    </footer>
  </div>
  <script>
    let slideshowIndex = 0;
    let savedScrollY = 0;

    function toggleMobileMenu(button) {
      const navigation = document.querySelector('.topnav');
      const isOpen = navigation?.classList.toggle('menu-open') || false;
      button.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
      button.textContent = isOpen ? 'Close' : 'Menu';
    }

    function setTheme(theme) {
      document.body.setAttribute('data-theme', theme);
      localStorage.setItem('tomato-theme', theme);
    }

    function toggleDarkMode() {
      const current = document.body.getAttribute('data-mode') === 'dark' ? 'light' : 'dark';
      document.body.setAttribute('data-mode', current);
      localStorage.setItem('tomato-mode', current);
    }

    function toggleFullscreen() {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen?.();
      } else {
        document.exitFullscreen?.();
      }
    }

    let datasetPage = 1;
    const datasetPageSize = 24;

    function filterDatasetGallery(resetPage = true) {
      if (resetPage) datasetPage = 1;
      const split = document.getElementById('dataset-split-filter')?.value || 'all';
      const className = document.getElementById('dataset-class-filter')?.value || 'all';
      const samples = Array.from(document.querySelectorAll('.dataset-sample'));
      const matchedSamples = samples.filter((sample) => {
        const splitMatches = split === 'all' || sample.dataset.split === split;
        const classMatches = className === 'all' || sample.dataset.class === className;
        return splitMatches && classMatches;
      });
      const totalPages = Math.max(1, Math.ceil(matchedSamples.length / datasetPageSize));
      datasetPage = Math.max(1, Math.min(datasetPage, totalPages));
      const pageStart = (datasetPage - 1) * datasetPageSize;
      const pageItems = new Set(matchedSamples.slice(pageStart, pageStart + datasetPageSize));
      samples.forEach((sample) => { sample.hidden = !pageItems.has(sample); });
      const countNode = document.getElementById('dataset-visible-count');
      if (countNode) countNode.textContent = `${matchedSamples.length} image${matchedSamples.length === 1 ? '' : 's'}`;
      const pageInfo = document.getElementById('dataset-page-info');
      if (pageInfo) pageInfo.textContent = `Page ${datasetPage} of ${totalPages}`;
      const previousButton = document.getElementById('dataset-prev');
      const nextButton = document.getElementById('dataset-next');
      if (previousButton) previousButton.disabled = datasetPage <= 1;
      if (nextButton) nextButton.disabled = datasetPage >= totalPages;
    }

    function changeDatasetPage(direction) {
      datasetPage += direction;
      filterDatasetGallery(false);
      document.getElementById('dataset-gallery')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function getSlides() {
      const activePage = document.body.dataset.page;
      return Array.from(document.querySelectorAll('.slide-section')).filter((slide) => {
        return !slide.dataset.page || slide.dataset.page === activePage;
      });
    }

    function renderActiveSlide() {
      const slides = getSlides();
      slides.forEach((slide, index) => {
        slide.classList.toggle('active-slide', document.body.classList.contains('slideshow-mode') && index === slideshowIndex);
      });
    }

    function goToSlide(index) {
      const slides = getSlides();
      if (!slides.length) return;
      slideshowIndex = Math.max(0, Math.min(index, slides.length - 1));
      renderActiveSlide();
    }

    function toggleSlideshowMode() {
      const nextState = !document.body.classList.contains('slideshow-mode');
      if (nextState) {
        savedScrollY = window.scrollY || window.pageYOffset || 0;
        document.body.style.top = `-${savedScrollY}px`;
      }
      document.body.classList.toggle('slideshow-mode', nextState);
      document.documentElement.classList.toggle('slideshow-mode', nextState);
      localStorage.setItem('tomato-slideshow', nextState ? 'on' : 'off');
      if (nextState) {
        toggleFullscreen();
        setTimeout(() => {
          renderActiveSlide();
          goToSlide(slideshowIndex);
        }, 120);
      } else if (document.fullscreenElement) {
        document.querySelectorAll('.slide-section').forEach((slide) => slide.classList.remove('active-slide'));
        document.body.style.top = '';
        document.exitFullscreen?.();
        window.scrollTo(0, savedScrollY);
      } else {
        document.querySelectorAll('.slide-section').forEach((slide) => slide.classList.remove('active-slide'));
        document.body.style.top = '';
        window.scrollTo(0, savedScrollY);
      }
    }

    const savedTheme = localStorage.getItem('tomato-theme') || 'default';
    const savedMode = localStorage.getItem('tomato-mode') || 'light';
    const savedSlideshow = localStorage.getItem('tomato-slideshow') || 'off';
    document.body.setAttribute('data-theme', savedTheme);
    document.body.setAttribute('data-mode', savedMode);
    filterDatasetGallery(false);
    if (savedSlideshow === 'on') {
      document.body.classList.add('slideshow-mode');
      document.documentElement.classList.add('slideshow-mode');
      document.body.style.top = '0px';
      renderActiveSlide();
    }

    const revealObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
        }
      });
    }, { threshold: 0.14 });

    document.querySelectorAll('.reveal').forEach((node) => revealObserver.observe(node));

    const blockSlideScroll = (event) => {
      if (document.body.classList.contains('slideshow-mode')) {
        event.preventDefault();
      }
    };

    window.addEventListener('wheel', blockSlideScroll, { passive: false });
    window.addEventListener('touchmove', blockSlideScroll, { passive: false });

    document.addEventListener('keydown', (event) => {
      if (!document.body.classList.contains('slideshow-mode')) return;
      const slides = getSlides();
      if (!slides.length) return;
      if (event.key === 'ArrowDown' || event.key === 'PageDown' || event.key === 'ArrowRight') {
        event.preventDefault();
        goToSlide(slideshowIndex + 1);
      } else if (event.key === 'ArrowUp' || event.key === 'PageUp' || event.key === 'ArrowLeft') {
        event.preventDefault();
        goToSlide(slideshowIndex - 1);
      } else if (event.key === 'Escape') {
        document.body.classList.remove('slideshow-mode');
        document.documentElement.classList.remove('slideshow-mode');
        document.querySelectorAll('.slide-section').forEach((slide) => slide.classList.remove('active-slide'));
        document.body.style.top = '';
        localStorage.setItem('tomato-slideshow', 'off');
        if (document.fullscreenElement) {
          document.exitFullscreen?.();
        }
        window.scrollTo(0, savedScrollY);
      }
    });
  </script>
</body>
</html>
"""


def prepare_preview(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_base64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def load_base64_image_path(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def svg_to_base64(svg: str) -> str:
    return base64.b64encode(svg.encode("utf-8")).decode("utf-8")


def render_dataset_pie_chart() -> str:
    total = sum(item["count"] for item in dataset_distribution)
    center_x = 180
    center_y = 180
    radius = 120
    start_angle = -90.0
    svg_parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="520" height="360" viewBox="0 0 520 360">',
        '<rect width="520" height="360" rx="28" fill="#ffffff"/>',
        '<text x="34" y="42" fill="#12385d" font-size="24" font-family="Arial" font-weight="700">Dataset Distribution</text>',
    ]

    for item in dataset_distribution:
        fraction = item["count"] / total
        sweep = 360.0 * fraction
        end_angle = start_angle + sweep
        start_rad = np.deg2rad(start_angle)
        end_rad = np.deg2rad(end_angle)
        x1 = center_x + radius * np.cos(start_rad)
        y1 = center_y + radius * np.sin(start_rad)
        x2 = center_x + radius * np.cos(end_rad)
        y2 = center_y + radius * np.sin(end_rad)
        large_arc = 1 if sweep > 180 else 0
        path = (
            f"M {center_x} {center_y} "
            f"L {x1:.2f} {y1:.2f} "
            f"A {radius} {radius} 0 {large_arc} 1 {x2:.2f} {y2:.2f} Z"
        )
        svg_parts.append(f'<path d="{path}" fill="{item["color"]}"/>')
        start_angle = end_angle

    svg_parts.append('<circle cx="180" cy="180" r="62" fill="#ffffff"/>')
    svg_parts.append(f'<text x="180" y="172" text-anchor="middle" fill="#12385d" font-size="18" font-family="Arial" font-weight="700">{total}</text>')
    svg_parts.append('<text x="180" y="196" text-anchor="middle" fill="#6b8f96" font-size="13" font-family="Arial">valid images</text>')

    legend_y = 92
    for item in dataset_distribution:
        svg_parts.append(f'<rect x="320" y="{legend_y}" width="18" height="18" rx="6" fill="{item["color"]}"/>')
        svg_parts.append(f'<text x="348" y="{legend_y + 14}" fill="#12385d" font-size="16" font-family="Arial" font-weight="700">{item["label"]}</text>')
        svg_parts.append(f'<text x="500" y="{legend_y + 14}" text-anchor="end" fill="#6b8f96" font-size="13" font-family="Arial">{item["count"]} images</text>')
        legend_y += 42

    svg_parts.append("</svg>")
    return svg_to_base64("".join(svg_parts))


def render_training_history_chart(model_name: str, points: list[dict[str, float]]) -> str:
    width = 640
    height = 320
    left = 58
    top = 42
    chart_w = 520
    chart_h = 190
    bottom = top + chart_h
    right = left + chart_w

    def xy(index: int, value: float, total_points: int) -> tuple[float, float]:
        x = left if total_points == 1 else left + (chart_w * index / (total_points - 1))
        y = bottom - (value / 100.0) * chart_h
        return x, y

    train_points = []
    val_points = []
    for idx, point in enumerate(points):
        train_points.append(xy(idx, point["train_accuracy"], len(points)))
        val_points.append(xy(idx, point["val_accuracy"], len(points)))

    train_path = " ".join(
        [f"M {train_points[0][0]:.2f} {train_points[0][1]:.2f}"]
        + [f"L {x:.2f} {y:.2f}" for x, y in train_points[1:]]
    )
    val_path = " ".join(
        [f"M {val_points[0][0]:.2f} {val_points[0][1]:.2f}"]
        + [f"L {x:.2f} {y:.2f}" for x, y in val_points[1:]]
    )

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" rx="24" fill="#ffffff"/>',
        f'<text x="28" y="34" fill="#12385d" font-size="22" font-family="Arial" font-weight="700">{model_name}</text>',
        f'<text x="28" y="56" fill="#6b8f96" font-size="13" font-family="Arial">Training vs validation accuracy by epoch</text>',
    ]

    for pct in [0, 25, 50, 75, 100]:
        y = bottom - (pct / 100.0) * chart_h
        svg_parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#e2edf2" stroke-width="1"/>')
        svg_parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" fill="#6b8f96" font-size="12" font-family="Arial">{pct}</text>')

    for idx, point in enumerate(points):
        x, _ = xy(idx, point["train_accuracy"], len(points))
        svg_parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#f1f6f8" stroke-width="1"/>')
        svg_parts.append(f'<text x="{x:.2f}" y="{bottom + 22}" text-anchor="middle" fill="#6b8f96" font-size="12" font-family="Arial">E{point["epoch"]}</text>')

    svg_parts.append(f'<path d="{train_path}" fill="none" stroke="#0f6c78" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
    svg_parts.append(f'<path d="{val_path}" fill="none" stroke="#d96d3a" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')

    for x, y in train_points:
        svg_parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#0f6c78"/>')
    for x, y in val_points:
        svg_parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#d96d3a"/>')

    svg_parts.append('<rect x="390" y="24" width="14" height="14" rx="4" fill="#0f6c78"/>')
    svg_parts.append('<text x="412" y="36" fill="#12385d" font-size="13" font-family="Arial">Train Accuracy</text>')
    svg_parts.append('<rect x="390" y="48" width="14" height="14" rx="4" fill="#d96d3a"/>')
    svg_parts.append('<text x="412" y="60" fill="#12385d" font-size="13" font-family="Arial">Validation Accuracy</text>')
    svg_parts.append("</svg>")
    return svg_to_base64("".join(svg_parts))


def render_confusion_matrix_image() -> str:
    matrix = metrics["test_metrics"]["confusion_matrix"]
    labels = [label.replace("_", " ").title() for label in class_names]
    image = Image.new("RGB", (520, 420), "#ffffff")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    draw.rounded_rectangle((20, 20, 500, 400), radius=22, outline="#cfe3f2", width=3, fill="#f8fcff")
    draw.text((160, 40), "Confusion Matrix", fill="#12385d", font=title_font)

    start_x = 170
    start_y = 110
    cell = 110

    draw.text((250, 82), "Predicted", fill="#1b75bb", font=font)
    draw.text((70, 225), "Actual", fill="#1b75bb", font=font)

    for idx, label in enumerate(labels):
        draw.text((start_x + idx * cell + 15, 95), label[:12], fill="#12385d", font=font)
        draw.text((30, start_y + idx * cell + 42), label[:12], fill="#12385d", font=font)

    for row in range(2):
        for col in range(2):
            x1 = start_x + col * cell
            y1 = start_y + row * cell
            x2 = x1 + cell - 10
            y2 = y1 + cell - 10
            fill = "#dff1ff" if row == col else "#eef6fb"
            draw.rounded_rectangle((x1, y1, x2, y2), radius=18, outline="#7db4de", width=2, fill=fill)
            value = str(matrix[row][col])
            bbox = draw.textbbox((0, 0), value, font=font)
            tx = x1 + ((x2 - x1) - (bbox[2] - bbox[0])) / 2
            ty = y1 + ((y2 - y1) - (bbox[3] - bbox[1])) / 2
            draw.text((tx, ty), value, fill="#12385d", font=font)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_hybrid_heatmap(image: Image.Image) -> str:
    tensor = transform(image).unsqueeze(0).to(device)
    activations = {}
    gradients = {}

    target_layer = model.cnn_branch[10]

    def forward_hook(_module, _input, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    pred_idx = int(logits.argmax(dim=1).item())
    logits[0, pred_idx].backward()

    handle_f.remove()
    handle_b.remove()

    acts = activations["value"]
    grads = gradients["value"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze(0)
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    cam_np = cam.detach().cpu().numpy()
    cam_img = Image.fromarray(np.uint8(cam_np * 255), mode="L").resize(image.size, Image.Resampling.BILINEAR)

    heat_alpha = np.array(cam_img, dtype=np.uint8)
    heat_rgb = np.zeros((heat_alpha.shape[0], heat_alpha.shape[1], 3), dtype=np.uint8)
    heat_rgb[..., 0] = np.clip(heat_alpha * 1.15, 0, 255)
    heat_rgb[..., 1] = np.clip(heat_alpha * 0.55, 0, 255)
    heat_rgb[..., 2] = np.clip(heat_alpha * 0.18, 0, 255)
    overlay = Image.fromarray(heat_rgb, mode="RGB")

    base = image.convert("RGB")
    blended = Image.blend(base, overlay, alpha=0.38)
    draw = ImageDraw.Draw(blended)
    draw.rounded_rectangle((16, 16, 220, 58), radius=14, fill=(18, 53, 81))
    draw.text((30, 28), "Hybrid Heatmap", fill="white", font=ImageFont.load_default())
    return prepare_preview(blended)


def build_misclassified_panels() -> list[dict[str, object]]:
    panels = []
    for key in ["cnn", "transformer", "hybrid"]:
        model_entry = research_data["models"][key]
        samples = []
        for item in model_entry["misclassified_samples"][:4]:
            sample_path = item.get("path")
            image_data = None
            if sample_path and Path(sample_path).is_file():
                image_data = load_base64_image_path(sample_path)
            samples.append(
                {
                    **item,
                    "image_data": image_data,
                }
            )
        panels.append(
            {
                "display_name": model_entry["display_name"],
                "count": len(model_entry["misclassified_samples"]),
                "samples": samples,
            }
        )
    return panels


def explain_prediction(prediction: str) -> str:
    if prediction == "Early Vegetative":
        return "The model detected visual patterns that are more consistent with the early vegetative stage, such as stronger leaf-focused structure and pre-flowering plant appearance."
    return "The model detected visual cues that are more consistent with flowering initiation, including features that suggest transition toward the flowering stage."


def assess_image_quality(image: Image.Image) -> dict[str, object]:
    width, height = image.size
    preview = image.convert("RGB").resize((256, 256))
    rgb = np.asarray(preview, dtype=np.float32)
    grayscale = np.asarray(preview.convert("L"), dtype=np.float32)
    edge_map = np.asarray(preview.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    brightness = float(grayscale.mean())
    sharpness = float(edge_map.var())
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    green_ratio = float(((green > red * 1.05) & (green > blue * 1.05) & (green > 45)).mean())

    warnings = []
    rejected = width < 96 or height < 96
    if rejected:
        warnings.append("Image resolution is below the minimum 96 x 96 requirement.")
    if brightness < 45:
        warnings.append("The image appears too dark for reliable visual analysis.")
    elif brightness > 225:
        warnings.append("The image appears overexposed and may hide plant details.")
    if sharpness < 120:
        warnings.append("The image may be blurred or lack clear edge detail.")
    if green_ratio < 0.015:
        warnings.append("Very little green plant content was detected; confirm that a tomato plant is visible.")

    if rejected:
        status, css_class = "Reject", "reject"
    elif warnings:
        status, css_class = "Review", "review"
    else:
        status, css_class = "Pass", "pass"

    return {
        "status": status,
        "css_class": css_class,
        "can_predict": not rejected,
        "message": " ".join(warnings) if warnings else "The image passed the automated quality checks and is suitable for model inference.",
        "checks": [
            {"label": "Resolution", "value": f"{width} x {height}"},
            {"label": "Brightness", "value": f"{brightness:.1f} / 255"},
            {"label": "Edge Clarity", "value": f"{sharpness:.1f}"},
            {"label": "Green Content", "value": f"{green_ratio * 100:.1f}%"},
        ],
    }


def calculate_confidence_info(probabilities: list[float]) -> dict[str, str]:
    ranked = sorted(probabilities, reverse=True)
    top_probability = ranked[0]
    margin = ranked[0] - ranked[1]
    entropy = -sum(probability * np.log2(max(probability, 1e-12)) for probability in probabilities)
    normalized_uncertainty = entropy / np.log2(len(probabilities))

    if top_probability >= 0.85 and margin >= 0.50:
        level = "High"
        css_class = "high"
        message = "The classes are clearly separated. This prediction is suitable for demonstration, but confidence does not guarantee correctness."
    elif top_probability >= 0.70 and margin >= 0.25:
        level = "Medium"
        css_class = "medium"
        message = "The model shows a preference, but some ambiguity remains. Visual review is recommended before using the result."
    else:
        level = "Low"
        css_class = "low"
        message = "The class probabilities are close. Treat this result as uncertain and request expert or manual verification."

    return {
        "level": level,
        "css_class": css_class,
        "top_probability": f"{top_probability * 100:.2f}%",
        "margin": f"{margin * 100:.2f} pp",
        "uncertainty": f"{normalized_uncertainty * 100:.2f}%",
        "message": message,
    }


def predict_image(image: Image.Image) -> tuple[str, list[dict[str, str]], str, dict[str, str]]:
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    best_idx = int(torch.tensor(probs).argmax().item())
    prediction = class_names[best_idx].replace("_", " ").title()
    scores = [
        {"label": label.replace("_", " ").title(), "score_text": f"{prob * 100:.2f}"}
        for label, prob in zip(class_names, probs)
    ]
    return prediction, scores, explain_prediction(prediction), calculate_confidence_info(probs)


def build_prediction_screenshot(payload: dict[str, object]) -> bytes:
    image = Image.new("RGB", (1280, 720), "#eef6fb")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    draw.rounded_rectangle((34, 34, 1246, 686), radius=28, fill="#ffffff", outline="#d4e7f5", width=3)
    draw.rounded_rectangle((54, 54, 1226, 164), radius=24, fill="#17395d")
    draw.text((88, 92), "Tomato Growth Stage Classifier", fill="#ffffff", font=title_font)
    draw.text((88, 120), "Prediction export snapshot", fill="#d7ebfb", font=font)

    uploaded = Image.open(io.BytesIO(payload["image_bytes"])).convert("RGB")
    uploaded.thumbnail((460, 420))
    image.paste(uploaded, (88, 214))
    draw.rounded_rectangle((78, 204, 568, 644), radius=22, outline="#c9e0f1", width=2)

    draw.text((640, 220), f"Predicted Class: {payload['prediction']}", fill="#12385d", font=title_font)
    draw.text((640, 254), f"Source: {payload['source']}", fill="#5f7d98", font=font)
    draw.text((640, 282), f"Top Confidence: {payload['top_score']}%", fill="#1b75bb", font=font)
    draw.text((900, 282), f"Reliability: {payload['confidence_info']['level']}", fill="#1b75bb", font=font)
    draw.text((640, 322), "Confidence Scores", fill="#12385d", font=font)

    top = 350
    for item in payload["scores"]:
        label = item["label"]
        score = float(item["score_text"])
        draw.text((640, top), f"{label} - {item['score_text']}%", fill="#12385d", font=font)
        draw.rounded_rectangle((640, top + 24, 1100, top + 44), radius=10, fill="#e5f0f9")
        draw.rounded_rectangle((640, top + 24, 640 + int(4.6 * score), top + 44), radius=10, fill="#2f84c5")
        top += 80

    draw.text((640, 520), "Result Explanation", fill="#12385d", font=font)
    explanation = str(payload["explanation"])
    draw.multiline_text((640, 548), explanation, fill="#5d7d9a", font=font, spacing=6)

    return image_to_png_bytes(image)


def build_pdf_report() -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setFillColor(colors.HexColor("#17395d"))
    pdf.roundRect(36, height - 108, width - 72, 62, 14, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(56, height - 78, APP_TITLE)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(56, height - 95, "PDF report generated from the local thesis demo app")

    pdf.setFillColor(colors.HexColor("#12385d"))
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(48, height - 136, "Project Information")
    pdf.setFont("Helvetica", 10)
    info_lines = [
        "Student: Ma Aye Aye Aung",
        "Supervisor: Dr. Yu Yu Than",
        "Institution: University of Computer Studies, Yangon",
        "Thesis Title: Hybrid CNN-Transformer Approach for Early Tomato Growth Stage Classification",
        f"Model: {model_info['name']}",
        f"Classes: {model_info['classes']}",
        f"Test Accuracy: {model_info['test_accuracy']}",
        f"F1-score: {model_info['f1_score']}",
    ]
    y = height - 156
    for line in info_lines:
        pdf.drawString(54, y, line)
        y -= 16

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(48, y - 8, "Prediction History")
    pdf.setFont("Helvetica", 10)
    y -= 28
    if prediction_history:
        for idx, item in enumerate(prediction_history[:8], start=1):
            pdf.drawString(
                54,
                y,
                f"{idx}. {item['source']} | {item['prediction']} | Top confidence: {item['top_score']}%",
            )
            y -= 16
            if y < 150:
                pdf.showPage()
                y = height - 60
                pdf.setFont("Helvetica", 10)
    else:
        pdf.drawString(54, y, "No predictions recorded yet.")
        y -= 16

    if last_prediction:
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(48, y - 8, "Latest Prediction")
        pdf.setFont("Helvetica", 10)
        y -= 28
        pdf.drawString(54, y, f"Source: {last_prediction['source']}")
        y -= 16
        pdf.drawString(54, y, f"Predicted class: {last_prediction['prediction']}")
        y -= 16
        pdf.drawString(54, y, f"Top confidence: {last_prediction['top_score']}%")
        y -= 24
        pdf.drawString(54, y, f"Reliability level: {last_prediction['confidence_info']['level']}")
        y -= 24
        img_reader = io.BytesIO(last_prediction["image_bytes"])
        pdf.drawInlineImage(Image.open(img_reader), 54, max(90, y - 220), width=180, height=180)
        pdf.drawString(260, max(240, y - 10), "Explanation:")
        text_obj = pdf.beginText(260, max(224, y - 26))
        text_obj.setFont("Helvetica", 10)
        for line in str(last_prediction["explanation"]).split(". "):
            if line:
                text_obj.textLine(line.strip())
        pdf.drawText(text_obj)

    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def download_timestamp(prefix: str, extension: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}.{extension}"


def render_app_page(active_page: str):
    global last_prediction
    prediction = None
    scores: list[dict[str, str]] = []
    image_data = None
    error = None
    explanation = None
    confidence_info = None
    quality_info = None
    heatmap_image = None

    if request.method == "POST":
        sample_path = request.form.get("sample_path")
        file = request.files.get("image")

        if sample_path:
            try:
                image = Image.open(sample_path).convert("RGB")
                image_data = prepare_preview(image)
                quality_info = assess_image_quality(image)
                if quality_info["can_predict"]:
                    prediction, scores, explanation, confidence_info = predict_image(image)
                    heatmap_image = generate_hybrid_heatmap(image)
                else:
                    error = quality_info["message"]
            except Exception:
                error = "The sample image could not be loaded."
        elif not file or not file.filename:
            error = "Please choose an image file."
        else:
            try:
                image = Image.open(file.stream).convert("RGB")
                image_data = prepare_preview(image)
                quality_info = assess_image_quality(image)
                if quality_info["can_predict"]:
                    prediction, scores, explanation, confidence_info = predict_image(image)
                    heatmap_image = generate_hybrid_heatmap(image)
                else:
                    error = quality_info["message"]
                source_name = file.filename
            except Exception:
                error = "The uploaded file could not be processed as an image."
                source_name = None

        if prediction:
            if sample_path:
                source_name = Path(sample_path).name
            top_score = max(scores, key=lambda item: float(item["score_text"]))["score_text"]
            prediction_history.insert(
                0,
                {
                    "source": source_name or "Uploaded image",
                    "prediction": prediction,
                    "top_score": top_score,
                    "confidence_level": confidence_info["level"],
                    "quality_status": quality_info["status"],
                },
            )
            del prediction_history[8:]
            last_prediction = {
                "source": source_name or "Uploaded image",
                "prediction": prediction,
                "top_score": top_score,
                "scores": scores,
                "explanation": explanation,
                "confidence_info": confidence_info,
                "quality_info": quality_info,
                "image_bytes": image_to_png_bytes(image),
            }

    return render_template_string(
        HTML,
        title=APP_TITLE,
        active_page=active_page,
        page_meta=PAGE_META[active_page],
        prediction=prediction,
        scores=scores,
        image_data=image_data,
        error=error,
        explanation=explanation,
        confidence_info=confidence_info,
        quality_info=quality_info,
        model_info=model_info,
        model_details=model_details,
        reproducibility_items=reproducibility_items,
        reference_items=reference_items,
        ablation_rows=ablation_rows,
        comparison_rows=comparison_rows,
        experiment_comparison=experiment_comparison,
        multi_run_summary=load_multi_run_summary(),
        baseline_models=baseline_models,
        baseline_chart_rows=baseline_chart_rows,
        dataset_distribution=dataset_distribution,
        dataset_gallery=dataset_gallery,
        dataset_is_full=dataset_is_full,
        hero_sample=hero_sample,
        stage_previews=stage_previews,
        dataset_pie_chart=render_dataset_pie_chart(),
        dataset_info=dataset_info,
        experiment_summary=experiment_summary,
        class_descriptions=class_descriptions,
        sample_images=sample_images,
        prediction_history=prediction_history,
        training_history_panels=[
            {**panel, "chart": render_training_history_chart(panel["name"], panel["points"])}
            for panel in training_history_panels
        ],
        per_class_panels=per_class_panels,
        misclassified_panels=build_misclassified_panels() if active_page == "results" else [],
        confusion_matrix_image=render_confusion_matrix_image(),
        heatmap_image=heatmap_image,
        logo_image=load_base64_file(LOGO_PATH),
    )


@app.get("/")
def index():
    return render_app_page("home")


@app.get("/research")
def research_page():
    return render_app_page("research")


@app.get("/dataset")
def dataset_page():
    return render_app_page("dataset")


@app.get("/results")
def results_page():
    return render_app_page("results")


@app.route("/predictor", methods=["GET", "POST"])
def predictor_page():
    return render_app_page("predictor")


@app.get("/reports")
def reports_page():
    return render_app_page("reports")


@app.get("/references")
def references_page():
    return render_app_page("references")


@app.get("/dataset-image/<split>/<class_name>/<filename>")
def dataset_image(split: str, class_name: str, filename: str):
    if split not in {"train", "val", "test"} or class_name not in {"early_vegetative", "flowering_initiation"}:
        abort(404)
    class_root = (DATASET_ROOT / split / class_name).resolve()
    image_path = (class_root / filename).resolve()
    if not image_path.is_relative_to(class_root) or not image_path.is_file():
        abort(404)
    return send_file(image_path)


@app.post("/clear-history")
def clear_history():
    prediction_history.clear()
    return redirect(url_for("reports_page"))


@app.get("/download-report")
def download_report():
    lines = [
        APP_TITLE,
        "",
        "Student: Ma Aye Aye Aung",
        "Supervisor: Dr. Yu Yu Than",
        "Institution: University of Computer Studies, Yangon",
        "Thesis Title: Hybrid CNN-Transformer Approach for Early Tomato Growth Stage Classification",
        "",
        f"Model: {model_info['name']}",
        f"Classes: {model_info['classes']}",
        f"Test Accuracy: {model_info['test_accuracy']}",
        f"F1-score: {model_info['f1_score']}",
        "",
        "Prediction History:",
    ]

    if prediction_history:
        for idx, item in enumerate(prediction_history, start=1):
            lines.append(
                f"{idx}. Source: {item['source']} | Prediction: {item['prediction']} | Top confidence: {item['top_score']}%"
            )
    else:
        lines.append("No predictions recorded yet.")

    lines.extend(
        [
            "",
            "Confusion Matrix:",
            f"{metrics['test_metrics']['confusion_matrix'][0]}",
            f"{metrics['test_metrics']['confusion_matrix'][1]}",
        ]
    )

    report_text = "\n".join(lines)
    return Response(
        report_text,
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment; filename={download_timestamp('tomato_prediction_report', 'txt')}"},
    )


@app.get("/download-experiments.json")
def download_experiments_json():
    package = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "thesis_title": "Hybrid CNN-Transformer Approach for Early Tomato Growth Stage Classification",
        "dataset": research_data["dataset_summary"],
        "single_run_results": research_data["models"],
        "trained_hybrid_checkpoint": metrics,
        "multi_run_results": json.loads(MULTI_RUN_PATH.read_text(encoding="utf-8")) if MULTI_RUN_PATH.exists() else None,
        "reproducibility": reproducibility_items,
    }
    return Response(
        json.dumps(package, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": f"attachment; filename={download_timestamp('tomato_experiments', 'json')}"},
    )


@app.get("/download-experiments.csv")
def download_experiments_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["experiment", "model", "seed", "accuracy", "precision", "recall", "f1_score"])
    for key in ["cnn", "transformer", "hybrid"]:
        value = research_data["models"][key]
        writer.writerow(["single_run", value["display_name"], "recorded", value["accuracy"], value["precision"], value["recall"], value["f1_score"]])
    if MULTI_RUN_PATH.exists():
        multi_run = json.loads(MULTI_RUN_PATH.read_text(encoding="utf-8"))
        for value in multi_run.get("models", {}).values():
            for run in value["runs"]:
                writer.writerow(["multi_run", value["display_name"], run["seed"], run["accuracy"], run["precision"], run["recall"], run["f1_score"]])
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={download_timestamp('tomato_experiments', 'csv')}"},
    )


@app.get("/download-pdf-report")
def download_pdf_report():
    pdf_bytes = build_pdf_report()
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={download_timestamp('tomato_prediction_report', 'pdf')}"},
    )


@app.get("/export-prediction-screenshot")
def export_prediction_screenshot():
    if not last_prediction:
        return redirect(url_for("predictor_page"))
    png_bytes = build_prediction_screenshot(last_prediction)
    return Response(
        png_bytes,
        mimetype="image/png",
        headers={"Content-Disposition": f"attachment; filename={download_timestamp('tomato_prediction_snapshot', 'png')}"},
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
