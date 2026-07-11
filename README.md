# Hybrid CNN-Transformer Tomato Growth Stage Classifier

A master-thesis research implementation for classifying two visually similar early tomato growth stages from plant images:

- Early Vegetative
- Flowering Initiation

The proposed network combines a CNN branch for local shape, texture, and edge features with a Transformer branch for global image context. The repository includes training scripts, baseline and multi-seed experiments, a trained checkpoint, recorded metrics, explainable-AI visualization, uncertainty reporting, and a responsive Flask research website.

## Highlights

- Hybrid CNN-Transformer architecture with feature fusion
- CNN-only and Transformer-only baselines
- Branch-level ablation comparison
- Three-seed repeated experiments with mean and standard deviation
- Grad-CAM-style model attention visualization
- Confidence, entropy-based uncertainty, and image-quality checks
- Dataset explorer, error analysis, reports, CSV/JSON exports, and references
- Responsive desktop and mobile interface

## Recorded Results

The best recorded single hybrid run achieved 100% accuracy on the 65-image test split. Under an equal three-epoch, three-seed protocol, the recorded mean test accuracies were:

| Model | Mean Accuracy | Standard Deviation |
| --- | ---: | ---: |
| CNN | 96.92% | 4.07% |
| Transformer | 88.72% | 3.87% |
| Hybrid CNN-Transformer | 96.41% | 1.78% |

The hybrid model had the lowest run-to-run accuracy variation. The 100% result should be interpreted as the best single run, not guaranteed general performance.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:5001](http://127.0.0.1:5001).

The repository includes three demo images, so prediction and the main website work without downloading the full private research dataset.

## Full Dataset

The complete dataset is not included because it is approximately 3.1 GB. Prepare it with this structure:

```text
Tomato_Plant_Stages_Dataset/
├── train/
│   ├── early_vegetative/
│   └── flowering_initiation/
├── val/
│   ├── early_vegetative/
│   └── flowering_initiation/
└── test/
    ├── early_vegetative/
    └── flowering_initiation/
```

Set its location before running the website or experiment scripts:

```bash
export TOMATO_DATASET_ROOT="/absolute/path/to/Tomato_Plant_Stages_Dataset"
python app.py
```

## Training

Train the proposed hybrid model:

```bash
python train_hybrid_tomato.py \
  --data-root "$TOMATO_DATASET_ROOT" \
  --epochs 5 \
  --batch-size 8 \
  --output-dir outputs/hybrid_tomato_run1
```

Run baseline analysis:

```bash
python baseline_and_analysis.py --epochs 3 --models cnn transformer hybrid
```

Run the equal-protocol repeated experiment:

```bash
python multi_run_experiments.py
```

## Website Pages

- `/` - Thesis overview and visual stage preview
- `/research` - Methodology, architecture, ablation, and reproducibility
- `/dataset` - Filterable and paginated dataset explorer
- `/results` - Baselines, repeated runs, learning curves, and error analysis
- `/predictor` - Live inference, confidence, uncertainty, quality checks, and heatmap
- `/reports` - Prediction history, exports, and confusion matrix
- `/references` - Primary methodological and agricultural-imaging literature

## Research Limitations

The study uses a relatively small, two-class dataset. Recorded performance does not establish generalization across different tomato varieties, cameras, seasons, field environments, or later growth stages. External field validation and expert assessment of uncertainty and heatmaps remain future work.

## Project Structure

```text
app.py                       Flask research website and inference application
hybrid_model.py              CNN, Transformer, and hybrid model definitions
train_hybrid_tomato.py       Hybrid training pipeline
baseline_and_analysis.py     Baseline training and error analysis
multi_run_experiments.py     Three-seed repeated experiment
outputs/                     Trained checkpoint and recorded metrics
demo_data/                   Small public inference sample set
assets/                      Website logo asset
```
