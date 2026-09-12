# VisionXAI Model Training - Bangla Image Captioning Architecture

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://tensorflow.org/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning framework for **Bangla Image Captioning with Visual Attention**. This repository contains the complete dataset collection pipeline, feature extraction caching using **InceptionV3**, sequence modeling with **Bahdanau Additive Attention** and **GRU Decoder**, evaluation metric logging (BLEU-1 to BLEU-4), and model checkpoint management.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture & Key Features](#-architecture--key-features)
3. [Dataset Ablation Study Highlights](#-dataset-ablation-study-highlights)
4. [Extended Evaluation Metrics & ModelEvaluator](#-extended-evaluation-metrics--modelevaluator)
5. [Caption Selection Policy & Strategy](#-caption-selection-policy--strategy)
6. [Quick Start & Setup](#-quick-start--setup)
7. [Running Unit Tests](#-running-unit-tests)
8. [Repository Structure](#-repository-structure)
9. [Consolidated Project Documentation](#-consolidated-project-documentation)

---

## 🔍 Project Overview

This project implements an encoder-decoder architecture with Bahdanau attention tailored for generating rich Bangla natural language descriptions for arbitrary input images.

Key components of the training pipeline include:

* **Pre-trained CNN Feature Extraction**: Leverages InceptionV3 pre-trained on ImageNet to extract 8x8x2048 feature maps for input images, cached locally to accelerate training.
* **Modular Multi-Dataset Parser**: Normalizes annotations from diverse datasets including `rxxch9vw59.2`, `BAN-Cap`, `banglalekha_image_captions`, and `image_captioning_dataset`.
* **Bahdanau Attention Mechanism**: Dynamically focuses on relevant spatial regions of the image feature grid during caption generation.
* **Recurrent Decoder**: GRU-based sequence generation with word embeddings and teacher forcing.
* **Comprehensive Metrics & Validation**: Automated BLEU score evaluation and epoch checkpointing.

---

## 🏗️ Architecture & Key Features

* **`CNN_Encoder`**: Maps InceptionV3 feature maps (8x8x2048) into a dense embedding space of size `embedding_dim`.
* **`BahdanauAttention`**: Computes attention weights over 64 spatial locations using additive score alignment:
  $$\text{score} = W_a \tanh(W_1 \cdot \text{features} + W_2 \cdot \text{hidden})$$
* **`RNN_Decoder`**: Predicts next token using context vector concatenation with previous token embeddings.
* **Dataset Factory (`code/caption_parsers.py`)**: Extensible parser registry supporting CSV, JSON, and Excel caption formats with automated file path verification.

---

## 📊 Dataset Ablation Study Highlights

A detailed ablation study was performed across training iterations (`dataset_ablation_comparison.md`):

* **Baseline (`rxxch9vw59.2`)**: Achieved stable convergence across BLEU metrics.
* **Extended Dataset (`rxxch9vw59.2` + `BAN-Cap`)**: Demonstrated improved vocabulary diversity and evaluation scores.
* **Data Quality Refinement**: Identified and purged noisy corrupted captions from `Bangla Image Captioning` while maintaining parsing compatibility for `image_captioning_dataset`.

---

## 🚀 Quick Start & Setup

### Docker Build & Run (CodeOcean Container)

```cmd
cd environment && docker build . --tag aed8529a-ded6-4419-ad11-4a8ce0dd580b
docker run --platform linux/amd64 --rm --gpus all --workdir /code --volume "%cd%/data":/data --volume "%cd%/code":/code --volume "%cd%/results":/results aed8529a-ded6-4419-ad11-4a8ce0dd580b bash run
```

### Development with Interactive Jupyter Server

1. **Start Container**:

   ```cmd
   docker run -p 8888:8888 -it --platform linux/amd64 --rm --gpus all --workdir /code --volume "%cd%/data":/data --volume "%cd%/code":/code --volume "%cd%/results":/results aed8529a-ded6-4419-ad11-4a8ce0dd580b /bin/bash
   ```

2. **Activate Conda Environment**:

   ```bash
   conda activate venv
   ```

3. **Launch Jupyter**:

   ```bash
   jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
   ```

---

## 🧪 Running Unit Tests

Run the test suite using `pytest` to verify dataset parsers and model registry components:

```bash
# Run specific caption parser tests
python -m pytest tests/test_caption_parsers.py

# Run all test suites across the repository
python -m pytest
```

---

## 📁 Repository Structure

```text
VisionXAI-ModelTraining/
├── code/
│   ├── bangla_image_caption.ipynb  # Primary training & evaluation notebook
│   ├── caption_parsers.py          # Modular dataset parser registry & normalizers
│   └── model_registry.py           # Encoder, Decoder, Attention PyTorch modules
├── tests/
│   ├── test_caption_parsers.py     # Unit tests for multi-dataset loading & parsing
│   └── test_model_registry.py      # Unit tests for model architectural components
├── data/                           # Dataset storage directory
├── environment/                    # Docker container configuration
├── dataset_ablation_comparison.md  # Detailed experimental comparison report
└── README.md                       # Main repository documentation & archive
```

---

# Model Training Capsule Built using Codeocean

## Project Building and Running

### Build image

```cmd
cd environment && docker build . --tag aed8529a-ded6-4419-ad11-4a8ce0dd580b
```

### Run image

```cmd
docker run --platform linux/amd64 --rm --gpus all --workdir /code --volume "%cd%/data":/data --volume "%cd%/code":/code --volume "%cd%/results":/results aed8529a-ded6-4419-ad11-4a8ce0dd580b bash run
```

## Running Jupyter for Development

1. **Start a Docker Container:**
   You can start a new container from the image you built using the following command in Command Line:

   ```cmd
   docker run -p 8888:8888 -it --platform linux/amd64 --rm --gpus all --workdir /code --volume "%cd%/data":/data --volume "%cd%/code":/code --volume "%cd%/results":/results aed8529a-ded6-4419-ad11-4a8ce0dd580b /bin/bash
   ```

   This command will start a new container based on the image tagged as `aed8529a-ded6-4419-ad11-4a8ce0dd580b` and open an interactive shell (`/bin/bash`) within the container.

2. **Activate Miniconda Environment:**
   Before launching the Jupyter Notebook server, ensure that you have activated your Miniconda environment. If you haven't activated it yet, you can do so by running:

   ```bash
   source /opt/conda/bin/activate
   ```

   This command activates the Miniconda environment.

3. **Launch Jupyter Notebook:**
   Once your Miniconda environment is activated, you can launch the Jupyter Notebook server in Command Line by running:

   ```bash
   jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
   ```

   This command will start the Jupyter Notebook server and open a web browser with the Jupyter Dashboard, where you can navigate your files and create or open notebooks.

4. **Access Jupyter Notebook:**
   After running the `jupyter notebook` command, you should see output in your terminal with a URL that starts with `http://127.0.0.1:8888`. Open this URL in your web browser, and you should be directed to the Jupyter Dashboard, where you can create or open notebooks.

## Running Unit Tests

To run the unit test suite for caption parsers, ablation component logic, and model registries:

### Run all tests

```bash
python -m pytest
```

### Run specific test files

```bash
python -m pytest tests/test_caption_parsers.py
python -m pytest tests/test_model_registry.py
```

---

## 📈 Extended Evaluation Metrics & ModelEvaluator

### Key Additions Made

* **`ModelEvaluator` Class ([`code/eval_metrics.py`](code/eval_metrics.py))**:
  * **Classification Metrics**: Accuracy, Precision, Recall, Specificity, F1-Score, and F2-Score (`ModelEvaluator.classification_metrics`).
  * **Asymmetric Jaccard Similarity**: `ModelEvaluator.jaccard_similarity` ($\frac{Q}{Q + R + S}$) and token-level binary vectorization (`ModelEvaluator.token_jaccard_via_binary`).
  * **ROC Curve & AUC Score**: `ModelEvaluator.compute_roc_auc` using exact trapezoidal integration over False Positive Rate (FPR) vs. True Positive Rate (TPR) thresholds.
* **ROC & PR Curve Plotting**:
  * **`plot_roc_auc_curve`**: Plots the model's Receiver Operating Characteristic curve against the random guess baseline ($\text{AUC} = 0.50$), with area shading under the curve and saving to `results/roc_auc_curve.png`.
  * **`plot_pr_curve`**: Plots the Precision-Recall curve with Cyan shading and saving to `results/pr_auc_curve.png`.
* **Patched Jupyter Notebook ([`code/bangla_image_caption.ipynb`](code/bangla_image_caption.ipynb))**:
  * Updated imports to include `ModelEvaluator`, `plot_roc_auc_curve`, `plot_pr_curve`, and dataset subgroup metrics.
  * Appended dedicated evaluation cells for computing and rendering ROC/AUC and PR-AUC curves alongside dataset subgroup breakdown tables in `results/generated_captions_report.html`.

### Evaluation Metrics Breakdown

#### 1. Precision-Recall Curve & PR-AUC (Precision-Recall Area Under Curve)
* **Relevance for Image Captioning**: While ROC-AUC evaluates the trade-off between True Positive Rate and False Positive Rate, PR-AUC measures Precision vs. Recall across sequence token confidence thresholds. In datasets where exact matches or high-quality captions are rare (imbalanced positive class), PR-AUC provides a more sensitive and informative evaluation of model precision.

#### 2. Precision-Recall / F-Beta Score & Specificity
* **F2-Score ($\beta=2.0$)**: Weighs recall twice as heavily as precision. Useful for assessing how well the model avoids missing key descriptive Bengali words present in ground truth annotations ($F_2 = 5 \cdot \frac{P \cdot R}{4P + R}$).
* **Specificity (True Negative Rate)**: Measures how effectively non-relevant target vocabulary tokens are excluded from predictions.

#### 3. Asymmetric Jaccard Overlap & Gower Dissimilarity
* **Gower / Mixed-Attribute Distance**: Measures dissimilarity between predicted captions and ground truth across mixed attribute types (token overlap, sequence length ratio, and edit distance).
* **Asymmetric Jaccard Similarity**: Evaluates intersection over union ($\frac{Q}{Q + R + S}$) for multi-token Bengali vocabulary sets.

#### 4. Subgroup / Component Metric Segmentation (`fairness_by_group`)
* **Relevance for Image Captioning**: Evaluates captioning performance segmented by dataset source component (`rxxch9vw59.2`, `ban-cap`, `image_captioning_dataset`, `bangla_image_captioning`). This provides fine-grained visibility into per-dataset BLEU, ROUGE, and exact match rates.

### Recommended Implementation Roadmap

| Metric | Origin Module | Primary Function / Equation | Integration Point |
| :--- | :--- | :--- | :--- |
| **PR-AUC & PR Curve** | CSE904 (`ModelEvaluator`) | $\text{PR-AUC} = \sum (R_i - R_{i-1}) \cdot P_i$ | [`code/eval_metrics.py`](code/eval_metrics.py), Notebook & HTML Report |
| **F2-Score** | CSE904 (`ModelEvaluator`) | $F_2 = 5 \cdot \frac{P \cdot R}{4P + R}$ | `ModelEvaluator.classification_metrics` |
| **Gower Dissimilarity** | CSE901 (`gower_calc.py`) | $S_{ij} = \frac{1}{p} \sum_{k=1}^p s_{ijk}$ | Metric evaluation & per-image score logging |
| **Dataset Group Performance** | CSE904 (`fairness_by_group`) | Per-component aggregated BLEU / ROUGE | HTML Report & console logs |

---

## 🎯 Caption Selection Policy & Strategy

To normalize dataset representation across varying annotation counts per image (e.g., `rxxch9vw59.2` has 2 captions/img; `BAN-Cap` & `image_captioning_dataset` have 5 captions/img; `Bangla Image Captioning` has up to 15 captions/img), a **quality-first diverse top-2 selection policy** is implemented in `code/caption_selection/selector.py` and integrated into `export_captions_to_xlsx`:

1. **Normalization & Filtering**:
   * **Unicode NFC Normalization**: Normalizes text while preserving Bengali combining marks, vowel signs, and danda punctuation.
   * **Deduplication**: Rejects exact and near-duplicate captions per image.
   * **Degeneracy Filter**: Rejects empty, corrupted, repetitive, or non-Bengali text.
2. **Quality-First Scoring**:
   * Scores candidate captions based on visible grounding terms (e.g., লোক, নদী, গাছ), action predicates (e.g., বসে, হাঁটছে), and color/spatial attributes (e.g., লাল, বড়, পাশে).
   * Penalizes abstract or poetic descriptions lacking concrete scene entities.
3. **Diversity-Aware Top-2 Selection**:
   * **Caption 1**: Selected as the highest quality-scoring caption.
   * **Caption 2**: Selected as the highest-scoring candidate that provides maximum lexical diversity (measured via Jaccard distance) relative to Caption 1.
   * **Fallback**: Images with exactly two valid captions retain both; images with low-scoring captions fall back gracefully without dropping samples.

Research documentation and analysis schemas are archived in [`docs/caption_selection_research/outline.yaml`](docs/caption_selection_research/outline.yaml) and [`docs/caption_selection_research/fields.yaml`](docs/caption_selection_research/fields.yaml).

---

## 📚 Consolidated Project Documentation

[`docs/consolidated_project_documentation_historical_notes.md`](docs/consolidated_project_documentation_historical_notes.md) is the project-level technical record for the training work.

It consolidates the important material that was previously stored as raw inspection logs, notebook cell dumps, diagnostic traces, and implementation notes, including:

* system architecture and runtime configuration;
* dataset parser strategy, dataset ablation, and quality analysis;
* training, fine-tuning, checkpointing, and evaluation workflow;
* mixed-precision, gradient, and inference troubleshooting notes;
* verification results and a source inventory for the original historical material.
