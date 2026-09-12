import sys
import os
import pytest
import numpy as np

# Ensure code directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

import eval_metrics
from eval_metrics import (
    plot_roc_auc_curve,
    plot_pr_curve,
    normalize_bengali_text,
    tokenize,
    levenshtein_distance,
    levenshtein_ratio,
    token_jaccard,
    compute_corpus_metrics,
)
from model_evaluation import ModelEvaluator


def test_eval_metrics_does_not_reexport_model_evaluator():
    """ModelEvaluator lives in model_evaluation, not eval_metrics."""
    assert not hasattr(eval_metrics, "ModelEvaluator")


def test_normalize_bengali_text_and_tokenize():
    """Test text normalization and tokenization for Bengali text."""
    raw_text = "  আজকে আবহাওয়া খুব ভালো।  "
    norm_text = normalize_bengali_text(raw_text)
    assert "।" not in norm_text
    assert norm_text == "আজকে আবহাওয়া খুব ভালো"
    
    tokens = tokenize(norm_text)
    assert tokens == ["আজকে", "আবহাওয়া", "খুব", "ভালো"]


def test_levenshtein_and_jaccard_helpers():
    """Test character Levenshtein distance/ratio and token Jaccard similarity."""
    dist = levenshtein_distance("পাখি", "পাখিরা")
    assert dist == 2
    
    ratio = levenshtein_ratio("পাখি", "পাখি")
    assert ratio == pytest.approx(1.0)
    
    jaccard = token_jaccard(["একটি", "সুন্দর", "পাখি"], ["একটি", "পাখি"])
    # 2 common / 3 union = 2/3
    assert jaccard == pytest.approx(2.0 / 3.0)


def test_classification_metrics():
    """Test standard classification metrics calculation."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    
    metrics = ModelEvaluator.classification_metrics(y_true, y_pred)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "specificity" in metrics
    assert "f1_score" in metrics
    assert "f2_score" in metrics
    
    # 3 TP, 1 FP, 1 FN, 3 TN
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(0.75)
    assert metrics["recall"] == pytest.approx(0.75)


def test_asymmetric_jaccard_similarity():
    """Test binary vector Jaccard similarity in ModelEvaluator."""
    vec1 = np.array([1, 1, 0, 0])
    vec2 = np.array([1, 0, 1, 0])
    
    # Intersection = 1, Union = 3 -> 1/3
    sim = ModelEvaluator.jaccard_similarity(vec1, vec2)
    assert sim == pytest.approx(1.0 / 3.0)


def test_token_jaccard_via_binary():
    """Test token Jaccard similarity computed via shared binary vectorization."""
    toks1 = ["একটি", "সুন্দর", "পাখি"]
    toks2 = ["একটি", "পাখি"]
    
    # Common = 2 ("একটি", "পাখি"), Union = 3 ("একটি", "সুন্দর", "পাখি") -> 2/3
    sim = ModelEvaluator.token_jaccard_via_binary(toks1, toks2)
    assert sim == pytest.approx(2.0 / 3.0)

    # Empty token lists
    assert ModelEvaluator.token_jaccard_via_binary([], []) == pytest.approx(1.0)


def test_roc_auc_computation():
    """Test ROC curve thresholds and AUC calculation via trapezoidal integration."""
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    
    fprs, tprs, auc_score = ModelEvaluator.compute_roc_auc(y_true, y_scores)
    
    assert len(fprs) > 0
    assert len(tprs) > 0
    assert 0.0 <= auc_score <= 1.0


def test_pr_auc_computation():
    """Test PR curve thresholds and Precision-Recall AUC calculation."""
    y_true = np.array([0, 1, 0, 1, 1])
    y_scores = np.array([0.1, 0.9, 0.2, 0.8, 0.7])

    precisions, recalls, pr_auc_score = ModelEvaluator.compute_pr_auc(y_true, y_scores)

    assert len(precisions) > 0
    assert len(recalls) > 0
    assert 0.0 <= pr_auc_score <= 1.0


def test_gower_dissimilarity():
    """Test Gower Mixed-Attribute Dissimilarity metric combining Jaccard, length, and Levenshtein."""
    # Identical strings -> 0 dissimilarity
    dissim_identical = ModelEvaluator.compute_gower_dissimilarity("একটি বিড়াল", "একটি বিড়াল")
    assert dissim_identical == pytest.approx(0.0)

    # Completely different strings
    dissim_diff = ModelEvaluator.compute_gower_dissimilarity("ক খ গ", "চ ছ জ ঝ")
    assert 0.4 < dissim_diff <= 1.0


def test_group_metrics_by_dataset():
    """Test dataset subgroup performance segmentation using DatasetComponentFactory matching."""
    per_image = {
        "data/rxxch9vw59.2/images/img1.jpg": {
            "exact_match": 1,
            "normalized_exact_match": 1,
            "r1_f1": 0.8,
            "r2_f1": 0.6,
            "rl_f1": 0.8,
            "char_lev_ratio": 0.9,
            "token_jaccard": 0.85,
        },
        "data/Flickr 8k Dataset/Images/img2.jpg": {
            "exact_match": 0,
            "normalized_exact_match": 0,
            "r1_f1": 0.4,
            "r2_f1": 0.2,
            "rl_f1": 0.4,
            "char_lev_ratio": 0.5,
            "token_jaccard": 0.35,
        },
    }

    segmented = ModelEvaluator.group_metrics_by_dataset(per_image)
    assert isinstance(segmented, dict)
    assert len(segmented) > 0


def test_compute_corpus_metrics():
    """Test compute_corpus_metrics incorporating Gower dissimilarity."""
    dataset_pairs = [
        ("img1.jpg", ["একটি পাখি ডালের উপর বসে আছে।"], "একটি পাখি ডালে বসে আছে"),
        ("img2.jpg", ["একটি বিড়াল মাঠে ঘুমাচ্ছে।"], "একটি ছোট বিড়াল মাঠে ঘুমাচ্ছে"),
    ]
    
    summary, per_img = compute_corpus_metrics(dataset_pairs)
    assert "mean_gower_dissimilarity" in summary
    assert "mean_token_jaccard" in summary
    assert summary["count"] == 2
    assert summary["mean_gower_dissimilarity"] >= 0.0


def test_compute_corpus_metrics_accepts_mapping_inputs():
    """Test direct references/predictions dictionaries used by notebook workflows."""
    references = {
        "img1.jpg": ["একটি পাখি ডালের উপর বসে আছে।"],
        "img2.jpg": ["একটি বিড়াল মাঠে ঘুমাচ্ছে।"],
    }
    predictions = {
        "img1.jpg": "একটি পাখি ডালে বসে আছে",
        "img2.jpg": "একটি ছোট বিড়াল মাঠে ঘুমাচ্ছে",
    }

    metrics = compute_corpus_metrics(references, predictions)

    assert "per_image" in metrics
    assert set(metrics["per_image"]) == {"img1.jpg", "img2.jpg"}
    assert metrics["token_jaccard"] > 0.0


def test_curve_plotting_functions(tmp_path):
    """Test plot_roc_auc_curve and plot_pr_curve output files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fprs = np.array([0.0, 0.5, 1.0])
    tprs = np.array([0.0, 0.8, 1.0])
    roc_out = str(tmp_path / "roc_curve.png")
    plot_roc_auc_curve(fprs, tprs, 0.75, roc_out)
    assert os.path.exists(roc_out)

    precisions = np.array([1.0, 0.8, 0.0])
    recalls = np.array([0.0, 0.5, 1.0])
    pr_out = str(tmp_path / "pr_curve.png")
    plot_pr_curve(precisions, recalls, 0.75, pr_out)
    assert os.path.exists(pr_out)
