import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from eval_metrics import EPS, levenshtein_ratio, token_jaccard


class ModelEvaluator:
    """
    Model Evaluator utility.
    Provides classification metrics, asymmetric Jaccard similarity, exact ROC/PR
    curve coordinates, mixed-attribute Gower dissimilarity, and dataset-grouped
    metric summaries.
    """

    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute Accuracy, Precision, Recall, Specificity, F1, and F2 scores."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        precision = tp / (tp + fp + EPS) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + EPS) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp + EPS) if (tn + fp) > 0 else 0.0

        f1 = 2 * (precision * recall) / (precision + recall + EPS) if (precision + recall) > 0 else 0.0
        f2 = 5 * (precision * recall) / (4 * precision + recall + EPS) if (4 * precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (len(y_true) + EPS)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "f2": float(f2),
            "f1_score": float(f1),
            "f2_score": float(f2),
        }

    @staticmethod
    def jaccard_similarity(binary_vec1: np.ndarray, binary_vec2: np.ndarray) -> float:
        """Calculate Asymmetric Jaccard Similarity = Intersection / Union."""
        q = np.sum((binary_vec1 == 1) & (binary_vec2 == 1))
        r = np.sum((binary_vec1 == 1) & (binary_vec2 == 0))
        s = np.sum((binary_vec1 == 0) & (binary_vec2 == 1))
        denom = q + r + s
        if denom == 0:
            return 1.0
        return float(q / denom)

    @staticmethod
    def token_jaccard_via_binary(tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate token Jaccard similarity using a shared binary vocabulary indicator matrix."""
        if not tokens1 and not tokens2:
            return 1.0
        vocab = sorted(list(set(tokens1) | set(tokens2)))
        if not vocab:
            return 1.0
        s1, s2 = set(tokens1), set(tokens2)
        v1 = np.array([1 if w in s1 else 0 for w in vocab], dtype=int)
        v2 = np.array([1 if w in s2 else 0 for w in vocab], dtype=int)
        return ModelEvaluator.jaccard_similarity(v1, v2)

    @staticmethod
    def compute_roc_auc(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute ROC curve (FPR, TPR) coordinates and AUC score using trapezoidal integration."""
        y_true = np.asarray(y_true)
        y_probs = np.asarray(y_probs)

        desc_indices = np.argsort(y_probs)[::-1]
        y_true_sorted = y_true[desc_indices]

        tps = np.cumsum(y_true_sorted == 1)
        fps = np.cumsum(y_true_sorted == 0)

        total_pos = np.sum(y_true == 1)
        total_neg = np.sum(y_true == 0)

        tprs = tps / float(total_pos + EPS) if total_pos > 0 else np.zeros_like(tps, dtype=float)
        fprs = fps / float(total_neg + EPS) if total_neg > 0 else np.zeros_like(fps, dtype=float)

        tprs = np.insert(tprs, 0, 0.0)
        fprs = np.insert(fprs, 0, 0.0)

        auc = 0.0
        for i in range(1, len(fprs)):
            auc += (fprs[i] - fprs[i - 1]) * (tprs[i] + tprs[i - 1]) / 2.0

        return fprs, tprs, float(auc)

    @staticmethod
    def compute_pr_auc(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute Precision-Recall curve coordinates and PR-AUC score."""
        y_true = np.asarray(y_true)
        y_probs = np.asarray(y_probs)

        desc_indices = np.argsort(y_probs)[::-1]
        y_true_sorted = y_true[desc_indices]

        tps = np.cumsum(y_true_sorted == 1)
        fps = np.cumsum(y_true_sorted == 0)

        total_pos = np.sum(y_true == 1)

        recalls = tps / float(total_pos + EPS) if total_pos > 0 else np.zeros_like(tps, dtype=float)
        precisions = tps / (tps + fps + EPS)

        recalls = np.insert(recalls, 0, 0.0)
        precisions = np.insert(precisions, 0, 1.0)

        pr_auc = 0.0
        for i in range(1, len(recalls)):
            pr_auc += (recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2.0

        return precisions, recalls, float(pr_auc)

    @staticmethod
    def compute_gower_dissimilarity(norm_pred: str, norm_ref: str) -> float:
        """
        Compute Gower Mixed-Attribute Dissimilarity from token overlap, string
        length difference, and normalized Levenshtein edit distance.
        """
        if norm_pred == norm_ref:
            return 0.0

        if not norm_pred and not norm_ref:
            return 0.0

        p_toks = norm_pred.split()
        r_toks = norm_ref.split()

        d1 = 1.0 - token_jaccard(p_toks, r_toks)

        len_max = max(len(norm_pred), len(norm_ref))
        d2 = abs(len(norm_pred) - len(norm_ref)) / float(len_max + EPS) if len_max > 0 else 0.0

        d3 = 1.0 - levenshtein_ratio(norm_pred, norm_ref)

        gower_score = (d1 + d2 + d3) / 3.0
        return float(gower_score)

    @staticmethod
    def group_metrics_by_dataset(per_image: Dict[str, dict]) -> Dict[str, dict]:
        """Group performance metrics segmented by dataset source component."""
        try:
            from caption_parsers import DatasetComponentFactory
        except ImportError:
            from code.caption_parsers import DatasetComponentFactory

        components = DatasetComponentFactory.get_components()

        groups = defaultdict(list)
        for img_path, d in per_image.items():
            file_basename = os.path.basename(img_path)
            comp_name = "other"
            for component in components:
                if component.matches(file_basename) or component.matches(img_path):
                    comp_name = component.name
                    break
            groups[comp_name].append(d)

        segmented = {}
        for comp, items in groups.items():
            count = len(items)
            if count == 0:
                continue
            segmented[comp] = {
                "count": count,
                "exact_match_pct": float(np.mean([x.get("exact_match", 0) for x in items])) * 100.0,
                "norm_exact_match_pct": float(np.mean([x.get("normalized_exact_match", 0) for x in items])) * 100.0,
                "mean_rouge1": float(np.mean([x.get("r1_f1", 0.0) for x in items])),
                "mean_rouge2": float(np.mean([x.get("r2_f1", 0.0) for x in items])),
                "mean_rougeL": float(np.mean([x.get("rl_f1", 0.0) for x in items])),
                "mean_char_lev": float(np.mean([x.get("char_lev_ratio", 0.0) for x in items])),
                "mean_token_jaccard": float(np.mean([x.get("token_jaccard", 0.0) for x in items])),
            }
        return segmented
