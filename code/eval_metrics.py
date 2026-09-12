import math
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import unicodedata
import re

EPS = 1e-8

def tokenize(s: str) -> List[str]:
    return s.strip().split()


def normalize_bengali_text(s: str) -> str:
    """Normalize Bengali or general text for robust, language-aware exact matching.

    Steps:
    - Unicode normalize (NFC)
    - Remove Bengali danda (।) and other punctuation
    - Collapse whitespace
    - Trim
    """
    if s is None:
        return ""
    s = s.strip()
    s = unicodedata.normalize("NFC", s)
    # Remove Bengali danda which frequently appears at sentence ends
    s = s.replace("।", "")
    # Remove punctuation while preserving Bengali combining marks and vowel signs.
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if len(tokens)>=n else []

def rouge_n_one_ref(ref: str, pred: str, n: int):
    r_toks = tokenize(ref)
    p_toks = tokenize(pred)
    r_grams = Counter(ngrams(r_toks, n))
    p_grams = Counter(ngrams(p_toks, n))
    match = sum((p_grams & r_grams).values())
    prec = match / (sum(p_grams.values()) + EPS)
    rec = match / (sum(r_grams.values()) + EPS)
    f1 = (2*prec*rec) / (prec + rec + EPS)
    return prec, rec, f1

def rouge_n(refs: List[str], pred: str, n: int):
    # Best reference-based (max F1)
    best = (0.0, 0.0, 0.0)
    for r in refs:
        p, r_, f = rouge_n_one_ref(r, pred, n)
        if f > best[2]:
            best = (p, r_, f)
    return {"precision": best[0], "recall": best[1], "f1": best[2]}

def lcs_len(a: List[str], b: List[str]) -> int:
    # Dynamic programming LCS length
    la, lb = len(a), len(b)
    dp = [0]*(lb+1)
    for i in range(1, la+1):
        prev = 0
        for j in range(1, lb+1):
            tmp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = tmp
    return dp[lb]

def rouge_l(refs: List[str], pred: str):
    best = (0.0, 0.0, 0.0)
    p_toks = tokenize(pred)
    for r in refs:
        r_toks = tokenize(r)
        lcs = lcs_len(r_toks, p_toks)
        prec = lcs / (len(p_toks) + EPS)
        rec = lcs / (len(r_toks) + EPS)
        f1 = (2*prec*rec) / (prec + rec + EPS)
        if f1 > best[2]:
            best = (prec, rec, f1)
    return {"precision": best[0], "recall": best[1], "f1": best[2]}

def token_overlap_scores(refs: List[str], pred: str):
    # For macro precision/recall: compute best-match ref by F1 and derive token precision & recall
    best_match = None
    best_f1 = -1
    p_toks = tokenize(pred)
    p_counts = Counter(p_toks)
    for r in refs:
        r_toks = tokenize(r)
        r_counts = Counter(r_toks)
        matches = sum((p_counts & r_counts).values())
        prec = matches / (sum(p_counts.values()) + EPS)
        rec = matches / (sum(r_counts.values()) + EPS)
        f1 = (2*prec*rec) / (prec + rec + EPS)
        if f1 > best_f1:
            best_f1 = f1
            best_match = (prec, rec, f1)
    return {"precision": best_match[0], "recall": best_match[1], "f1": best_match[2]}


def levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein (edit) distance between two strings (characters).
    Simple dynamic programming implementation with O(len(a)*len(b)) time and O(min(len(a),len(b))) memory.
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # ensure a is the shorter for memory efficiency
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        cur = [j] + [0] * la
        bj = b[j - 1]
        for i in range(1, la + 1):
            cost = 0 if a[i - 1] == bj else 1
            cur[i] = min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + cost)
        prev = cur
    return prev[la]


def levenshtein_ratio(a: str, b: str) -> float:
    """Normalized similarity: 1 - distance / max_len (0..1)
    Returns 1.0 for identical strings and approaches 0 for very different strings.
    """
    a = a or ""
    b = b or ""
    if not a and not b:
        return 1.0
    dist = levenshtein_distance(a, b)
    denom = max(len(a), len(b))
    if denom == 0:
        return 1.0
    return 1.0 - (dist / denom)


def token_jaccard(a: List[str], b: List[str]) -> float:
    """Token-level Jaccard similarity between two token lists."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = sa.intersection(sb)
    union = sa.union(sb)
    return len(inter) / (len(union) + EPS)


def compute_corpus_metrics(
    references: Dict[str, List[str]], predictions: Dict[str, str] = None
):
    """references: image -> list of reference strings; predictions: image -> predicted string"""
    return_tuple = False
    if predictions is None:
        return_tuple = True
        dataset_pairs = references
        references = {img: refs for img, refs, _pred in dataset_pairs}
        predictions = {img: pred for img, _refs, pred in dataset_pairs}

    imgs = list(predictions.keys())
    total = len(imgs)
    exact_matches = 0
    # Less brittle exact-match using lightweight normalization
    normalized_exact_matches = 0

    rouge1, rouge2, rougel = [], [], []
    macro_precisions, macro_recalls = [], []
    # character-level Levenshtein ratio, token-level Jaccard, and Gower dissimilarity
    char_levs, token_jaccards, gower_dissimilarities = [], [], []
    per_image = {}

    for img in imgs:
        pred = predictions[img]
        refs = references.get(img, [])
        if not refs:
            # skip if no references
            continue
        if pred.strip() in [r.strip() for r in refs]:
            exact_matches += 1

        # Normalized exact match (less brittle; removes punctuation/spacing differences)
        norm_pred = normalize_bengali_text(pred)
        norm_refs = [normalize_bengali_text(r) for r in refs]
        if norm_pred and norm_pred in norm_refs:
            normalized_exact_matches += 1

        # --- character-level Levenshtein ratio (best reference) ---
        best_char_lev = 0.0
        for r in norm_refs:
            lev = levenshtein_ratio(norm_pred, r)
            if lev > best_char_lev:
                best_char_lev = lev
        char_levs.append(best_char_lev)

        # --- token-level Jaccard (best reference) ---
        p_toks = tokenize(pred)
        best_tok_j = 0.0
        for r in refs:
            r_toks = tokenize(r)
            j = token_jaccard(p_toks, r_toks)
            if j > best_tok_j:
                best_tok_j = j
        token_jaccards.append(best_tok_j)

        # --- Gower dissimilarity (best reference) ---
        best_gower = 1.0
        for r in norm_refs:
            gow = ModelEvaluator.compute_gower_dissimilarity(norm_pred, r)
            if gow < best_gower:
                best_gower = gow
        gower_dissimilarities.append(best_gower)

        r1 = rouge_n(refs, pred, 1)
        r2 = rouge_n(refs, pred, 2)
        rl = rouge_l(refs, pred)
        token_scores = token_overlap_scores(refs, pred)

        rouge1.append(r1["f1"])
        rouge2.append(r2["f1"])
        rougel.append(rl["f1"])
        macro_precisions.append(token_scores["precision"])
        macro_recalls.append(token_scores["recall"])
        per_image[img] = {
            "pred": pred,
            "r1_f1": r1["f1"],
            "r2_f1": r2["f1"],
            "rl_f1": rl["f1"],
            "precision": token_scores["precision"],
            "recall": token_scores["recall"],
            "f1": token_scores["f1"],
            "exact_match": int(pred.strip() in [r.strip() for r in refs]),
            "normalized_exact_match": int(norm_pred in norm_refs),
            "char_lev_ratio": float(best_char_lev),
            "token_jaccard": float(best_tok_j),
            "gower_dissimilarity": float(best_gower),
        }

    metrics = {
        "accuracy": exact_matches / (total + EPS),
        "normalized_accuracy": normalized_exact_matches / (total + EPS),
        "macro_precision": float(np.mean(macro_precisions)) if macro_precisions else 0.0,
        "macro_recall": float(np.mean(macro_recalls)) if macro_recalls else 0.0,
        "rouge1": float(np.mean(rouge1)) if rouge1 else 0.0,
        "rouge2": float(np.mean(rouge2)) if rouge2 else 0.0,
        "rougeL": float(np.mean(rougel)) if rougel else 0.0,
        "char_lev_ratio": float(np.mean(char_levs)) if char_levs else 0.0,
        "token_jaccard": float(np.mean(token_jaccards)) if token_jaccards else 0.0,
        "gower_dissimilarity": float(np.mean(gower_dissimilarities)) if gower_dissimilarities else 0.0,
        "per_image": per_image,
    }
    if return_tuple:
        summary = dict(metrics)
        summary["mean_gower_dissimilarity"] = summary["gower_dissimilarity"]
        summary["mean_token_jaccard"] = summary["token_jaccard"]
        summary["count"] = len(per_image)
        summary.pop("per_image", None)
        return summary, per_image
    return metrics

def save_metrics_csv(per_image: Dict[str, dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "pred", "precision", "recall", "f1", "r1_f1", "r2_f1", "rl_f1", "exact_match", "normalized_exact_match", "char_lev_ratio", "token_jaccard", "gower_dissimilarity"])
        for img, d in per_image.items():
            writer.writerow([img, d["pred"], d["precision"], d["recall"], d["f1"], d["r1_f1"], d["r2_f1"], d["rl_f1"], d["exact_match"], d.get("normalized_exact_match", 0), d.get("char_lev_ratio", 0.0), d.get("token_jaccard", 0.0), d.get("gower_dissimilarity", 0.0)])


class ModelEvaluator:
    """
    Model Evaluator utility.
    Provides classification metrics (Accuracy, Precision, Recall, Specificity, F1, F2),
    asymmetric Jaccard similarity, and exact ROC curve / AUC computation.
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
        """Compute Precision-Recall curve coordinates (Precisions, Recalls) and PR-AUC score."""
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
        Compute Gower Mixed-Attribute Dissimilarity.
        Combines:
          - f1: Token Jaccard Dissimilarity (1 - Token Jaccard)
          - f2: String Length Ratio Difference (|len1 - len2| / max_len)
          - f3: Levenshtein Edit Distance Ratio (1 - Lev Ratio)
        """
        if norm_pred == norm_ref:
            return 0.0

        if not norm_pred and not norm_ref:
            return 0.0
        
        p_toks = norm_pred.split()
        r_toks = norm_ref.split()

        # Attribute 1: Token overlap dissimilarity
        d1 = 1.0 - token_jaccard(p_toks, r_toks)

        # Attribute 2: Sequence length difference ratio
        len_max = max(len(norm_pred), len(norm_ref))
        d2 = abs(len(norm_pred) - len(norm_ref)) / float(len_max + EPS) if len_max > 0 else 0.0

        # Attribute 3: Character edit dissimilarity
        d3 = 1.0 - levenshtein_ratio(norm_pred, norm_ref)

        gower_score = (d1 + d2 + d3) / 3.0
        return float(gower_score)

    @staticmethod
    def group_metrics_by_dataset(per_image: Dict[str, dict]) -> Dict[str, dict]:
        """
        Group performance metrics segmented by dataset source component (fairness_by_group).
        """
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


def plot_roc_auc_curve(fprs: np.ndarray, tprs: np.ndarray, auc_score: float, out_path: str = None):
    """Plot Receiver Operating Characteristic (ROC) curve with AUC area shading."""
    fprs_list = [float(x) for x in fprs]
    tprs_list = [float(x) for x in tprs]

    fig = plt.figure(figsize=(7, 6))
    plt.plot(fprs_list, tprs_list, color="#2563eb", linewidth=2.5, label=f"Model ROC (AUC = {auc_score:.4f})")
    plt.plot([0.0, 1.0], [0.0, 1.0], color="#e74c3c", linestyle="--", linewidth=1.5, label="Random Guess (AUC = 0.5000)")
    plt.fill_between(fprs_list, tprs_list, color="#3b82f6", alpha=0.2)

    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True)

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_pr_curve(precisions: np.ndarray, recalls: np.ndarray, pr_auc_score: float, out_path: str = None):
    """Plot Precision-Recall (PR) curve with PR-AUC area shading."""
    precisions_list = [float(x) for x in precisions]
    recalls_list = [float(x) for x in recalls]

    fig = plt.figure(figsize=(7, 6))
    plt.plot(recalls_list, precisions_list, color="#06b6d4", linewidth=2.5, label=f"Model PR (PR-AUC = {pr_auc_score:.4f})")
    plt.fill_between(recalls_list, precisions_list, color="#06b6d4", alpha=0.2)

    plt.title("Precision-Recall (PR) Curve", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", frameon=True)

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_epoch_metrics(history: Dict[str, List[float]], out_path: str = None):
    # history keys: 'train_loss', 'val_loss', 'val_accuracy', 'val_rouge1', etc.
    fig = plt.figure(figsize=(10,5))
    if "train_loss" in history:
        plt.plot(history["train_loss"], label="train_loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val_loss")
    if "val_accuracy" in history:
        plt.plot(history["val_accuracy"], label="val_accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

def plot_hist(scores: List[float], title: str, out_path: str = None):
    fig = plt.figure(figsize=(6,4))
    plt.hist(scores, bins=30)
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Count")
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


