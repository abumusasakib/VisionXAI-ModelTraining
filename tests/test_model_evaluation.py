import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from model_evaluation import ModelEvaluator


def test_model_evaluator_package_imports_directly():
    import model_evaluation

    assert model_evaluation.__all__ == ["ModelEvaluator"]
    assert model_evaluation.ModelEvaluator is ModelEvaluator
    assert ModelEvaluator.__name__ == "ModelEvaluator"


def test_model_evaluator_package_classification_metrics():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])

    metrics = ModelEvaluator.classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1_score"] == metrics["f1"]


def test_model_evaluator_handles_empty_and_single_class_inputs():
    empty_metrics = ModelEvaluator.classification_metrics([], [])
    assert empty_metrics["accuracy"] == pytest.approx(0.0)
    assert empty_metrics["precision"] == pytest.approx(0.0)
    assert empty_metrics["recall"] == pytest.approx(0.0)

    fprs, tprs, roc_auc_score = ModelEvaluator.compute_roc_auc([0, 0], [0.1, 0.2])
    assert fprs.tolist() == [0.0, pytest.approx(0.5), pytest.approx(1.0)]
    assert tprs.tolist() == [0.0, 0.0, 0.0]
    assert roc_auc_score == pytest.approx(0.0)

    precisions, recalls, pr_auc_score = ModelEvaluator.compute_pr_auc([0, 0], [0.1, 0.2])
    assert precisions[0] == pytest.approx(1.0)
    assert recalls.tolist() == [0.0, 0.0, 0.0]
    assert pr_auc_score == pytest.approx(0.0)
