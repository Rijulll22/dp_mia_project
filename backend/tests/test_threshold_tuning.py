import pytest
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from privacy_engine import PrivacyEngine


class MockDegradedModel:
    """Simulates a severely noise-collapsed model where all probabilities are < 0.015."""
    def __init__(self, prob_val=0.005):
        self.prob_val = prob_val

    def predict_proba(self, X):
        probs = np.full((len(X), 2), 1.0 - self.prob_val)
        probs[:, 1] = self.prob_val
        return probs


class MockDiscriminativeModel:
    """Simulates a well-performing model with clear separation."""
    def __init__(self, true_labels):
        self.true_labels = true_labels

    def predict_proba(self, X):
        n = len(X)
        probs = np.zeros((n, 2))
        for i in range(n):
            if self.true_labels[i] == 1:
                p1 = 0.85
            else:
                p1 = 0.10
            probs[i, 1] = p1
            probs[i, 0] = 1.0 - p1
        return probs


def test_all_positive_degraded_model_fallback():
    """
    Test 1: Severe noise collapse (as in plain_laplace).
    Candidate threshold 0.01 would predict 100% positives and achieve trivial F1 = 0.0034.
    The principled decision rule must reject this degenerate candidate and fall back to 0.50.
    """
    engine = PrivacyEngine(seed=42)
    n_val = 1000
    n_fraud = 5 # 0.5% fraud
    y_val = np.zeros(n_val, dtype=int)
    y_val[:n_fraud] = 1

    # Model outputs ~0.012 for all samples.
    # At th=0.01: (0.012 >= 0.01) is True for all -> 100% predicted positive.
    # At th >= 0.02: 0% predicted positive.
    clf = MockDegradedModel(prob_val=0.012)
    selected_thresh = engine._tune_threshold(clf, np.zeros((n_val, 2)), y_val)

    assert selected_thresh == 0.5, f"Expected fallback 0.5, got {selected_thresh}"

    # Evaluate on test set with 0.5
    n_test = 2000
    y_test = np.zeros(n_test, dtype=int)
    y_test[:10] = 1
    eval_metrics = engine.evaluate_model_metrics(clf, np.zeros((n_test, 2)), y_test, threshold=selected_thresh)

    # At threshold 0.5, predicted positives must be 0 (no false alarms)
    assert eval_metrics["predicted_frauds"] == 0
    assert eval_metrics["f1"] == 0.0
    assert eval_metrics["precision"] == 0.0
    assert eval_metrics["recall"] == 0.0


def test_all_negative_model_fallback():
    """
    Test 2: All-negative predictions.
    When a model outputs near-zero for all samples, fallback must be 0.50.
    """
    engine = PrivacyEngine(seed=42)
    n_val = 500
    y_val = np.zeros(n_val, dtype=int)
    y_val[:10] = 1

    clf = MockDegradedModel(prob_val=0.0001)
    selected_thresh = engine._tune_threshold(clf, np.zeros((n_val, 2)), y_val)

    assert selected_thresh == 0.5


def test_ordinary_useful_predictions():
    """
    Test 3: Useful, non-degraded model.
    Validation tuning should find an optimal non-degenerate threshold
    that achieves high F1, precision, and recall.
    """
    engine = PrivacyEngine(seed=42)
    n_val = 1000
    y_val = np.zeros(n_val, dtype=int)
    y_val[:50] = 1 # 50 fraud, 950 normal

    clf = MockDiscriminativeModel(true_labels=y_val)
    selected_thresh = engine._tune_threshold(clf, np.zeros((n_val, 2)), y_val)

    # Optimal threshold should cleanly separate 0.10 and 0.85
    assert 0.10 < selected_thresh < 0.85, f"Selected threshold {selected_thresh} out of expected range"

    # Evaluate on clean test set
    n_test = 1000
    y_test = np.zeros(n_test, dtype=int)
    y_test[:50] = 1
    clf_test = MockDiscriminativeModel(true_labels=y_test)
    eval_metrics = engine.evaluate_model_metrics(clf_test, np.zeros((n_test, 2)), y_test, threshold=selected_thresh)

    assert eval_metrics["f1"] >= 0.95
    assert eval_metrics["precision"] >= 0.95
    assert eval_metrics["recall"] >= 0.95
    assert eval_metrics["predicted_frauds"] == 50


def test_imbalanced_labels_validation_only():
    """
    Test 4: Strict validation-only isolation.
    Tuning on validation with different distribution does not leak test labels.
    """
    engine = PrivacyEngine(seed=42)
    n_val = 500
    y_val = np.zeros(n_val, dtype=int)
    y_val[:20] = 1

    clf = MockDiscriminativeModel(true_labels=y_val)
    thresh = engine._tune_threshold(clf, np.zeros((n_val, 2)), y_val)

    # Test set has different size and labels
    n_test = 800
    y_test = np.zeros(n_test, dtype=int)
    y_test[:40] = 1
    clf_test = MockDiscriminativeModel(true_labels=y_test)

    eval_metrics = engine.evaluate_model_metrics(clf_test, np.zeros((n_test, 2)), y_test, threshold=thresh)
    assert eval_metrics["f1"] >= 0.95
    assert eval_metrics["predicted_frauds"] == 40


def test_trivial_all_positive_baseline_rejected():
    """
    Test 5: Explicit verification that trivial prevalence F1 is rejected.
    Prevalence pi = 10/1000 = 0.01.
    Trivial all-positive F1 = 2 * 0.01 / 1.01 = 0.0198.
    A candidate achieving only 0.0198 by predicting all positives is rejected.
    """
    engine = PrivacyEngine(seed=42)
    y_val = np.zeros(1000, dtype=int)
    y_val[:10] = 1

    # Candidate at 0.01 predicts all 1s
    class TrivialAllPositiveCandidate:
        def predict_proba(self, X):
            return np.full((len(X), 2), 0.015)

    clf = TrivialAllPositiveCandidate()
    thresh = engine._tune_threshold(clf, np.zeros((1000, 2)), y_val)
    assert thresh == 0.5
