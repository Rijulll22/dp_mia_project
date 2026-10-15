import pytest
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from privacy_engine import PrivacyEngine

class MockScoreModel:
    """Mock model returning pre-defined continuous probabilities."""
    def __init__(self, probas: np.ndarray):
        self.probas = probas

    def predict_proba(self, X):
        return np.column_stack([1.0 - self.probas, self.probas])

class MockDecisionFunctionModel:
    """Mock model returning continuous decision function scores without predict_proba."""
    def __init__(self, scores: np.ndarray):
        self.scores = scores

    def decision_function(self, X):
        return self.scores

class MockHardPredictionModel:
    """Mock model providing only hard discrete predictions (no continuous scores)."""
    def __init__(self, preds: np.ndarray):
        self.preds = preds

    def predict(self, X):
        return self.preds

class MockNonFiniteModel:
    """Mock model returning non-finite (NaN or Inf) probabilities."""
    def predict_proba(self, X):
        res = np.zeros((len(X), 2))
        res[:, 0] = 0.5
        res[:, 1] = np.nan
        return res

def test_score_vs_hard_prediction_discrepancy():
    """
    Construct a scenario where ROC-AUC and PR-AUC computed from continuous scores
    differ substantially from metrics computed from hard binary predictions.
    Verifies evaluate_model_metrics strictly uses continuous scores.
    """
    engine = PrivacyEngine(seed=42)
    
    # Ground truth labels
    y_test = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    # Continuous probabilities that have perfect rank ordering (AUC = 1.0),
    # but at threshold 0.5, several positive examples fall below 0.5.
    # Scores: class 0 has [0.1, 0.15, 0.2, 0.25, 0.3]
    #         class 1 has [0.35, 0.4, 0.6, 0.7, 0.8]
    # Continuous ROC-AUC is 1.0!
    # Hard predictions at threshold 0.5:
    # class 0 preds: [0, 0, 0, 0, 0] (5 TN)
    # class 1 preds: [0, 0, 1, 1, 1] (2 FN, 3 TP)
    # Hard ROC-AUC is 0.5 * (1.0 + 3/5) = 0.80!
    probas = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.6, 0.7, 0.8])
    X_dummy = np.zeros((len(y_test), 2))
    
    model = MockScoreModel(probas)
    metrics = engine.evaluate_model_metrics(model, X_dummy, y_test, threshold=0.5)
    
    continuous_roc_auc = float(roc_auc_score(y_test, probas))
    continuous_pr_auc = float(average_precision_score(y_test, probas, pos_label=1))
    
    hard_preds = (probas >= 0.5).astype(int)
    hard_roc_auc = float(roc_auc_score(y_test, hard_preds))
    hard_pr_auc = float(average_precision_score(y_test, hard_preds, pos_label=1))
    
    # Assert continuous and hard metrics genuinely differ
    assert continuous_roc_auc == 1.0
    assert hard_roc_auc == 0.80
    assert continuous_roc_auc != hard_roc_auc
    assert continuous_pr_auc != hard_pr_auc
    
    # Verify evaluate_model_metrics returned the continuous score metrics
    assert metrics["rocAuc"] == round(continuous_roc_auc, 4)
    assert metrics["prAuc"] == round(continuous_pr_auc, 4)
    assert metrics["rocAuc"] != round(hard_roc_auc, 4)
    
    # Verify threshold-dependent metrics used hard predictions at threshold
    assert metrics["recall"] == 0.60  # 3/5
    assert metrics["precision"] == 1.0 # 3/3

def test_decision_function_used_when_predict_proba_absent():
    """
    Verifies that when predict_proba is not implemented, decision_function
    is used for continuous scores rather than hard predictions.
    """
    engine = PrivacyEngine(seed=42)
    y_test = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([-2.5, -1.0, -0.5, 0.2, 1.5, 3.0])
    X_dummy = np.zeros((len(y_test), 2))
    
    model = MockDecisionFunctionModel(scores)
    metrics = engine.evaluate_model_metrics(model, X_dummy, y_test, threshold=0.0)
    
    expected_roc_auc = float(roc_auc_score(y_test, scores))
    expected_pr_auc = float(average_precision_score(y_test, scores, pos_label=1))
    
    assert metrics["rocAuc"] == round(expected_roc_auc, 4)
    assert metrics["prAuc"] == round(expected_pr_auc, 4)

def test_hard_predictions_do_not_fabricate_auc():
    """
    Verifies that when a model ONLY provides hard predictions, ROC-AUC and PR-AUC
    are None (undefined/null) rather than fabricated from hard labels.
    """
    engine = PrivacyEngine(seed=42)
    y_test = np.array([0, 0, 1, 1])
    preds = np.array([0, 0, 1, 1])
    X_dummy = np.zeros((len(y_test), 2))
    
    model = MockHardPredictionModel(preds)
    metrics = engine.evaluate_model_metrics(model, X_dummy, y_test, threshold=0.5)
    
    assert metrics["rocAuc"] is None
    assert metrics["prAuc"] is None
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0

def test_non_finite_scores_return_none():
    """
    Verifies that NaN or Inf prediction scores do not cause silent fallback
    or invalid metrics; rocAuc and prAuc must be None.
    """
    engine = PrivacyEngine(seed=42)
    y_test = np.array([0, 0, 1, 1])
    X_dummy = np.zeros((len(y_test), 2))
    
    model = MockNonFiniteModel()
    metrics = engine.evaluate_model_metrics(model, X_dummy, y_test, threshold=0.5)
    
    assert metrics["rocAuc"] is None
    assert metrics["prAuc"] is None

def test_validation_only_threshold_tuning():
    """
    Verifies threshold is tuned on validation partition only and not adjusted on test labels.
    """
    engine = PrivacyEngine(seed=42)
    # Val set has optimal threshold at 0.7
    y_val = np.array([0, 0, 0, 0, 1, 1])
    val_probas = np.array([0.1, 0.2, 0.3, 0.65, 0.75, 0.85])
    val_model = MockScoreModel(val_probas)
    
    X_val_dummy = np.zeros((len(y_val), 2))
    th = engine._tune_threshold(val_model, X_val_dummy, y_val)
    
    # Check that tuned threshold achieves optimal validation F1
    preds = (val_probas >= th).astype(int)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_val, preds)
    assert f1 == 1.0  # Perfect separation at threshold between 0.65 and 0.75
