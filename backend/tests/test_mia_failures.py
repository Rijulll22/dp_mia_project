import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from attack_engine import run_shadow_mia
from privacy_engine import PrivacyEngine, MECHANISM_LAPLACE_CLIPPED
from sklearn.linear_model import LogisticRegression

def test_successful_attack_auc_05():
    # Construct a case where the attack succeeds but genuinely has 0.5 AUC (random guessing)
    target_model = MagicMock()
    # Predict exactly 0.5 for everything
    target_model.predict_proba.return_value = np.array([[0.5, 0.5]] * 10)
    
    # Needs enough shadow data to not trigger "Insufficient shadow training data"
    # Pool size = 60, n_shadow = 2. split_size = min(30, max(20, 60//3)) = 20
    X_shadow = np.random.rand(60, 2)
    y_shadow = np.random.randint(0, 2, 60)
    
    X_train_target = np.random.rand(10, 2)
    y_train_target = np.random.randint(0, 2, 10)
    X_test_target = np.random.rand(10, 2)
    y_test_target = np.random.randint(0, 2, 10)

    res = run_shadow_mia(
        target_model=target_model,
        X_train_target=X_train_target,
        y_train_target=y_train_target,
        X_test_target=X_test_target,
        y_test_target=y_test_target,
        X_shadow=X_shadow,
        y_shadow=y_shadow,
        n_shadow=2,
    )
    
    assert "error" not in res
    assert res["attack_auc"] is not None
    # Because predict_proba is perfectly uninformative, AUC should be near 0.5 (or exactly 0.5 if all preds are identical)
    # The attack classifier might learn noise, but on eval it will just see identical features [0.5, 0.5, entropy, ce_loss]
    # So attack_probs will be identical, roc_auc_score gives 0.5
    assert res["attack_auc"] == 0.5
    assert res["attacker_advantage"] == 0.0


def test_insufficient_shadow_data():
    target_model = MagicMock()
    X = np.random.rand(2, 2)
    y = np.array([0, 1])
    
    res = run_shadow_mia(
        target_model=target_model,
        X_train_target=X,
        y_train_target=y,
        X_test_target=X,
        y_test_target=y,
        X_shadow=X,
        y_shadow=y,
        n_shadow=2,
    )
    
    assert "error" in res
    assert "Insufficient shadow training data" in res["error"]
    assert res["attack_auc"] is None
    assert res["attacker_advantage"] is None


@patch("attack_engine.LogisticRegression.fit")
def test_attack_fails_to_fit(mock_fit):
    mock_fit.side_effect = ValueError("Simulated fit failure")
    target_model = MagicMock()
    
    X_shadow = np.random.rand(200, 2)
    y_shadow = np.random.randint(0, 2, 200)
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    
    res = run_shadow_mia(
        target_model=target_model,
        X_train_target=X,
        y_train_target=y,
        X_test_target=X,
        y_test_target=y,
        X_shadow=X_shadow,
        y_shadow=y_shadow,
        n_shadow=2,
        classifier_type="random_forest",
    )
    
    assert "error" in res
    assert "Attack classifier training failed" in res["error"]
    assert res["attack_auc"] is None


@patch("attack_engine.roc_auc_score")
def test_metric_error_propagated(mock_auc):
    mock_auc.side_effect = ValueError("Simulated metric error (e.g. invalid predictions or only one class in y_true)")
    target_model = MagicMock()
    target_model.predict_proba.return_value = np.array([[0.5, 0.5]] * 10)
    
    X_shadow = np.random.rand(200, 2)
    y_shadow = np.random.randint(0, 2, 200)
    X_train_target = np.random.rand(10, 2)
    y_train_target = np.random.randint(0, 2, 10)
    X_test_target = np.random.rand(10, 2)
    y_test_target = np.random.randint(0, 2, 10)

    res = run_shadow_mia(
        target_model=target_model,
        X_train_target=X_train_target,
        y_train_target=y_train_target,
        X_test_target=X_test_target,
        y_test_target=y_test_target,
        X_shadow=X_shadow,
        y_shadow=y_shadow,
        n_shadow=2,
    )
    
    assert "error" in res
    assert "MIA metric calculation failed" in res["error"]
    assert res["attack_auc"] is None


def test_missing_target_members_for_balanced_set():
    # 2 members, 10 nonmembers -> cannot form balanced set of 10
    target_model = MagicMock()
    target_model.predict_proba.return_value = np.array([[0.5, 0.5]] * 10)
    
    X_shadow = np.random.rand(200, 2)
    y_shadow = np.random.randint(0, 2, 200)
    
    X_train_target = np.random.rand(2, 2) # only 2 members
    y_train_target = np.random.randint(0, 2, 2)
    X_test_target = np.random.rand(10, 2) # 10 nonmembers
    y_test_target = np.random.randint(0, 2, 10)

    res = run_shadow_mia(
        target_model=target_model,
        X_train_target=X_train_target,
        y_train_target=y_train_target,
        X_test_target=X_test_target,
        y_test_target=y_test_target,
        X_shadow=X_shadow,
        y_shadow=y_shadow,
        n_shadow=2,
    )
    
    assert "error" in res
    assert "Insufficient members" in res["error"]
    assert res["attack_auc"] is None


@patch("privacy_engine.run_shadow_mia")
def test_utility_metrics_preserved_when_mia_fails(mock_run_mia):
    # Mock MIA to return an error
    mock_run_mia.return_value = {
        "attack_accuracy": None, "attack_auc": None, "attacker_advantage": None,
        "n_members_evaluated": 0, "n_nonmembers_evaluated": 0,
        "error": "Simulated MIA failure inside pipeline"
    }
    
    import pandas as pd
    engine = PrivacyEngine(seed=42)
    
    # Generate tiny benchmark data for fast test
    X_df, y_series = engine.generate_benchmark_dataset(n_samples=200, n_features=5, fraud_ratio=0.5)
    
    # Run pipeline
    out = engine.run_pipeline(
        X=X_df,
        y=y_series,
        epsilon=1.0,
        mechanism=MECHANISM_LAPLACE_CLIPPED,
        classifier_type="logistic_regression",
        n_shadow=2,
    )
    
    # Assert utility metrics are present
    assert "metrics" in out
    assert out["metrics"]["rocAuc"] is not None
    assert out["metrics"]["f1"] is not None
    
    # Assert MIA result contains error
    assert "mia_result" in out
    assert "error" in out["mia_result"]
    assert out["mia_result"]["error"] == "Simulated MIA failure inside pipeline"

    # Now verify experiment runner handles this
    from experiment_runner import update_db_record
    from database import SessionLocal, Experiment
    import os
    
    exp_id = "TEST_MIA_FAIL"
    config_key = "test_mia_fail_key"
    
    with SessionLocal() as db:
        db.query(Experiment).filter(Experiment.id == exp_id).delete()
        db.commit()
        
    # The runner does this:
    mia_err = out.get("mia_result", {}).get("error")
    status = "failed" if mia_err else "completed"
    
    update_db_record(
        exp_id, status=status, attempt=1, error_msg=mia_err, out=out,
        eps=1.0, mech=MECHANISM_LAPLACE_CLIPPED, clf="logistic_regression", seed=42, config_key=config_key
    )
    
    with SessionLocal() as db:
        exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
        # Status should be failed
        assert exp.status == "failed"
        # Error message should be saved
        assert exp.error_message == "Simulated MIA failure inside pipeline"
        # Utility metrics should STILL be saved!
        assert exp.metrics is not None
        assert "rocAuc" in exp.metrics
        assert "f1" in exp.metrics
        db.query(Experiment).filter(Experiment.id == exp_id).delete()
        db.commit()


def test_informative_attack_scores():
    """
    Constructs an informative case where members have high confidence and
    nonmembers have lower confidence, proving the attack detects leakage
    (AUC > 0.5 and Advantage > 0).
    """
    from attack_engine import run_shadow_mia
    from sklearn.tree import DecisionTreeClassifier

    # A simple dataset where an unregularized tree overfits 100% to training members
    rng = np.random.RandomState(42)
    X_shadow = rng.randn(200, 4)
    y_shadow = (X_shadow[:, 0] + X_shadow[:, 1] > 0).astype(int)

    X_train_target = rng.randn(40, 4)
    y_train_target = (X_train_target[:, 0] + X_train_target[:, 1] > 0).astype(int)

    X_test_target = rng.randn(40, 4)
    y_test_target = (X_test_target[:, 0] + X_test_target[:, 1] > 0).astype(int)

    # Fit an unpruned decision tree that severely overfits train members
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_target, y_train_target)

    res = run_shadow_mia(
        target_model=dt,
        X_train_target=X_train_target,
        y_train_target=y_train_target,
        X_test_target=X_test_target,
        y_test_target=y_test_target,
        X_shadow=X_shadow,
        y_shadow=y_shadow,
        classifier_type="random_forest",
        n_shadow=2,
        seed=42,
    )

    assert "error" not in res
    assert res["attack_auc"] is not None
    # Overfitted tree should give attack AUC above 0.5
    assert res["attack_auc"] >= 0.55
    assert res["attacker_advantage"] > 0.05


def test_non_finite_attack_scores():
    """
    Verifies that non-finite scores produce an explicit error and AUC is None,
    never fabricated or masked as 0.5.
    """
    target_model = MagicMock()
    target_model.predict_proba.return_value = np.array([[0.5, 0.5]] * 20)

    X_shadow = np.random.rand(200, 2)
    y_shadow = np.random.randint(0, 2, 200)
    X = np.random.rand(20, 2)
    y = np.random.randint(0, 2, 20)

    # Patch attack_clf predict_proba by patching LogisticRegression.predict_proba when called with size matching X_eval (40)
    orig_predict_proba = LogisticRegression.predict_proba
    def mock_predict_proba(self, X_input):
        if len(X_input) == 40: # This is X_eval
            return np.array([[0.5, np.nan]] * len(X_input))
        return orig_predict_proba(self, X_input)

    with patch.object(LogisticRegression, "predict_proba", side_effect=mock_predict_proba, autospec=True):
        res = run_shadow_mia(
            target_model=target_model,
            X_train_target=X,
            y_train_target=y,
            X_test_target=X,
            y_test_target=y,
            X_shadow=X_shadow,
            y_shadow=y_shadow,
            n_shadow=2,
        )

    assert "error" in res
    assert "Attack probability calculation failed" in res["error"] or "non-finite" in res["error"]
    assert res["attack_auc"] is None
    assert res["attacker_advantage"] is None


def test_per_class_mia_error_propagation_no_fallback():
    """
    Verifies that when metric calculation fails in run_per_class_mia,
    it returns an explicit error and AUC is None (not silently 0.5).
    """
    from attack_engine import run_per_class_mia
    target_model = MagicMock()
    target_model.predict_proba.return_value = np.array([[0.5, 0.5]] * 40)

    attack_clf = MagicMock()
    # 20 members, 20 nonmembers => X_eval has 40 samples
    attack_clf.predict.return_value = np.zeros(40, dtype=int)
    attack_clf.predict_proba.return_value = np.full((40, 2), np.nan)

    X = np.random.rand(20, 2)
    y = np.ones(20, dtype=int)

    res = run_per_class_mia(
        target_model=target_model,
        X_train_target=X,
        y_train_target=y,
        X_test_target=X,
        y_test_target=y,
        attack_clf=attack_clf,
        class_label=1,
        min_samples=20,
    )

    assert "error" in res
    assert res["attack_auc"] is None
    assert res["attacker_advantage"] is None
    assert "Per-class MIA metric calculation failed" in res["error"]


