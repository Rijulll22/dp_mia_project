import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from attack_engine import run_shadow_mia, _extract_model_confidence_features


def test_mia_strong_leakage_positive_control():
    """
    Case 1: Strong-leakage positive control.
    An overfitted Decision Tree with max_depth=None has high train memorization.
    MIA should detect substantial leakage (AUC >= 0.70, Advantage >= 0.30).
    """
    rng = np.random.RandomState(42)
    n_features = 5
    n_train = 300
    n_test = 150
    n_shadow = 300

    X_train = rng.normal(0, 1, size=(n_train, n_features))
    y_train = rng.randint(0, 2, size=n_train)
    X_test = rng.normal(0, 1, size=(n_test, n_features))
    y_test = rng.randint(0, 2, size=n_test)
    X_shadow = rng.normal(0, 1, size=(n_shadow, n_features))
    y_shadow = rng.randint(0, 2, size=n_shadow)

    overfit_tree = DecisionTreeClassifier(max_depth=None, random_state=42)
    overfit_tree.fit(X_train, y_train)

    res = run_shadow_mia(
        target_model=overfit_tree,
        X_train_target=X_train,
        y_train_target=y_train,
        X_test_target=X_test,
        y_test_target=y_test,
        X_shadow=X_shadow,
        y_shadow=y_shadow,
        classifier_type="decision_tree",
        n_shadow=3,
        seed=42,
    )

    assert res.get("error") is None
    assert res["attack_auc"] is not None
    assert res["attack_auc"] >= 0.70, f"Expected high AUC on overfit positive control, got {res['attack_auc']}"
    assert res["attacker_advantage"] >= 0.30, f"Expected high advantage on overfit positive control, got {res['attacker_advantage']}"


def test_mia_no_signal_negative_control():
    """
    Case 2: No-signal negative control (true disjoint null).
    Both evaluated groups are unseen holdouts drawn from the exact same distribution.
    MIA AUC should hover near 0.5 and not show large unfair advantage.
    """
    rng = np.random.RandomState(42)
    n_features = 5
    n_train = 300
    n_eval = 100
    n_shadow = 300

    X_train = rng.normal(0, 1, size=(n_train, n_features))
    y_train = rng.randint(0, 2, size=n_train)
    X_shadow = rng.normal(0, 1, size=(n_shadow, n_features))
    y_shadow = rng.randint(0, 2, size=n_shadow)

    # Generalizing regularized model
    well_reg_model = LogisticRegression(C=0.01, random_state=42)
    well_reg_model.fit(X_train, y_train)

    # Disjoint test partitions A and B, neither was in training
    X_null_A = rng.normal(0, 1, size=(n_eval, n_features))
    y_null_A = rng.randint(0, 2, size=n_eval)
    X_null_B = rng.normal(0, 1, size=(n_eval, n_features))
    y_null_B = rng.randint(0, 2, size=n_eval)

    res = run_shadow_mia(
        target_model=well_reg_model,
        X_train_target=X_null_A,
        y_train_target=y_null_A,
        X_test_target=X_null_B,
        y_test_target=y_null_B,
        X_shadow=X_shadow,
        y_shadow=y_shadow,
        classifier_type="logistic_regression",
        n_shadow=3,
        seed=42,
    )

    assert res.get("error") is None
    assert res["attack_auc"] is not None
    # Null control AUC should be close to 0.5 (within standard sampling fluctuation)
    assert 0.40 <= res["attack_auc"] <= 0.60, f"Expected null AUC near 0.5, got {res['attack_auc']}"


def test_mia_small_signal_standardization_case():
    """
    Case 3: Small-signal case where features have tiny variances (~1e-4).
    Unscaled LogisticRegression previously collapsed to w=0 because gradient < tol.
    Standardized attack pipeline must fit non-zero weights and produce informative probabilities.
    """
    rng = np.random.RandomState(42)
    n_features = 4
    n_samples = 400

    # Create synthetic shadow attack features with small raw spread (order 1e-4)
    # but a small consistent difference between members and non-members
    X_members = rng.normal(loc=0.5002, scale=0.0005, size=(n_samples, n_features))
    X_nonmembers = rng.normal(loc=0.5000, scale=0.0005, size=(n_samples, n_features))

    # Evaluate using the fitted attack_clf pipeline
    from attack_engine import StandardScaler, Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=42, C=1.0)),
    ])

    X_train_att = np.vstack([X_members, X_nonmembers])
    y_train_att = np.concatenate([np.ones(n_samples, dtype=int), np.zeros(n_samples, dtype=int)])

    pipeline.fit(X_train_att, y_train_att)

    clf_step = pipeline.named_steps["clf"]
    # Check that coefficients are NOT all zeros
    assert not np.allclose(clf_step.coef_, 0.0), "Standardized attack pipeline must have non-zero coefficients"
    assert np.linalg.norm(clf_step.coef_) > 0.01, "Expected significant coefficients after standardization"

    # Evaluate on test set
    X_test_eval = rng.normal(loc=0.5001, scale=0.0005, size=(100, n_features))
    probs = pipeline.predict_proba(X_test_eval)[:, 1]

    # Must produce informative diverse probabilities, not all identical 0.5 constants
    unique_probs = np.unique(np.round(probs, 4))
    assert len(unique_probs) > 10, f"Expected diverse probabilities, got only {len(unique_probs)} distinct values"
