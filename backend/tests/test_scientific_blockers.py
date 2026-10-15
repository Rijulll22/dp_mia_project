import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from privacy_engine import PrivacyEngine, MECHANISM_LAPLACE_CLIPPED, CLIPPING_BOUND
from attack_engine import run_shadow_mia, _extract_model_confidence_features

# ==============================================================================
# TEST 1: ANALYTIC CLIPPED-NOISE EXPECTATION AND DERIVATIVE
# ==============================================================================

def g_analytic(x: float, b: float = 174.0, C: float = 3.0) -> float:
    """Exact expectation g(x) = E[clip(x + Z, -C, C)] for Z ~ Laplace(0, b)."""
    return float(x - b * np.exp(-C / b) * np.sinh(x / b))

def g_prime_analytic(x: float, b: float = 174.0, C: float = 3.0) -> float:
    """Exact derivative g'(x) = 1 - exp(-C/b) * cosh(x/b)."""
    return float(1.0 - np.exp(-C / b) * np.cosh(x / b))

def prob_interior_laplace(x: float, b: float = 174.0, C: float = 3.0) -> float:
    """P(-C < x + Z < C) = F_Z(C - x) - F_Z(-C - x) for Z ~ Laplace(0, b)."""
    F_upper = 1.0 - 0.5 * np.exp(-(C - x) / b)
    F_lower = 0.5 * np.exp(-(C + x) / b)
    return float(F_upper - F_lower)

def test_clipped_noise_expectation_and_derivative():
    b = 174.0
    C = 3.0

    # 1. Verification that g'(x) matches P(-C < x + Z < C) across [0, 1]
    for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gp = g_prime_analytic(x, b, C)
        p_int = prob_interior_laplace(x, b, C)
        assert np.isclose(gp, p_int, atol=1e-12), f"Mismatch at x={x}: gp={gp}, p_int={p_int}"

    # 2. Central finite difference verification of derivative
    eps = 1e-6
    for x in [0.1, 0.5, 0.9]:
        fd = (g_analytic(x + eps, b, C) - g_analytic(x - eps, b, C)) / (2 * eps)
        exact_p = g_prime_analytic(x, b, C)
        assert np.isclose(fd, exact_p, atol=1e-8), f"Finite difference mismatch at x={x}"

    # 3. Disproof of erroneous claim that slope is ~0.0339
    # The actual slope is ~0.01709 (half of 0.0339)
    slope_0 = g_prime_analytic(0.0, b, C)
    slope_1 = g_prime_analytic(1.0, b, C)
    assert 0.0170 <= slope_0 <= 0.0172, f"Slope at 0 is {slope_0}, expected ~0.01709"
    assert 0.0170 <= slope_1 <= 0.0172, f"Slope at 1 is {slope_1}, expected ~0.01708"
    assert not np.isclose(slope_0, 0.0339, atol=0.01), "Slope must NOT be 0.0339"

    # 4. Monte Carlo numerical check against analytic g(x)
    rng = np.random.RandomState(42)
    Z = rng.laplace(0.0, b, size=1_000_000)
    for x in [0.0, 0.5, 1.0]:
        mc_val = float(np.mean(np.clip(x + Z, -C, C)))
        exact_val = g_analytic(x, b, C)
        se = float(np.std(np.clip(x + Z, -C, C), ddof=1) / np.sqrt(len(Z)))
        # Within 3 standard errors
        assert abs(mc_val - exact_val) < 3 * se, f"MC mismatch at x={x}: mc={mc_val}, exact={exact_val}, se={se}"

# ==============================================================================
# TEST 2: LABEL-PERMUTATION PARTITION INTEGRITY & STATISTICAL BEHAVIOR
# ==============================================================================

def test_label_permutation_partition_integrity():
    engine = PrivacyEngine(seed=42)
    df_X, series_y = engine.generate_benchmark_dataset(n_samples=600, n_features=6, fraud_ratio=0.1)

    X_arr = df_X.values
    y_arr = series_y.values.astype(int)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # Split
    X_train_raw, X_temp, y_train, y_temp = train_test_split(X_arr, y_arr, test_size=0.6, random_state=42, stratify=y_arr)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    orig_test_labels = y_test.copy()
    orig_val_labels = y_val.copy()
    orig_train_labels = y_train.copy()

    aucs = []
    for s in [10, 20, 30, 40, 50]:
        rng_p = np.random.RandomState(s)
        y_train_shuf = y_train.copy()
        rng_p.shuffle(y_train_shuf)

        # Integrity assertions:
        # 1. Only y_train was shuffled
        assert np.array_equal(y_test, orig_test_labels), "y_test was contaminated"
        assert np.array_equal(y_val, orig_val_labels), "y_val was contaminated"
        # 2. Class counts in training partition are preserved
        assert np.sum(y_train_shuf == 1) == np.sum(orig_train_labels == 1)
        # 3. Array lengths preserved
        assert len(y_train_shuf) == len(orig_train_labels)

        clf = LogisticRegression(class_weight="balanced", random_state=42)
        clf.fit(X_train_scaled, y_train_shuf)

        test_proba = clf.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, test_proba)
        aucs.append(auc)

    # Over random label permutations, mean ROC-AUC should be centered around chance (~0.5)
    mean_auc = np.mean(aucs)
    assert 0.35 <= mean_auc <= 0.65, f"Mean permutation AUC {mean_auc} deviated too far from 0.5"

# ==============================================================================
# TEST 3: MIA CLASSES REACH FIT & DISTINGUISH CONSTANT SCORES FROM FAILURES
# ==============================================================================

def test_mia_classes_reach_fit_and_distinguish_constant_from_failure():
    # 1. Verify both classes reach attack_clf.fit()
    fit_labels_captured = []

    orig_fit = LogisticRegression.fit
    def capturing_fit(self, X, y, sample_weight=None):
        fit_labels_captured.append(np.unique(y))
        return orig_fit(self, X, y, sample_weight=sample_weight)

    rng = np.random.RandomState(42)
    X_pool = rng.randn(100, 4)
    y_pool = rng.randint(0, 2, 100)

    target_model = MagicMock()
    # Predict constant 0.5
    target_model.predict_proba.return_value = np.array([[0.5, 0.5]] * 20)

    with patch.object(LogisticRegression, "fit", side_effect=capturing_fit, autospec=True):
        res = run_shadow_mia(
            target_model=target_model,
            X_train_target=rng.randn(20, 4),
            y_train_target=rng.randint(0, 2, 20),
            X_test_target=rng.randn(20, 4),
            y_test_target=rng.randint(0, 2, 20),
            X_shadow=X_pool,
            y_shadow=y_pool,
            n_shadow=2,
            seed=42,
        )

    # Check that attack_clf.fit was called with BOTH classes [0, 1]
    # The last call to fit is attack_clf
    assert len(fit_labels_captured) > 0
    attack_labels = fit_labels_captured[-1]
    assert np.array_equal(attack_labels, [0, 1]), f"Attack classifier did not receive both classes: {attack_labels}"

    # 2. Legitimate constant 0.5 predictions should NOT produce an error
    assert "error" not in res
    assert res["attack_auc"] == 0.5
    assert res["attacker_advantage"] == 0.0

    # 3. An actual failure (e.g., metric calculation error) MUST produce an error and AUC None
    with patch("attack_engine.roc_auc_score", side_effect=ValueError("Test metric calculation error")):
        res_fail = run_shadow_mia(
            target_model=target_model,
            X_train_target=rng.randn(20, 4),
            y_train_target=rng.randint(0, 2, 20),
            X_test_target=rng.randn(20, 4),
            y_test_target=rng.randint(0, 2, 20),
            X_shadow=X_pool,
            y_shadow=y_pool,
            n_shadow=2,
            seed=42,
        )

    assert "error" in res_fail
    assert "MIA metric calculation failed" in res_fail["error"]
    assert res_fail["attack_auc"] is None
    assert res_fail["attacker_advantage"] is None


# ==============================================================================
# TEST 4: EXACT L-BFGS GRADIENT AND CONVERGENCE DIAGNOSTIC FOR MIA
# ==============================================================================

def test_mia_lbfgs_gradient_and_convergence_diagnostic():
    """
    Verifies that when member and nonmember confidence distributions are
    nearly identical (|mean_diff| < 1e-5), scikit-learn's exact normalized
    gradient norm at w=0, c=0 is < 1e-4, causing L-BFGS to terminate at
    iteration 0 with zero coefficients. Contrast with distinguishable features.
    """
    from sklearn.linear_model._logistic import LinearModelLoss, HalfBinomialLoss

    rng = np.random.RandomState(42)
    n_per_class = 5000
    n_features = 4

    # 1. Indistinguishable features: difference in means is ~1e-5 (matching empirical shadow MIA features)
    X_base = rng.normal(loc=0.73, scale=0.14, size=(n_per_class, n_features))
    X_members = X_base.copy()
    # Shift nonmembers by empirical magnitude observed in audit: [3.0e-5, 2.7e-5, -1.7e-5, -6.1e-5]
    X_nonmembers = X_base + np.array([3.0e-5, 2.7e-5, -1.7e-5, -6.1e-5])

    X_attack = np.vstack([X_members, X_nonmembers])
    y_attack = np.concatenate([np.ones(n_per_class, dtype=np.float64), np.zeros(n_per_class, dtype=np.float64)])

    loss_calc = LinearModelLoss(base_loss=HalfBinomialLoss(), fit_intercept=True)
    w0 = np.zeros(n_features + 1, dtype=np.float64)

    loss_val, grad_val = loss_calc.loss_gradient(
        coef=w0,
        X=X_attack,
        y=y_attack,
        sample_weight=None,
        l2_reg_strength=1.0,
    )

    # Analytical verification:
    # Under scikit-learn's 1/N scaled loss, grad[w] = 0.25 * (mean_nonmember - mean_member)
    mean_diff = X_nonmembers.mean(axis=0) - X_members.mean(axis=0)
    manual_grad_w = 0.25 * mean_diff
    assert np.allclose(grad_val[:n_features], manual_grad_w, atol=1e-12)
    assert np.isclose(grad_val[-1], 0.0, atol=1e-12)

    grad_norm_inf = float(np.max(np.abs(grad_val)))
    # Because mean_diff is ~1e-5, grad_norm is ~2.5e-6, which is strictly < 1e-4
    assert grad_norm_inf < 1e-4, f"Gradient norm {grad_norm_inf} should be < 1e-4"

    # Fit LogisticRegression with default tol=1e-4
    clf = LogisticRegression(max_iter=500, random_state=42, C=1.0, tol=1e-4)
    clf.fit(X_attack, y_attack)

    assert clf.n_iter_[0] == 0, f"Expected 0 iterations, got {clf.n_iter_[0]}"
    assert np.allclose(clf.coef_, 0.0)
    assert np.isclose(clf.intercept_[0], 0.0)

    # 2. Distinguishable features: member mean 0.9, nonmember mean 0.5 (diff = 0.4)
    X_dist_m = rng.normal(loc=0.9, scale=0.1, size=(n_per_class, n_features))
    X_dist_nm = rng.normal(loc=0.5, scale=0.1, size=(n_per_class, n_features))
    X_dist = np.vstack([X_dist_m, X_dist_nm])

    loss_dist, grad_dist = loss_calc.loss_gradient(
        coef=w0,
        X=X_dist,
        y=y_attack,
        sample_weight=None,
        l2_reg_strength=1.0,
    )

    grad_dist_norm = float(np.max(np.abs(grad_dist)))
    # 0.25 * 0.4 = 0.1 >> 1e-4
    assert grad_dist_norm > 0.05, f"Distinguishable gradient norm {grad_dist_norm} should be > 0.05"

    clf_dist = LogisticRegression(max_iter=500, random_state=42, C=1.0, tol=1e-4)
    clf_dist.fit(X_dist, y_attack)

    assert clf_dist.n_iter_[0] > 0, "Expected > 0 iterations on distinguishable data"
    assert np.linalg.norm(clf_dist.coef_) > 0.1, "Expected non-zero coefficients on distinguishable data"

