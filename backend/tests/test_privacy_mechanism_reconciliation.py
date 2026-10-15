import pytest
import numpy as np
import math
from privacy_engine import (
    PrivacyEngine,
    MECHANISM_LAPLACE_CLIPPED,
    MECHANISM_LAPLACE,
    MECHANISM_NONE,
    CLIPPING_BOUND,
    DEFAULT_SENSITIVITY,
)


def test_reconciled_sensitivity_and_scale():
    """
    Verifies noise scale calibration under both:
    1. Spec sensitivity Delta f = 1.0 (for [0, 1] normalized data)
    2. Legacy runner sensitivity Delta f = 6.0 (for [-3, 3] clipping)
    """
    engine = PrivacyEngine(seed=42)
    feature_count = 29
    epsilon_record = 1.0

    acc = engine.compute_epsilon_accounting(epsilon_record, feature_count)
    eps_feat = acc["epsilon_feature"]

    # 1. Spec sensitivity Delta f = 1.0
    scale_spec = engine.calculate_noise_scale(eps_feat, sensitivity=1.0, mechanism=MECHANISM_LAPLACE_CLIPPED)
    expected_spec_scale = 1.0 / (1.0 / 29.0) # 29.0
    assert math.isclose(scale_spec, expected_spec_scale, rel_tol=1e-5), f"Expected 29.0, got {scale_spec}"

    # 2. Legacy runner sensitivity Delta f = 6.0
    scale_legacy = engine.calculate_noise_scale(eps_feat, sensitivity=DEFAULT_SENSITIVITY, mechanism=MECHANISM_LAPLACE_CLIPPED)
    expected_legacy_scale = 6.0 * 29.0 # 174.0
    assert math.isclose(scale_legacy, expected_legacy_scale, rel_tol=1e-5), f"Expected 174.0, got {scale_legacy}"


def test_clipping_bounds_adaptation():
    """
    Verifies that clipping bounds adapt safely:
    - Delta f <= 1.0 bounds to [0.0, 1.0]
    - Delta f > 1.0 bounds to [-CLIPPING_BOUND, CLIPPING_BOUND] ([-3.0, 3.0])
    """
    engine = PrivacyEngine(seed=42)
    X_input = np.array([
        [-5.0, 5.0],
        [0.5, 0.5],
        [0.0, 1.0],
    ])

    eps_feat = 0.001 # Large noise

    # Spec: sensitivity=1.0 -> strictly in [0.0, 1.0]
    X_pert_spec = engine.inject_dp_noise(X_input, epsilon_feature=eps_feat, sensitivity=1.0, mechanism=MECHANISM_LAPLACE_CLIPPED, seed=42)
    assert np.all(X_pert_spec >= 0.0) and np.all(X_pert_spec <= 1.0), f"Spec bounds violated: min={X_pert_spec.min()}, max={X_pert_spec.max()}"

    # Legacy: sensitivity=6.0 -> strictly in [-3.0, 3.0]
    X_pert_legacy = engine.inject_dp_noise(X_input, epsilon_feature=eps_feat, sensitivity=6.0, mechanism=MECHANISM_LAPLACE_CLIPPED, seed=42)
    assert np.all(X_pert_legacy >= -3.0) and np.all(X_pert_legacy <= 3.0), f"Legacy bounds violated: min={X_pert_legacy.min()}, max={X_pert_legacy.max()}"


def test_composition_accounting():
    """
    Verifies basic sequential composition across feature dimension d.
    Total budget epsilon_record is partitioned into epsilon_record / d per feature.
    """
    engine = PrivacyEngine(seed=42)
    for eps in [0.1, 0.5, 1.0, 5.0, 10.0]:
        for d in [10, 15, 29, 30]:
            acc = engine.compute_epsilon_accounting(eps, d)
            assert acc["feature_count"] == d
            assert acc["epsilon_record"] == eps
            assert math.isclose(acc["epsilon_feature"] * d, eps, rel_tol=1e-5)
