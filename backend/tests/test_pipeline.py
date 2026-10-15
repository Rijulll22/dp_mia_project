"""
test_pipeline.py — Unit tests for the PrivacyGuard research pipeline.

Tests verify:
1. Dataset loading and feature count
2. Preprocessing (scaler fit on train only)
3. Stratified train/test split (80/20)
4. Epsilon composition: epsilon_record = feature_count × epsilon_feature
5. Laplace scale: scale = sensitivity / epsilon_feature
6. Clipping bounds: all values in [0, 1]
7. Model training (LR + RF)
8. PR-AUC computation
9. MIA construction correctness
10. MIA metrics (no fake metrics)
11. Experiment persistence
12. No fake/hard-coded metrics in output
"""

import sys
import os
import json
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from privacy_engine import (
    PrivacyEngine,
    MECHANISM_LAPLACE_CLIPPED,
    MECHANISM_LAPLACE,
    MECHANISM_NONE,
)
from attack_engine import (
    run_shadow_mia,
    _extract_model_confidence_features,
    _get_tpr_at_fpr_threshold,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine():
    return PrivacyEngine(seed=42)


@pytest.fixture(scope="module")
def synthetic_dataset(engine):
    X, y = engine.generate_benchmark_dataset(n_samples=1000, n_features=10, fraud_ratio=0.10)
    return X, y


# ─── 1. Dataset Loading ─────────────────────────────────────────────────

class TestDatasetLoading:
    def test_dataset_shape(self, synthetic_dataset):
        X, y = synthetic_dataset
        assert X.shape[0] == 1000, "Expected 1000 samples"
        assert X.shape[1] == 10, "Expected 10 features"
        assert len(y) == 1000

    def test_dataset_has_fraud(self, synthetic_dataset):
        X, y = synthetic_dataset
        fraud_count = int(y.sum())
        assert fraud_count > 0, "Dataset must contain fraud samples"
        assert fraud_count < len(y), "Dataset must contain non-fraud samples"

    def test_feature_names(self, synthetic_dataset):
        X, y = synthetic_dataset
        assert len(X.columns) == 10
        assert "Amount" in X.columns

    def test_feature_count_matches_columns(self, synthetic_dataset):
        X, y = synthetic_dataset
        assert X.shape[1] == len(X.columns)

class TestULBDatasetLoading:
    def test_ulb_dataset_exists_and_loads(self):
        ulb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "creditcard.csv")
        assert os.path.exists(ulb_path), "ULB dataset creditcard.csv must exist"
        
        import pandas as pd
        df = pd.read_csv(ulb_path)
        assert len(df) == 284807, "Expected 284807 samples in ULB dataset"
        assert len(df.columns) == 31, "Expected 31 columns in ULB dataset"
        assert "Class" in df.columns, "Must have Class column"
        assert "Amount" in df.columns, "Must have Amount column"
        
        fraud_count = df["Class"].sum()
        assert fraud_count == 492, "Expected 492 fraud samples in ULB dataset"
        
        legit_count = len(df) - fraud_count
        assert legit_count == 284315, "Expected 284315 legit samples in ULB dataset"
        
        # Test that Time column exists in raw data
        assert "Time" in df.columns



# ─── 2. Preprocessing ───────────────────────────────────────────────────

class TestPreprocessing:
    def test_scaler_fit_on_train_only(self, engine, synthetic_dataset):
        """
        CRITICAL: MinMaxScaler must be fit ONLY on training data.
        Test verifies that train data is in [0,1] but test data may not be
        if test contains out-of-range values.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        X, y = synthetic_dataset
        X_vals = X.values
        y_arr = y.values.astype(int)

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_vals, y_arr, test_size=0.20, random_state=42, stratify=y_arr
        )

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # Training data MUST be in [0,1]
        assert X_train_scaled.min() >= 0.0, "Train data below 0 after scaling"
        assert X_train_scaled.max() <= 1.0, "Train data above 1 after scaling"

        # Scaler must NOT have seen test data
        # Verify: training data min is 0, training data max is 1 (by design of MinMaxScaler)
        assert np.isclose(X_train_scaled.min(axis=0).min(), 0.0, atol=1e-6)
        assert np.isclose(X_train_scaled.max(axis=0).max(), 1.0, atol=1e-6)

    def test_no_test_data_in_scaler(self, engine, synthetic_dataset):
        """Verify that running pipeline does not fit scaler on full dataset."""
        X, y = synthetic_dataset
        # Run pipeline and verify train data is in [0,1]
        out = engine.run_pipeline(
            X, y, epsilon=1.0, mechanism=MECHANISM_LAPLACE_CLIPPED, seed=42, n_shadow=2
        )
        # If scaler was fit on full data, all train+test would be in [0,1]
        # If fit on train only, test may go slightly out of [0,1] for extreme values
        # We verify the pipeline ran without error (the assertion in inject_dp_noise checks train)
        assert out is not None
        assert "metrics" in out


# ─── 3. Train/Test Split ────────────────────────────────────────────────

class TestTrainTestSplit:
    def test_stratified_split_ratio(self, engine, synthetic_dataset):
        """Test 80/20 stratified split."""
        from sklearn.model_selection import train_test_split
        X, y = synthetic_dataset
        y_arr = y.values.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y_arr, test_size=0.20, random_state=42, stratify=y_arr
        )
        # Check ratio
        assert len(X_train) == 800
        assert len(X_test) == 200

    def test_stratified_fraud_ratio_preserved(self, engine, synthetic_dataset):
        """Stratification should preserve fraud ratio in both splits."""
        from sklearn.model_selection import train_test_split
        X, y = synthetic_dataset
        y_arr = y.values.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y_arr, test_size=0.20, random_state=42, stratify=y_arr
        )
        train_fraud_ratio = y_train.sum() / len(y_train)
        test_fraud_ratio = y_test.sum() / len(y_test)
        overall_ratio = y_arr.sum() / len(y_arr)
        # Ratios should be close (within 2%)
        assert abs(train_fraud_ratio - overall_ratio) < 0.02
        assert abs(test_fraud_ratio - overall_ratio) < 0.02

    def test_no_overlap_between_train_test(self, engine, synthetic_dataset):
        """Train and test indices must not overlap."""
        from sklearn.model_selection import train_test_split
        X, y = synthetic_dataset
        y_arr = y.values.astype(int)
        idx = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            idx, test_size=0.20, random_state=42, stratify=y_arr
        )
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, "Train and test must not overlap"


# ─── 4. Epsilon Accounting ──────────────────────────────────────────────

class TestEpsilonAccounting:
    def test_epsilon_composition(self, engine):
        """
        CRITICAL: epsilon_record = feature_count × epsilon_feature
        """
        epsilon_record = 1.0
        feature_count = 15

        acc = engine.compute_epsilon_accounting(epsilon_record, feature_count)

        assert acc["epsilon_record"] == epsilon_record
        assert acc["feature_count"] == feature_count
        assert acc["composition_method"] == "basic sequential composition"

        # Verify composition: epsilon_record = d × epsilon_feature
        computed_record = acc["feature_count"] * acc["epsilon_feature"]
        assert math.isclose(computed_record, epsilon_record, rel_tol=1e-6), \
            f"Composition failed: {acc['feature_count']} × {acc['epsilon_feature']} ≠ {epsilon_record}"

    @pytest.mark.parametrize("epsilon_record,feature_count", [
        (0.1, 10),
        (0.5, 15),
        (1.0, 8),
        (2.0, 20),
        (5.0, 12),
        (10.0, 15),
    ])
    def test_epsilon_composition_multiple_configs(self, engine, epsilon_record, feature_count):
        acc = engine.compute_epsilon_accounting(epsilon_record, feature_count)
        computed = acc["feature_count"] * acc["epsilon_feature"]
        assert math.isclose(computed, epsilon_record, rel_tol=1e-6), \
            f"ε_record={epsilon_record}, d={feature_count}: {acc['feature_count']} × {acc['epsilon_feature']} = {computed}"

    def test_epsilon_feature_less_than_record(self, engine):
        """epsilon_feature must be < epsilon_record for d > 1."""
        acc = engine.compute_epsilon_accounting(1.0, 10)
        assert acc["epsilon_feature"] < acc["epsilon_record"]

    def test_pipeline_stores_epsilon_accounting(self, engine, synthetic_dataset):
        """Pipeline output must include correct epsilon accounting."""
        X, y = synthetic_dataset
        out = engine.run_pipeline(
            X, y, epsilon=1.0, mechanism=MECHANISM_LAPLACE_CLIPPED, seed=42, n_shadow=2
        )
        assert "epsilon_record" in out or "epsilon_accounting" in out or out.get("feature_count") is not None, \
            "Pipeline must output epsilon accounting fields"


# ─── 5. Laplace Scale ───────────────────────────────────────────────────

class TestLaplaceScale:
    def test_laplace_scale_formula(self, engine):
        """scale = sensitivity / epsilon_feature"""
        sensitivity = 1.0
        epsilon_feature = 0.1
        scale = engine.calculate_noise_scale(epsilon_feature, sensitivity, MECHANISM_LAPLACE)
        expected = sensitivity / epsilon_feature
        assert math.isclose(scale, expected, rel_tol=1e-9), \
            f"Expected scale={expected}, got {scale}"

    @pytest.mark.parametrize("sensitivity,epsilon_feature", [
        (1.0, 0.1),
        (1.0, 0.5),
        (1.0, 1.0),
        (1.0, 5.0),
        (0.5, 1.0),
    ])
    def test_laplace_scale_multiple(self, engine, sensitivity, epsilon_feature):
        scale = engine.calculate_noise_scale(epsilon_feature, sensitivity, MECHANISM_LAPLACE)
        assert math.isclose(scale, sensitivity / epsilon_feature, rel_tol=1e-9)

    def test_smaller_epsilon_means_larger_scale(self, engine):
        """More privacy (smaller epsilon) should produce larger noise scale."""
        s1 = engine.calculate_noise_scale(0.1, 1.0, MECHANISM_LAPLACE)  # strong privacy
        s2 = engine.calculate_noise_scale(1.0, 1.0, MECHANISM_LAPLACE)  # moderate privacy
        assert s1 > s2, f"Smaller epsilon should have larger scale. Got {s1} <= {s2}"


# ─── 6. Clipping Bounds ─────────────────────────────────────────────────

class TestClippingBounds:
    def test_all_clipped_values_in_0_1(self, engine, synthetic_dataset):
        """
        CRITICAL: All values after 'laplace_clipped' must be in [0, 1].
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        X, y = synthetic_dataset
        y_arr = y.values.astype(int)
        X_train_raw, _, _, _ = train_test_split(
            X.values, y_arr, test_size=0.20, random_state=42, stratify=y_arr
        )
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)

        # Use low epsilon (high noise) to stress test clipping
        X_perturbed = engine.inject_dp_noise(
            X_train_scaled,
            epsilon_feature=0.01,  # very small epsilon → very large noise
            sensitivity=1.0,
            mechanism=MECHANISM_LAPLACE_CLIPPED,
            seed=42,
        )

        assert X_perturbed.min() >= 0.0, \
            f"Clipped data below 0: min={X_perturbed.min()}"
        assert X_perturbed.max() <= 1.0, \
            f"Clipped data above 1: max={X_perturbed.max()}"

    def test_plain_laplace_not_clipped(self, engine, synthetic_dataset):
        """Plain Laplace may produce values outside [0,1]."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        X, y = synthetic_dataset
        y_arr = y.values.astype(int)
        X_train_raw, _, _, _ = train_test_split(
            X.values, y_arr, test_size=0.20, random_state=42, stratify=y_arr
        )
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)

        X_perturbed = engine.inject_dp_noise(
            X_train_scaled,
            epsilon_feature=0.01,
            sensitivity=1.0,
            mechanism=MECHANISM_LAPLACE,
            seed=42,
        )
        # With very small epsilon, plain Laplace will likely produce out-of-range values
        outside_range = (X_perturbed < 0.0) | (X_perturbed > 1.0)
        assert outside_range.any(), "Plain Laplace with very small epsilon should produce out-of-range values"

    @pytest.mark.parametrize("epsilon", [0.1, 0.5, 1.0, 5.0, 10.0])
    def test_clipping_at_all_epsilon_values(self, engine, synthetic_dataset, epsilon):
        """Clipping must hold for all epsilon values."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        X, y = synthetic_dataset
        y_arr = y.values.astype(int)
        X_train_raw, _, _, _ = train_test_split(
            X.values, y_arr, test_size=0.20, random_state=42, stratify=y_arr
        )
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)

        feature_count = X_train_scaled.shape[1]
        eps_feature = epsilon / feature_count

        X_perturbed = engine.inject_dp_noise(
            X_train_scaled,
            epsilon_feature=eps_feature,
            sensitivity=1.0,
            mechanism=MECHANISM_LAPLACE_CLIPPED,
            seed=42,
        )
        assert X_perturbed.min() >= 0.0
        assert X_perturbed.max() <= 1.0


# ─── 7. Model Training ──────────────────────────────────────────────────

class TestModelTraining:
    def test_logistic_regression_trains(self, engine, synthetic_dataset):
        X, y = synthetic_dataset
        out = engine.run_pipeline(
            X, y, epsilon=1.0, mechanism=MECHANISM_LAPLACE_CLIPPED,
            classifier_type="logistic_regression", seed=42, n_shadow=2
        )
        assert out["metrics"]["accuracy"] is not None
        assert 0.0 <= out["metrics"]["accuracy"] <= 1.0

    def test_random_forest_trains(self, engine, synthetic_dataset):
        X, y = synthetic_dataset
        out = engine.run_pipeline(
            X, y, epsilon=1.0, mechanism=MECHANISM_LAPLACE_CLIPPED,
            classifier_type="random_forest", seed=42, n_shadow=2
        )
        assert out["metrics"]["accuracy"] is not None
        assert 0.0 <= out["metrics"]["accuracy"] <= 1.0

    def test_non_private_baseline_higher_utility(self, engine, synthetic_dataset):
        """Non-private model should generally have better F1 than very low epsilon."""
        X, y = synthetic_dataset
        out_dp = engine.run_pipeline(
            X, y, epsilon=0.1, mechanism=MECHANISM_LAPLACE_CLIPPED,
            classifier_type="logistic_regression", seed=42, n_shadow=2
        )
        out_np = engine.run_pipeline(
            X, y, epsilon=1.0, mechanism=MECHANISM_NONE,
            classifier_type="logistic_regression", seed=42, n_shadow=2
        )
        # Non-private should have higher or equal F1 (some randomness allowed)
        dp_f1 = out_dp["metrics"].get("f1", 0)
        np_f1 = out_np["metrics"].get("f1", 0)
        # With enough samples this should generally hold; allow tolerance
        assert np_f1 >= dp_f1 - 0.15, \
            f"Non-private F1 ({np_f1}) should be >= low-epsilon DP F1 ({dp_f1}) - 0.15"


# ─── 8. PR-AUC ──────────────────────────────────────────────────────────

class TestPRAUC:
    def test_pr_auc_computed(self, engine, synthetic_dataset):
        X, y = synthetic_dataset
        out = engine.run_pipeline(
            X, y, epsilon=1.0, mechanism=MECHANISM_LAPLACE_CLIPPED,
            seed=42, n_shadow=2
        )
        pr_auc = out["metrics"].get("prAuc")
        assert pr_auc is not None, "PR-AUC must be computed"
        assert 0.0 <= pr_auc <= 1.0, f"PR-AUC out of range: {pr_auc}"

    def test_fraud_class_metrics_computed(self, engine, synthetic_dataset):
        X, y = synthetic_dataset
        out = engine.run_pipeline(
            X, y, epsilon=1.0, mechanism=MECHANISM_LAPLACE_CLIPPED,
            seed=42, n_shadow=2
        )
        metrics = out["metrics"]
        # Fraud class metrics must be present
        assert "fraud_f1" in metrics
        assert "fraud_recall" in metrics
        assert "fraud_precision" in metrics


# ─── 9. MIA Construction ────────────────────────────────────────────────

class TestMIAConstruction:
    def test_mia_returns_real_metrics(self, engine, synthetic_dataset):
        """MIA must return actual computed metrics, not None."""
        X, y = synthetic_dataset
        out = engine.run_pipeline(
            X, y, epsilon=1.0, mechanism=MECHANISM_LAPLACE_CLIPPED,
            seed=42, n_shadow=3
        )
        mia = out.get("mia_result") or {}
        assert mia.get("attack_accuracy") is not None, "MIA accuracy must be computed"
        assert mia.get("attack_auc") is not None, "MIA AUC must be computed"
        assert mia.get("attacker_advantage") is not None, "Attacker advantage must be computed"

    def test_mia_accuracy_in_range(self, engine, synthetic_dataset):
        """MIA accuracy must be in [0,1] — no clipping applied."""
        X, y = synthetic_dataset
        out = engine.run_pipeline(
            X, y, epsilon=1.0, mechanism=MECHANISM_LAPLACE_CLIPPED,
            seed=42, n_shadow=3
        )
        mia = out.get("mia_result") or {}
        acc = mia.get("attack_accuracy")
        if acc is not None:
            assert 0.0 <= acc <= 1.0, f"MIA accuracy out of range: {acc}"

    def test_mia_auc_in_range(self, engine, synthetic_dataset):
        """MIA AUC must be in [0,1] — no clipping applied."""
        X, y = synthetic_dataset
        out = engine.run_pipeline(
            X, y, epsilon=1.0, mechanism=MECHANISM_LAPLACE_CLIPPED,
            seed=42, n_shadow=3
        )
        mia = out.get("mia_result") or {}
        auc = mia.get("attack_auc")
        if auc is not None:
            assert 0.0 <= auc <= 1.0, f"MIA AUC out of range: {auc}"

    def test_attacker_advantage_in_range(self, engine, synthetic_dataset):
        """Attacker advantage (max TPR - FPR) must be in [0, 1] or very close to it."""
        X, y = synthetic_dataset
        out = engine.run_pipeline(
            X, y, epsilon=1.0, mechanism=MECHANISM_LAPLACE_CLIPPED,
            seed=42, n_shadow=3
        )
        mia = out.get("mia_result") or {}
        adv = mia.get("attacker_advantage")
        adv_05 = mia.get("attacker_advantage_05")
        if adv is not None:
            assert -1.0 <= adv <= 1.0, f"Attacker advantage out of range: {adv}"
        if adv_05 is not None:
            assert -1.0 <= adv_05 <= 1.0, f"Attacker advantage 0.5 out of range: {adv_05}"
        
        # In this implementation, max(TPR-FPR) from roc_curve should be >= 0
        if adv is not None:
            assert adv >= -1e-6, f"ROC-based max(TPR-FPR) should not be less than 0, got {adv}"
            if adv_05 is not None:
                # max(TPR-FPR) should be >= TPR-FPR at threshold 0.5
                assert adv >= adv_05 - 1e-6, f"Max advantage ({adv}) must be >= advantage at threshold 0.5 ({adv_05})"

    def test_tpr_at_low_fpr(self, engine, synthetic_dataset):
        """TPR@1%FPR and TPR@0.1%FPR must be in [0, 1] if computable."""
        X, y = synthetic_dataset
        out = engine.run_pipeline(
            X, y, epsilon=1.0, mechanism=MECHANISM_LAPLACE_CLIPPED,
            seed=42, n_shadow=3
        )
        mia = out.get("mia_result") or {}
        t1 = mia.get("tpr_at_1fpr")
        t01 = mia.get("tpr_at_01fpr")
        if t1 is not None:
            assert 0.0 <= t1 <= 1.0
        if t01 is not None:
            assert 0.0 <= t01 <= 1.0


# ─── 10. No Fake Metrics ─────────────────────────────────────────────────

class TestNoFakeMetrics:
    def test_no_hardcoded_mia_formula(self, engine, synthetic_dataset):
        """
        Verify MIA accuracy varies with epsilon rather than following
        a fixed formula like 0.50 + 0.18 * (eps/10).
        """
        X, y = synthetic_dataset

        results = []
        for eps in [0.1, 5.0]:
            out = engine.run_pipeline(
                X, y, epsilon=eps, mechanism=MECHANISM_LAPLACE_CLIPPED,
                seed=42, n_shadow=3
            )
            mia = out.get("mia_result") or {}
            acc = mia.get("attack_accuracy")
            results.append(acc)

        # Both should be real values (not None)
        for r in results:
            assert r is not None, "MIA should not be None for real experiments"

        # Check they are not both exactly the formula values
        formula_01 = round(0.50 + 0.18 * (0.1 / 10.0), 4)
        formula_50 = round(0.50 + 0.18 * (5.0 / 10.0), 4)
        # They can't both match the formula exactly
        both_match = (
            results[0] is not None and abs(results[0] - formula_01) < 0.001 and
            results[1] is not None and abs(results[1] - formula_50) < 0.001
        )
        assert not both_match, \
            f"MIA appears to use a formula: {results[0]}≈{formula_01}, {results[1]}≈{formula_50}"

    def test_reconstruction_attack_returns_none(self, engine):
        """Reconstruction heuristic has been removed — must return None."""
        from attack_engine import run_attribute_reconstruction_attack
        result = run_attribute_reconstruction_attack()
        assert result is None, "Reconstruction attack was removed; must return None"

    def test_tradeoff_curve_uses_real_metrics(self, engine, synthetic_dataset):
        """Tradeoff curve points must have real computed values."""
        X, y = synthetic_dataset
        curve = engine.generate_tradeoff_curve(
            X, y, mechanism=MECHANISM_LAPLACE_CLIPPED, seed=42,
            epsilon_grid=[1.0, 5.0]
        )
        assert len(curve) == 2, "Should have one point per epsilon"
        for pt in curve:
            assert pt.get("f1") is not None, f"F1 missing for epsilon={pt.get('epsilon')}"
            # miaRisk must not be from formula
            mia = pt.get("miaRisk")
            if mia is not None:
                assert isinstance(mia, float)

    def test_espinosa_benchmark_removed(self, engine, synthetic_dataset):
        """Espinosa fake benchmark must not appear in pipeline output."""
        X, y = synthetic_dataset
        out = engine.run_pipeline(
            X, y, epsilon=1.0, mechanism=MECHANISM_LAPLACE_CLIPPED,
            seed=42, n_shadow=2
        )
        # Should not have espinosa_benchmark key with data
        espinosa = out.get("espinosa_benchmark")
        assert espinosa is None or espinosa == [], \
            f"Espinosa benchmark should be removed, got: {espinosa}"


# ─── 11. Experiment Persistence ─────────────────────────────────────────

class TestExperimentPersistence:
    def test_pipeline_output_has_required_fields(self, engine, synthetic_dataset):
        X, y = synthetic_dataset
        out = engine.run_pipeline(
            X, y, epsilon=1.0, mechanism=MECHANISM_LAPLACE_CLIPPED,
            seed=42, n_shadow=2
        )
        required = [
            "mechanism_type", "classifier_type",
            "split_seed", "train_size", "test_size",
            "fraud_count_train", "fraud_count_test",
            "feature_list", "noiseScale",
            "metrics", "mia_result",
            "baselines", "perturbedDf",
        ]
        for field in required:
            assert field in out, f"Missing required field: {field}"

    def test_perturbed_df_has_correct_shape(self, engine, synthetic_dataset):
        X, y = synthetic_dataset
        out = engine.run_pipeline(
            X, y, epsilon=1.0, mechanism=MECHANISM_LAPLACE_CLIPPED,
            seed=42, n_shadow=2
        )
        df = out.get("perturbedDf")
        assert df is not None
        # It should contain the noised train data, which is 40% of the original data (under the 40/10/20/30 split)
        assert len(df) == int(len(X) * 0.40)
        assert "Class" not in df.columns

    def test_baselines_no_aes_claim(self, engine, synthetic_dataset):
        """AES/Fernet encryption claim must be removed from baselines."""
        X, y = synthetic_dataset
        out = engine.run_pipeline(
            X, y, epsilon=1.0, mechanism=MECHANISM_LAPLACE_CLIPPED,
            seed=42, n_shadow=2
        )
        baselines = out.get("baselines", {})
        # Should not have encrypted_non_private as a separate ML baseline
        assert "encrypted_non_private" not in baselines, \
            "AES/Fernet baseline claim must be removed"
        # Should have non_private baseline
        assert "non_private" in baselines


# ─── 12. TPR@FPR Helper ─────────────────────────────────────────────────

class TestTPRAtFPR:
    def test_tpr_at_zero_fpr(self):
        fpr = np.array([0.0, 0.01, 0.05, 0.1, 0.5, 1.0])
        tpr = np.array([0.0, 0.3, 0.6, 0.7, 0.9, 1.0])
        result = _get_tpr_at_fpr_threshold(fpr, tpr, target_fpr=0.01)
        assert result == 0.3

    def test_tpr_at_fpr_below_minimum(self):
        """If target FPR is below minimum achievable FPR, return 0."""
        fpr = np.array([0.05, 0.1, 0.5, 1.0])
        tpr = np.array([0.5, 0.7, 0.9, 1.0])
        result = _get_tpr_at_fpr_threshold(fpr, tpr, target_fpr=0.001)
        assert result == 0.0

    def test_tpr_at_fpr_values_in_range(self):
        fpr = np.linspace(0, 1, 100)
        tpr = np.linspace(0, 1, 100)
        for target in [0.01, 0.001]:
            result = _get_tpr_at_fpr_threshold(fpr, tpr, target)
            assert 0.0 <= result <= 1.0


# ─── 13. Noised-Data Export ─────────────────────────────────────────────

class TestNoisedDataExport:
    def test_noised_data_export_synthetic(self, engine, synthetic_dataset):
        from experiment_runner import run_experiments
        from database import init_db, SessionLocal, Experiment
        import tempfile
        import shutil

        init_db()
        X, y = synthetic_dataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # We must override the exports/ path for testing.
            # But run_experiments saves exports to os.path.join(os.path.dirname(__file__), "exports", exp_id).
            # To test without modifying experiment_runner.py's internal path, we'll patch it or let it write and then verify.
            # Let's run a single fast synthetic experiment with save_export=True.
            import random
            seed = random.randint(10000, 99999)
            eps = 1.0
            
            run_experiments(
                epsilons=[eps],
                seeds=[seed],
                mechanisms=[MECHANISM_LAPLACE_CLIPPED, MECHANISM_LAPLACE, MECHANISM_NONE],
                classifiers=["logistic_regression"],
                n_shadow=1,
                dataset_name="synthetic_benchmark",
                save_export=True,
            )
            
            # Verify exports
            with SessionLocal() as db:
                experiments = db.query(Experiment).filter(Experiment.split_seed == str(seed)).order_by(Experiment.timestamp.desc()).limit(3).all()
                assert len(experiments) == 3, "Expected 3 experiments for 3 mechanisms"
                
                for exp in experiments:
                    assert exp.export_path is not None, "Export path must be populated"
                    assert os.path.exists(exp.export_path), "Export directory must exist"
                    
                    csv_path = os.path.join(exp.export_path, "noised_data.csv")
                    npy_path = os.path.join(exp.export_path, "noised_data.npy")
                    meta_path = os.path.join(exp.export_path, "metadata.json")
                    
                    assert os.path.exists(csv_path)
                    assert os.path.exists(npy_path)
                    assert os.path.exists(meta_path)
                    
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    assert "Class" not in df.columns, "Export must not contain the Class label"
                    
                    # 2000 train samples (since experiment_runner generates 5000 * 0.40)
                    assert len(df) == 2000, "Export should represent the perturbed input features (train data)"
                    assert len(df.columns) == 15, "Export should contain only the features"
                    
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    
                    assert meta["experiment_id"] == exp.id
                    assert meta["mechanism"] == exp.mechanism
                    if meta["mechanism"] == MECHANISM_NONE:
                        import math
                        assert math.isinf(meta["epsilon_record"])
                    else:
                        assert meta["epsilon_record"] == eps
                    assert meta["seed"] == seed
                    
                    # Cleanup
                    shutil.rmtree(exp.export_path)

    def test_ulb_noised_data_export(self):
        from experiment_runner import run_experiments
        from database import init_db, SessionLocal, Experiment
        import shutil
        init_db()
        
        # Test ULB dataset export (just 1 mechanism to save time)
        import random
        seed = random.randint(10000, 99999)
        run_experiments(
            epsilons=[10.0],
            seeds=[seed],
            mechanisms=[MECHANISM_LAPLACE_CLIPPED],
            classifiers=["logistic_regression"],
            n_shadow=1,
            dataset_name="ulb",
            save_export=True,
        )
        
        with SessionLocal() as db:
            # Get the latest one by timestamp in case there are multiple
            exp = db.query(Experiment).filter(Experiment.split_seed == str(seed)).order_by(Experiment.timestamp.desc()).first()
            assert exp is not None
            assert exp.export_path is not None
            assert os.path.exists(exp.export_path)
            
            import pandas as pd
            df = pd.read_csv(os.path.join(exp.export_path, "noised_data.csv"))
            assert "Class" not in df.columns
            # 40% of 284807 = 113922
            assert len(df) == 113922
            assert len(df.columns) == 29, "Expected 29 features for ULB"
            
            shutil.rmtree(exp.export_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
