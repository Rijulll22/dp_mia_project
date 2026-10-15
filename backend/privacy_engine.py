"""
privacy_engine.py — Scientifically Defensible Differential Privacy Pipeline

Key corrections from previous version:
1. TRAIN/TEST LEAKAGE FIX: MinMaxScaler is fit on training data only, then
   transforms both train and test separately.
2. EPSILON ACCOUNTING FIX: Applies composition — record-level epsilon
   = d × epsilon_feature. The input epsilon is treated as record-level;
   per-feature epsilon = epsilon_record / d.
3. TERMINOLOGY FIX: Renamed "truncated_laplace" / "Croft mechanism" to
   "laplace_clipped" = Laplace perturbation with bounded post-processing clip.
4. AES/FERNET CLAIM REMOVED: No encryption code exists; claim removed.
5. ESPINOSA BENCHMARK REMOVED: Hard-coded fake benchmark values removed.
6. FAKE MIA FORMULA REMOVED: tradeoff curve no longer uses formula-based MIA.
7. RECONSTRUCTION HEURISTIC REMOVED: Not a real attack; documented as future work.
8. PR-AUC ADDED: average_precision_score computed on test set.
9. FRAUD-CLASS METRICS ADDED: Per-class precision/recall/F1 for minority class.
"""

import json
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, Callable
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import MinMaxScaler

from attack_engine import run_shadow_mia, run_per_class_mia

# Supported mechanisms
MECHANISM_LAPLACE_CLIPPED = "laplace_clipped"   # Laplace + bounded clipping to [-C, C]
MECHANISM_LAPLACE = "plain_laplace"              # Unbounded Laplace
MECHANISM_GAUSSIAN = "gaussian"                  # (ε,δ)-Gaussian mechanism
MECHANISM_NONE = "non_private"                   # No perturbation

CLIPPING_BOUND = 3.0
DEFAULT_SENSITIVITY = 2.0 * CLIPPING_BOUND


class PrivacyEngine:
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def generate_benchmark_dataset(
        self, n_samples: int = 6000, n_features: int = 15, fraud_ratio: float = 0.05
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generates a realistic synthetic tabular dataset modeled on credit card transactions.
        Clearly labeled as synthetic — not presented as a real dataset.
        """
        rng = np.random.RandomState(self.seed)

        n_fraud = int(n_samples * fraud_ratio)
        n_normal = n_samples - n_fraud

        X_normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, n_features))
        y_normal = np.zeros(n_normal, dtype=int)

        X_fraud = rng.normal(loc=1.2, scale=1.8, size=(n_fraud, n_features))
        y_fraud = np.ones(n_fraud, dtype=int)

        X = np.vstack([X_normal, X_fraud])
        y = np.concatenate([y_normal, y_fraud])

        indices = np.arange(n_samples)
        rng.shuffle(indices)
        X = X[indices]
        y = y[indices]

        columns = [f"V{i+1}" for i in range(n_features - 1)] + ["Amount"]
        df_X = pd.DataFrame(X, columns=columns)
        df_X["Amount"] = np.abs(df_X["Amount"] * 85.0 + 15.0).round(2)

        series_y = pd.Series(y, name="Class")
        return df_X, series_y

    # ─── Epsilon Accounting ────────────────────────────────────────────────
    def compute_epsilon_accounting(
        self,
        epsilon_record: float,
        feature_count: int,
    ) -> Dict[str, Any]:
        """
        Correct epsilon composition under basic sequential composition.

        Under basic composition: ε_record = d × ε_feature
        Therefore: ε_feature = ε_record / d

        Parameters
        ----------
        epsilon_record : total desired record-level privacy budget
        feature_count : number of numerical features to be perturbed

        Returns
        -------
        dict with epsilon_record, epsilon_feature, feature_count, composition_method
        """
        if feature_count <= 0:
            raise ValueError("feature_count must be > 0")
        epsilon_feature = epsilon_record / feature_count
        return {
            "epsilon_record": float(epsilon_record),
            "epsilon_feature": round(float(epsilon_feature), 8),
            "feature_count": int(feature_count),
            "composition_method": "basic sequential composition",
        }

    def calculate_noise_scale(
        self,
        epsilon_feature: float,
        sensitivity: float,
        mechanism: str = MECHANISM_LAPLACE_CLIPPED,
        delta: float = 0.00001,
    ) -> float:
        """
        Computes noise scale calibrated to per-feature epsilon.

        For Laplace: scale = sensitivity / epsilon_feature
        For Gaussian: sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon_feature

        IMPORTANT: epsilon_feature is the per-feature budget, NOT record-level.
        """
        m = mechanism.lower().replace(" ", "_")
        if m == MECHANISM_GAUSSIAN:
            if delta <= 0 or delta >= 1:
                delta = 0.00001
            sigma = (sensitivity * math.sqrt(2.0 * math.log(1.25 / delta))) / epsilon_feature
            return float(sigma)
        else:
            # Laplace (both clipped and unclipped) use same scale formula
            return float(sensitivity / epsilon_feature)

    # ─── DP Noise Injection ────────────────────────────────────────────────
    def inject_dp_noise(
        self,
        X: np.ndarray,
        epsilon_feature: float,
        sensitivity: float = DEFAULT_SENSITIVITY,
        mechanism: str = MECHANISM_LAPLACE_CLIPPED,
        delta: float = 0.00001,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Injects DP noise using per-feature epsilon.

        Mechanisms:
        - 'laplace_clipped': Laplace noise + clip to [0,1]. Called
          "Laplace perturbation with bounded clipping" — NOT "Croft mechanism."
          The features must be normalized to [0,1] before calling this.
        - 'plain_laplace': Unbounded Laplace noise.
        - 'gaussian': Gaussian mechanism for (ε,δ)-DP.
        - 'non_private': No noise, returns X unchanged.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features), normalized to [0,1]
        epsilon_feature : PER-FEATURE epsilon (not record-level)
        sensitivity : L1/L2 sensitivity of a single feature
        mechanism : one of the MECHANISM_* constants
        delta : used for Gaussian only
        seed : random seed for reproducibility

        Returns
        -------
        X_perturbed : numpy array, same shape as X
        """
        if mechanism == MECHANISM_NONE:
            return X.copy()

        rng = np.random.RandomState(seed)
        scale = self.calculate_noise_scale(epsilon_feature, sensitivity, mechanism, delta)

        if mechanism == MECHANISM_GAUSSIAN:
            noise = rng.normal(loc=0.0, scale=scale, size=X.shape)
            return X + noise

        elif mechanism == MECHANISM_LAPLACE_CLIPPED:
            # Reconciled clipping bounds:
            # - If sensitivity <= 1.0: features are normalized in [0, 1] (Δf = 1.0 per spec),
            #   so clipping bounds are [0.0, 1.0].
            # - If sensitivity > 1.0: legacy bounds [-CLIPPING_BOUND, CLIPPING_BOUND] (e.g. C=3.0, Δf = 6.0).
            if sensitivity <= 1.0:
                c_min, c_max = 0.0, 1.0
            else:
                c_min, c_max = -CLIPPING_BOUND, CLIPPING_BOUND
            X_clipped = np.clip(X, c_min, c_max)
            noise = rng.laplace(loc=0.0, scale=scale, size=X_clipped.shape)
            X_perturbed = X_clipped + noise
            X_perturbed = np.clip(X_perturbed, c_min, c_max)
            return X_perturbed

        else:  # plain_laplace
            noise = rng.laplace(loc=0.0, scale=scale, size=X.shape)
            return X + noise

    def make_dp_perturb_fn(
        self,
        epsilon_feature: float,
        sensitivity: float = DEFAULT_SENSITIVITY,
        mechanism: str = MECHANISM_LAPLACE_CLIPPED,
        delta: float = 0.00001,
        seed: int = 42,
    ) -> Callable:
        """
        Returns a closure suitable for passing to run_shadow_mia as dp_perturb_fn.
        The closure applies the same DP mechanism and epsilon_feature as the target.
        """
        def perturb_fn(X: np.ndarray) -> np.ndarray:
            return self.inject_dp_noise(
                X,
                epsilon_feature=epsilon_feature,
                sensitivity=sensitivity,
                mechanism=mechanism,
                delta=delta,
                seed=seed,
            )
        return perturb_fn

    # ─── Classifier Factory ────────────────────────────────────────────────
    def _create_classifier(self, classifier_type: str = "logistic_regression", seed: Optional[int] = None):
        """Instantiate either Logistic Regression or Random Forest classifier."""
        s = seed if seed is not None else self.seed
        c_type = classifier_type.lower().replace(" ", "_")
        if c_type in ["random_forest", "rf"]:
            return RandomForestClassifier(
                n_estimators=50, max_depth=8, class_weight="balanced", random_state=s, n_jobs=-1
            )
        return LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=s, C=1.0
        )

    # ─── Metric Computation ───────────────────────────────────────────────
    def _tune_threshold(self, clf: Any, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Selects the classification threshold that maximizes F1 score on the validation set.
        
        Principled decision rule:
        1. Evaluates candidate thresholds strictly on validation data.
        2. Rejects degenerate candidate thresholds that predict all-positive or all-negative.
        3. Computes the trivial all-positive baseline F1 score based on validation prevalence:
           F1_trivial = 2 * prevalence / (prevalence + 1.0).
        4. If no non-degenerate candidate strictly beats F1_trivial, falls back to default 0.5
           to avoid degenerate 100% false-alarm collapse under severely noisy/degraded models.
        """
        try:
            y_val_proba = clf.predict_proba(X_val)[:, 1]
            n_val = len(y_val)
            if n_val == 0:
                return 0.5
            n_pos = int(np.sum(y_val == 1))
            if n_pos == 0 or n_pos == n_val:
                return 0.5

            prevalence = n_pos / n_val
            f1_trivial_positive = (2.0 * prevalence) / (prevalence + 1.0)

            best_f1 = f1_trivial_positive
            best_threshold = 0.5
            found_valid_candidate = False

            for th in np.linspace(0.01, 0.99, 99):
                preds = (y_val_proba >= th).astype(int)
                # Reject degenerate trivial predictions (all 0s or all 1s)
                if np.all(preds == 0) or np.all(preds == 1):
                    continue
                f1 = f1_score(y_val, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = th
                    found_valid_candidate = True

            if not found_valid_candidate:
                return 0.5
            return float(best_threshold)
        except Exception:
            return 0.5

    def evaluate_model_metrics(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Calculates comprehensive classification metrics on test partition.
        ROC-AUC and PR-AUC require continuous positive-class scores (from
        predict_proba[:, 1] or decision_function). They are NEVER computed
        from hard class predictions. If continuous scores are unavailable or
        non-finite, ROC-AUC and PR-AUC are None.
        Threshold-dependent metrics (precision, recall, f1, accuracy) use
        predictions evaluated at the specified validation-tuned threshold.
        """
        # 1. Obtain continuous positive-class scores
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_test)
                if probs.ndim == 2 and probs.shape[1] >= 2:
                    y_proba = probs[:, 1]
                elif probs.ndim == 1:
                    y_proba = probs
            except Exception:
                y_proba = None

        if y_proba is None and hasattr(model, "decision_function"):
            try:
                scores = model.decision_function(X_test)
                if scores.ndim == 1:
                    y_proba = scores
            except Exception:
                y_proba = None

        # Verify finiteness of continuous scores
        if y_proba is not None and not np.all(np.isfinite(y_proba)):
            y_proba = None

        # 2. Binary predictions at validation-tuned threshold
        if y_proba is not None:
            y_pred = (y_proba >= threshold).astype(int)
        else:
            try:
                y_pred = model.predict(X_test).astype(int)
            except Exception:
                y_pred = np.zeros(len(y_test), dtype=int)

        # Overall metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        # Continuous score metrics (only computed if y_proba is valid and test set has both classes)
        roc_auc = None
        pr_auc = None
        if y_proba is not None and len(np.unique(y_test)) >= 2:
            try:
                roc_auc = float(roc_auc_score(y_test, y_proba))
            except Exception:
                roc_auc = None

            try:
                pr_auc = float(average_precision_score(y_test, y_proba, pos_label=1))
            except Exception:
                pr_auc = None

        # Fraud class metrics (class=1, the minority class)
        try:
            p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
                y_test, y_pred, labels=[0, 1], zero_division=0
            )
            fraud_precision = float(p_per_class[1]) if len(p_per_class) > 1 else None
            fraud_recall = float(r_per_class[1]) if len(r_per_class) > 1 else None
            fraud_f1 = float(f1_per_class[1]) if len(f1_per_class) > 1 else None
            fraud_support = int(support[1]) if len(support) > 1 else None
        except Exception:
            fraud_precision = fraud_recall = fraud_f1 = fraud_support = None

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "rocAuc": round(roc_auc, 4) if roc_auc is not None else None,
            "prAuc": round(pr_auc, 4) if pr_auc is not None else None,
            "fraud_precision": round(fraud_precision, 4) if fraud_precision is not None else None,
            "fraud_recall": round(fraud_recall, 4) if fraud_recall is not None else None,
            "fraud_f1": round(fraud_f1, 4) if fraud_f1 is not None else None,
            "fraud_support": fraud_support,
            "predicted_frauds": int(np.sum(y_pred == 1)),
        }

    # ─── Main Pipeline ────────────────────────────────────────────────────
    def run_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        epsilon: float = 1.0,
        delta: float = 0.00001,
        sensitivity: float = DEFAULT_SENSITIVITY,
        mechanism: str = MECHANISM_LAPLACE_CLIPPED,
        classifier_type: str = "logistic_regression",
        test_size: float = 0.20,
        seed: int = 42,
        n_shadow: int = 5,
    ) -> Dict[str, Any]:
        """
        Full differential privacy training and evaluation pipeline.

        Pipeline:
        1. Stratified train/test split (80/20 default)
        2. Fit MinMaxScaler on TRAIN ONLY, transform both train and test
        3. Compute epsilon_feature = epsilon_record / d
        4. Inject DP noise into TRAINING data only
        5. Train target model on perturbed training data
        6. Evaluate on CLEAN test data
        7. Run genuine shadow-model MIA
        8. Store all metadata

        No fake metrics, no formula-generated curves, no hard-coded values.

        Parameters
        ----------
        X : raw feature DataFrame
        y : binary labels (0=normal, 1=fraud)
        epsilon : record-level epsilon (total privacy budget for one record)
        delta : used for Gaussian mechanism only
        sensitivity : per-feature L1 sensitivity (1.0 after MinMax normalization)
        mechanism : one of MECHANISM_* constants
        classifier_type : "logistic_regression" or "random_forest"
        test_size : fraction for test set (default 0.20)
        seed : random seed
        n_shadow : number of shadow models for MIA
        """
        # Normalize mechanism name
        m_norm = self._normalize_mechanism(mechanism)

        y_arr = y.values.astype(int)
        X_values = X.values

        # ── Step 1: Disjoint stratified partitions (40% train, 10% val, 20% test, 30% shadow) ───
        n_samples = len(X_values)
        if n_samples < 4:
            raise ValueError(f"Dataset too small ({n_samples} samples). Minimum 4 required for disjoint partitions.")

        try:
            # First split: 40% train, 60% temp
            X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
                X_values, y_arr, test_size=0.60, random_state=seed, stratify=y_arr
            )
            # Second split: 10% val (which is 1/6 of 60%), 50% temp2
            X_val_raw, X_temp2_raw, y_val, y_temp2 = train_test_split(
                X_temp_raw, y_temp, test_size=(5.0/6.0), random_state=seed, stratify=y_temp
            )
            # Third split: 20% test (which is 2/5 of 50%), 30% shadow
            X_test_raw, X_shadow_raw, y_test, y_shadow = train_test_split(
                X_temp2_raw, y_temp2, test_size=(3.0/5.0), random_state=seed, stratify=y_temp2
            )
        except ValueError:
            # Fallback without stratification
            try:
                X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
                    X_values, y_arr, test_size=0.60, random_state=seed
                )
                X_val_raw, X_temp2_raw, y_val, y_temp2 = train_test_split(
                    X_temp_raw, y_temp, test_size=(5.0/6.0), random_state=seed
                )
                X_test_raw, X_shadow_raw, y_test, y_shadow = train_test_split(
                    X_temp2_raw, y_temp2, test_size=(3.0/5.0), random_state=seed
                )
            except ValueError:
                raise ValueError("Dataset is too small to form disjoint partitions.")

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            raise ValueError("Disjoint partitions formed, but training or test set does not contain both classes.")

        # ── Step 2: Fit scaler on TRAIN ONLY ──────────────────────────────
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        X_shadow_scaled = scaler.transform(X_shadow_raw)

        # Verify normalization: training data should be in [0,1]
        # (test data may exceed this range for extreme values)
        assert X_train_scaled.min() >= -1e-5 and X_train_scaled.max() <= 1.0 + 1e-5, \
            f"Training data normalization failed: values outside [0,1], min={X_train_scaled.min()}, max={X_train_scaled.max()}"

        # ── Step 3: Epsilon accounting ─────────────────────────────────────
        feature_count = X_train_scaled.shape[1]
        if m_norm == MECHANISM_NONE:
            epsilon_feature = float("inf")
            noise_scale = 0.0
            eps_accounting = {
                "epsilon_record": float("inf"),
                "epsilon_feature": float("inf"),
                "feature_count": feature_count,
                "composition_method": "non-private (no DP)",
            }
        else:
            eps_accounting = self.compute_epsilon_accounting(epsilon, feature_count)
            epsilon_feature = eps_accounting["epsilon_feature"]
            noise_scale = self.calculate_noise_scale(epsilon_feature, sensitivity, m_norm, delta)

        # ── Step 4: Inject DP noise into TRAINING data only ───────────────
        if m_norm == MECHANISM_NONE:
            X_train_dp = X_train_scaled.copy()
            dp_perturb_fn = None
        else:
            X_train_dp = self.inject_dp_noise(
                X_train_scaled,
                epsilon_feature=epsilon_feature,
                sensitivity=sensitivity,
                mechanism=m_norm,
                delta=delta,
                seed=seed,
            )
            dp_perturb_fn = self.make_dp_perturb_fn(
                epsilon_feature=epsilon_feature,
                sensitivity=sensitivity,
                mechanism=m_norm,
                delta=delta,
                seed=seed,
            )

        # ── Step 5: Train target model on DP-perturbed training data ──────
        clf_target = self._create_classifier(classifier_type, seed)
        clf_target.fit(X_train_dp, y_train)

        # ── Step 6: Threshold tuning on validation, then evaluate on clean test data ───
        best_threshold = self._tune_threshold(clf_target, X_val_scaled, y_val)
        dp_metrics = self.evaluate_model_metrics(clf_target, X_test_scaled, y_test, threshold=best_threshold)

        # Confusion matrix
        try:
            y_pred = (clf_target.predict_proba(X_test_scaled)[:, 1] >= best_threshold).astype(int)
        except Exception:
            y_pred = clf_target.predict(X_test_scaled)

        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        else:
            tn = int(cm[0, 0]) if cm.shape[0] >= 1 else 0
            fp = fn = tp = 0

        # ── Step 7: Non-private baseline (same split, no DP) ──────────────
        clf_baseline = self._create_classifier(classifier_type, seed)
        clf_baseline.fit(X_train_scaled, y_train)
        baseline_threshold = self._tune_threshold(clf_baseline, X_val_scaled, y_val)
        baseline_metrics = self.evaluate_model_metrics(clf_baseline, X_test_scaled, y_test, threshold=baseline_threshold)

        baselines = {
            "non_private": {
                **baseline_metrics,
                "description": "Standard training on unperturbed data (epsilon = inf)",
                "best_threshold": baseline_threshold,
            },
        }

        # ── Step 8: Model comparison (LR vs RF on same DP data) ───────────
        lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        lr_model.fit(X_train_dp, y_train)
        lr_metrics = self.evaluate_model_metrics(lr_model, X_test_scaled, y_test)

        rf_model = RandomForestClassifier(
            n_estimators=50, max_depth=8, class_weight="balanced", random_state=seed, n_jobs=-1
        )
        rf_model.fit(X_train_dp, y_train)
        rf_metrics = self.evaluate_model_metrics(rf_model, X_test_scaled, y_test)

        model_comparison = {
            "logistic_regression": lr_metrics,
            "random_forest": rf_metrics,
        }

        # ── Step 9: Genuine shadow-model MIA ──────────────────────────────
        mia_result = run_shadow_mia(
            target_model=clf_target,
            X_train_target=X_train_scaled, # Query clean data for members!
            y_train_target=y_train,
            X_test_target=X_test_scaled,
            y_test_target=y_test,
            X_shadow=X_shadow_scaled,
            y_shadow=y_shadow,
            classifier_type=classifier_type,
            n_shadow=n_shadow,
            seed=seed,
            dp_perturb_fn=dp_perturb_fn,
        )
        attack_clf = mia_result.pop("attack_clf", None)

        mia_per_class = {
            "all": mia_result,
            "legitimate": None,
            "fraud": None,
        }
        fraud_count_train = int(np.sum(y_train == 1))
        fraud_count_test = int(np.sum(y_test == 1))
        
        if attack_clf is not None:
            mia_fraud = run_per_class_mia(
                target_model=clf_target,
                X_train_target=X_train_scaled,
                y_train_target=y_train,
                X_test_target=X_test_scaled,
                y_test_target=y_test,
                attack_clf=attack_clf,
                class_label=1,
                seed=seed,
            )
            mia_per_class["fraud"] = mia_fraud if mia_fraud else {"note": "Insufficient fraud samples."}
            
            mia_legit = run_per_class_mia(
                target_model=clf_target,
                X_train_target=X_train_scaled,
                y_train_target=y_train,
                X_test_target=X_test_scaled,
                y_test_target=y_test,
                attack_clf=attack_clf,
                class_label=0,
                seed=seed,
            )
            mia_per_class["legitimate"] = mia_legit if mia_legit else {"note": "Insufficient legit samples."}

        # ── Step 10: Build perturbed dataset for export ───────────────────
        # Return EXACTLY the noised feature matrix passed to the model (X_train_dp)
        # Do not regenerate noise. Do not include Class labels.
        perturbed_df = pd.DataFrame(
            np.round(X_train_dp, 5), columns=list(X.columns)
        )

        # ── Step 11: Mechanism label (correct terminology) ─────────────────
        mechanism_label = {
            MECHANISM_LAPLACE_CLIPPED: "Laplace perturbation with bounded clipping",
            MECHANISM_LAPLACE: "Laplace perturbation (unbounded)",
            MECHANISM_GAUSSIAN: "Gaussian mechanism",
            MECHANISM_NONE: "Non-private",
        }.get(m_norm, m_norm)

        # ── Step 12: Relative utility loss vs non-private baseline ─────────
        base_f1 = float(baseline_metrics["f1"]) if baseline_metrics.get("f1") else None
        dp_f1 = float(dp_metrics["f1"]) if dp_metrics.get("f1") else None
        if base_f1 and base_f1 > 0 and dp_f1 is not None:
            relative_utility_loss = max(0.0, round((base_f1 - dp_f1) / base_f1 * 100, 2))
        else:
            relative_utility_loss = None

        # ── Step 13: Model artifacts for download ──────────────────────────
        model_artifacts = {
            "model_type": classifier_type,
            "framework": "scikit-learn",
            "epsilon_record": eps_accounting["epsilon_record"],
            "epsilon_feature": eps_accounting["epsilon_feature"],
            "feature_count": eps_accounting["feature_count"],
            "composition_method": eps_accounting["composition_method"],
            "delta": float(delta),
            "sensitivity": float(sensitivity),
            "mechanism": m_norm,
            "mechanism_label": mechanism_label,
            "features": list(X.columns),
            "noise_scale": round(noise_scale, 6),
        }
        if hasattr(clf_target, "coef_"):
            model_artifacts["coefficients"] = [round(float(c), 6) for c in clf_target.coef_[0].tolist()]
            model_artifacts["intercept"] = [round(float(i), 6) for i in clf_target.intercept_.tolist()]
        if hasattr(clf_target, "feature_importances_"):
            model_artifacts["feature_importances"] = [
                round(float(fi), 6) for fi in clf_target.feature_importances_.tolist()
            ]

        return {
            # Mechanism metadata
            "mechanism_type": m_norm,
            "mechanism_label": mechanism_label,
            "classifier_type": classifier_type,

            # Epsilon accounting
            "epsilon_record": eps_accounting["epsilon_record"],
            "epsilon_feature": eps_accounting["epsilon_feature"],
            "feature_count": eps_accounting["feature_count"],
            "composition_method": eps_accounting["composition_method"],
            "noiseScale": round(noise_scale, 4),

            # Dataset split metadata
            "split_seed": seed,
            "train_size": int(len(X_train_raw)),
            "test_size": int(len(X_test_raw)),
            "fraud_count_train": fraud_count_train,
            "fraud_count_test": fraud_count_test,
            "feature_list": list(X.columns),

            # Metrics
            "metrics": {
                **dp_metrics,
                "confusionMatrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
                # Legacy fields (kept for schema compat, not used as research claims)
                "miaRisk": mia_result.get("attack_accuracy"),
                "reconRisk": None,  # REMOVED — was a heuristic, not a real attack
            },
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "processed_rows": int(len(y_test)),

            # MIA results
            "mia_result": mia_result,
            "mia_per_class": mia_per_class,

            # Baselines (non-private only — no fake AES baseline)
            "baselines": baselines,

            # Model comparison
            "model_comparison": model_comparison,

            # Relative utility loss
            "relative_utility_loss": relative_utility_loss,
            "best_threshold": float(best_threshold),

            # Export artifacts
            "perturbedDf": perturbed_df,
            "modelArtifacts": model_artifacts,
            "classifier": clf_target,

            # Legacy compat fields
            "empirical_attacks": {
                "mia": mia_result,
                "reconstruction": None,  # REMOVED
            },
            "tradeoffCurve": [],  # Populated by generate_tradeoff_curve separately
            "pareto_curve": [],
        }

    def generate_tradeoff_curve(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitivity: float = DEFAULT_SENSITIVITY,
        mechanism: str = MECHANISM_LAPLACE_CLIPPED,
        delta: float = 0.00001,
        classifier_type: str = "logistic_regression",
        epsilon_grid: Optional[List[float]] = None,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Computes real utility metrics across epsilon values.
        NO formula-based MIA curves — MIA values come from actual shadow-model attacks.

        Note: Running full MIA for each epsilon point is expensive.
        For the tradeoff curve, we run a lightweight 2-shadow MIA per point.
        For rigorous results, use experiment_runner.py.
        """
        if epsilon_grid is None:
            epsilon_grid = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

        m_norm = self._normalize_mechanism(mechanism)
        y_arr = y.values.astype(int)
        X_values = X.values
        curve = []

        try:
            # For lightweight curve, just split 80/20 train/test to save time,
            # but we need shadow data as well.
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X_values, y_arr, test_size=0.20, random_state=seed, stratify=y_arr
            )
            # Use test as shadow pool for curve simplicity
            X_shadow_raw = X_test_raw
            y_shadow = y_test
        except ValueError:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X_values, y_arr, test_size=0.20, random_state=seed
            )
            X_shadow_raw = X_test_raw
            y_shadow = y_test

        # Fit scaler on train only
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        X_shadow_scaled = scaler.transform(X_shadow_raw)

        feature_count = X_train_scaled.shape[1]

        for eps in epsilon_grid:
            try:
                if m_norm == MECHANISM_NONE:
                    X_dp = X_train_scaled.copy()
                    eps_feature = float("inf")
                else:
                    eps_accounting = self.compute_epsilon_accounting(eps, feature_count)
                    eps_feature = eps_accounting["epsilon_feature"]
                    X_dp = self.inject_dp_noise(
                        X_train_scaled,
                        epsilon_feature=eps_feature,
                        sensitivity=sensitivity,
                        mechanism=m_norm,
                        delta=delta,
                        seed=seed,
                    )

                clf = self._create_classifier(classifier_type, seed)
                clf.fit(X_dp, y_train)
                metrics = self.evaluate_model_metrics(clf, X_test_scaled, y_test)

                # Lightweight MIA (2 shadow models for curve — not final results)
                if m_norm != MECHANISM_NONE:
                    dp_fn = self.make_dp_perturb_fn(eps_feature, sensitivity, m_norm, delta, seed)
                else:
                    dp_fn = None

                mia_curve = run_shadow_mia(
                    target_model=clf,
                    X_train_target=X_train_scaled,
                    y_train_target=y_train,
                    X_test_target=X_test_scaled,
                    y_test_target=y_test,
                    X_shadow=X_shadow_scaled,
                    y_shadow=y_shadow,
                    classifier_type=classifier_type,
                    n_shadow=2,  # lightweight for curve
                    seed=seed,
                    dp_perturb_fn=dp_fn,
                )

                curve.append({
                    "epsilon": float(eps),
                    "epsilon_feature": round(eps_feature, 6) if eps_feature != float("inf") else None,
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "rocAuc": metrics.get("rocAuc"),
                    "prAuc": metrics.get("prAuc"),
                    "fraud_f1": metrics.get("fraud_f1"),
                    "fraud_recall": metrics.get("fraud_recall"),
                    # MIA from actual shadow-model attack (no formulas)
                    "miaAccuracy": mia_curve.get("attack_accuracy"),
                    "miaAuc": mia_curve.get("attack_auc"),
                    "attackerAdvantage": mia_curve.get("attacker_advantage"),
                    "tprAt1Fpr": mia_curve.get("tpr_at_1fpr"),
                    # Legacy field name
                    "miaRisk": mia_curve.get("attack_accuracy"),
                    "reconRisk": None,  # REMOVED
                })
            except Exception as e:
                print(f"[PrivacyEngine] Tradeoff curve error at eps={eps}: {e}")
                # Do NOT add formula fallback — skip this point instead
                continue

        return curve

    # ─── Helper: Normalize Mechanism Name ─────────────────────────────────
    def _normalize_mechanism(self, mechanism: str) -> str:
        """Normalizes mechanism string to canonical internal name."""
        m = mechanism.lower().replace(" ", "_").replace("-", "_")
        if m in ("laplace_clipped", "truncated_laplace", "truncated", "laplace+clip"):
            return MECHANISM_LAPLACE_CLIPPED
        elif m in ("plain_laplace", "laplace", "unbounded_laplace"):
            return MECHANISM_LAPLACE
        elif m in ("gaussian", "gauss"):
            return MECHANISM_GAUSSIAN
        elif m in ("non_private", "none", "no_dp"):
            return MECHANISM_NONE
        # Default: laplace_clipped
        return MECHANISM_LAPLACE_CLIPPED
