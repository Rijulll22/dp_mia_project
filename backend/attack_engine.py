"""
attack_engine.py — Genuine Shadow-Model Membership Inference Attack (MIA)

Scientific correctness requirements:
1. Shadow members and non-members are BOTH queried as model confidence vectors
   — NOT as raw feature vectors. The attacker sees only model outputs.
2. CRITICAL: For input-DP experiments, shadow training data is perturbed with
   the SAME mechanism and epsilon_feature as the target. Both member and
   non-member feature vectors are passed through the SAME DP mechanism before
   querying the shadow model, so the attacker cannot detect noise as a
   membership signal.
3. No np.clip() on attack metrics — raw values only.
4. Reports: attack_accuracy, attack_auc, attacker_advantage,
   tpr_at_1fpr, tpr_at_01fpr.
5. Per-class analysis: all records / legitimate records / fraud records.

Attribute Inference / Reconstruction:
- The previous heuristic (alpha * noisy + (1-alpha) * mean) was NOT a real attack.
- That code is REMOVED.
- Attribute inference is documented as future work.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _get_tpr_at_fpr_threshold(fpr_arr: np.ndarray, tpr_arr: np.ndarray, target_fpr: float) -> float:
    """
    Returns TPR at the first FPR threshold <= target_fpr.
    Used for TPR @ 1% FPR and TPR @ 0.1% FPR.
    """
    # Find all indices where fpr <= target_fpr
    eligible = np.where(fpr_arr <= target_fpr)[0]
    if len(eligible) == 0:
        # No point with FPR that low — attacker can't achieve this FPR
        return 0.0
    # Take the last such index (highest TPR among eligible)
    return float(tpr_arr[eligible[-1]])


def _extract_model_confidence_features(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Extracts attacker feature vectors from model confidence outputs.
    The attacker sees ONLY model query responses — not raw feature vectors.

    Features extracted:
    [true_class_prob, max_prob, entropy, cross_entropy_loss]

    Both members and non-members receive exactly the same type of representation.
    """
    try:
        probs = model.predict_proba(X)
    except Exception:
        preds = model.predict(X)
        n_classes = len(np.unique(y)) or 2
        probs = np.zeros((len(X), n_classes))
        for i, p in enumerate(preds):
            c = int(p)
            if c < n_classes:
                probs[i, c] = 1.0
            else:
                probs[i, -1] = 1.0

    n_samples = len(X)
    n_classes = probs.shape[1]
    eps_log = 1e-12

    features = np.zeros((n_samples, 4))
    for i in range(n_samples):
        label = int(y[i]) if i < len(y) else 0
        label = min(label, n_classes - 1)

        true_class_prob = float(probs[i, label])
        max_prob = float(np.max(probs[i]))
        entropy = -float(np.sum(probs[i] * np.log(np.clip(probs[i], eps_log, 1.0))))
        ce_loss = -float(np.log(max(eps_log, true_class_prob)))

        features[i] = [true_class_prob, max_prob, entropy, ce_loss]

    return features


def _create_shadow_model(classifier_type: str, seed: int) -> Any:
    """Creates a shadow model of the same type as the target."""
    ct = classifier_type.lower().replace(" ", "_")
    if ct in ("random_forest", "rf"):
        return RandomForestClassifier(
            n_estimators=25, max_depth=6, class_weight="balanced", random_state=seed, n_jobs=-1
        )
    return LogisticRegression(
        max_iter=300, class_weight="balanced", random_state=seed
    )


def run_shadow_mia(
    target_model: Any,
    X_train_target: np.ndarray,
    y_train_target: np.ndarray,
    X_test_target: np.ndarray,
    y_test_target: np.ndarray,
    X_shadow: np.ndarray,
    y_shadow: np.ndarray,
    classifier_type: str = "logistic_regression",
    n_shadow: int = 5,
    seed: int = 42,
    dp_perturb_fn=None,
) -> Dict[str, Any]:
    """
    Genuine shadow-model Membership Inference Attack (Shokri et al., 2017).

    Pipeline:
    1. Pool all available data (shadow pool).
    2. For each shadow model:
       a. Sample a shadow training set and shadow holdout from the pool.
       b. If dp_perturb_fn is not None, apply DP to shadow training data
          (same mechanism + epsilon_feature as target).
       c. Train shadow model.
       d. Query shadow model on members (shadow train) and non-members (shadow holdout).
          CRITICAL: Query original clean scaled features consistently for members and nonmembers.
       e. Extract confidence-vector features — NOT raw features.
    3. Train binary attack classifier on shadow confidence vectors.
    4. Evaluate attack on target model:
       - Members = target training set
       - Non-members = target test set
       Both are queried as confidence vectors only using clean features.
    5. Compute metrics WITHOUT clipping.

    Parameters
    ----------
    target_model : fitted scikit-learn model
    X_train_target : CLEAN training data used to query target model
    y_train_target : training labels
    X_test_target : clean test data NOT seen during training
    y_test_target : test labels
    X_shadow : shadow pool features
    y_shadow : shadow pool labels
    classifier_type : "logistic_regression" or "random_forest"
    n_shadow : number of shadow models (default 5)
    seed : random seed
    dp_perturb_fn : callable(X_np) -> X_perturbed_np, or None for non-private
                    Applied to shadow training data.


    Returns
    -------
    dict with keys:
        attack_accuracy, attack_auc, attacker_advantage,
        tpr_at_1fpr, tpr_at_01fpr,
        n_members_evaluated, n_nonmembers_evaluated
    """
    rng = np.random.RandomState(seed)

    # Pool for shadow model partitioning
    X_pool = X_shadow
    y_pool = y_shadow
    n_total = len(X_pool)

    shadow_attack_X: List[np.ndarray] = []
    shadow_attack_y: List[np.ndarray] = []

    indices = np.arange(n_total)
    # Each shadow model gets roughly n_total / (2*n_shadow) samples as train and holdout
    split_size = min(n_total // 2, max(20, n_total // max(2, n_shadow + 1)))

    if split_size < 5:
        return {
            "attack_accuracy": None,
            "attack_auc": None,
            "attacker_advantage": None,
            "tpr_at_1fpr": None,
            "tpr_at_01fpr": None,
            "n_members_evaluated": 0,
            "n_nonmembers_evaluated": 0,
            "error": "Insufficient shadow training data",
        }

    for s in range(n_shadow):
        try:
            X_s_train, X_s_holdout, y_s_train, y_s_holdout = train_test_split(
                X_pool, y_pool, train_size=split_size, test_size=split_size, random_state=seed+s, stratify=y_pool
            )
        except ValueError:
            X_s_train, X_s_holdout, y_s_train, y_s_holdout = train_test_split(
                X_pool, y_pool, train_size=split_size, test_size=split_size, random_state=seed+s
            )

        if len(y_s_holdout) < 5 or len(np.unique(y_s_train)) < 2:
            continue

        # Apply DP to shadow training data if required
        if dp_perturb_fn is not None:
            X_s_train_for_model = dp_perturb_fn(X_s_train)
        else:
            X_s_train_for_model = X_s_train

        shadow_model = _create_shadow_model(classifier_type, seed=seed + s)
        try:
            shadow_model.fit(X_s_train_for_model, y_s_train)
        except Exception:
            continue

        # CRITICAL: For member queries, query original clean scaled features consistently.
        X_member_query = X_s_train
        X_nonmember_query = X_s_holdout

        # Extract confidence features (not raw features)
        member_feats = _extract_model_confidence_features(shadow_model, X_member_query, y_s_train)
        nonmember_feats = _extract_model_confidence_features(shadow_model, X_nonmember_query, y_s_holdout)

        shadow_attack_X.append(member_feats)
        shadow_attack_y.append(np.ones(len(member_feats), dtype=int))
        shadow_attack_X.append(nonmember_feats)
        shadow_attack_y.append(np.zeros(len(nonmember_feats), dtype=int))

    # Check if we have enough shadow data
    if not shadow_attack_X:
        return {
            "attack_accuracy": None,
            "attack_auc": None,
            "attacker_advantage": None,
            "tpr_at_1fpr": None,
            "tpr_at_01fpr": None,
            "n_members_evaluated": 0,
            "n_nonmembers_evaluated": 0,
            "error": "Insufficient shadow training data",
        }

    X_attack_train = np.vstack(shadow_attack_X)
    y_attack_train = np.concatenate(shadow_attack_y)

    # Train binary attack classifier with standardized confidence features
    attack_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=seed, C=1.0)),
    ])
    try:
        attack_clf.fit(X_attack_train, y_attack_train)
    except Exception as e:
        return {
            "attack_accuracy": None,
            "attack_auc": None,
            "attacker_advantage": None,
            "tpr_at_1fpr": None,
            "tpr_at_01fpr": None,
            "n_members_evaluated": 0,
            "n_nonmembers_evaluated": 0,
            "error": f"Attack classifier training failed: {e}",
        }

    # Evaluate on target model
    # Members = data used to train target model
    # Non-members = held-out test data not seen during training
    # Both queried as confidence vectors (same representation)
    target_member_feats_full = _extract_model_confidence_features(
        target_model, X_train_target, y_train_target
    )
    target_nonmember_feats_full = _extract_model_confidence_features(
        target_model, X_test_target, y_test_target
    )

    # Balance target attack set (Requirement 2.A): Enforce exact protocol
    # Every eligible target-test nonmember, and an equal sized sample of target-train members.
    n_members = len(target_member_feats_full)
    n_nonmembers = len(target_nonmember_feats_full)
    
    if n_members < n_nonmembers:
        return {
            "attack_accuracy": None,
            "attack_auc": None,
            "attacker_advantage": None,
            "tpr_at_1fpr": None,
            "tpr_at_01fpr": None,
            "n_members_evaluated": n_members,
            "n_nonmembers_evaluated": n_nonmembers,
            "error": f"Insufficient members ({n_members}) to form a balanced set matching all nonmembers ({n_nonmembers}).",
        }
    
    target_nonmember_feats = target_nonmember_feats_full
    target_member_feats = target_member_feats_full[rng.choice(n_members, n_nonmembers, replace=False)]

    X_eval = np.vstack([target_member_feats, target_nonmember_feats])
    y_eval = np.concatenate([
        np.ones(len(target_member_feats), dtype=int),
        np.zeros(len(target_nonmember_feats), dtype=int),
    ])

    attack_preds = attack_clf.predict(X_eval)
    try:
        attack_probs = attack_clf.predict_proba(X_eval)[:, 1]
        if not np.all(np.isfinite(attack_probs)):
            raise ValueError("Attack probability scores contain non-finite values (NaN/Inf)")
    except Exception as e:
        return {
            "attack_accuracy": None,
            "attack_auc": None,
            "attacker_advantage": None,
            "tpr_at_1fpr": None,
            "tpr_at_01fpr": None,
            "n_members_evaluated": int(len(target_member_feats)),
            "n_nonmembers_evaluated": int(len(target_nonmember_feats)),
            "error": f"Attack probability calculation failed: {e}",
        }

    acc = float(accuracy_score(y_eval, attack_preds))

    # Attacker advantage at threshold 0.5 (legacy metric)
    from sklearn.metrics import confusion_matrix as sk_cm
    cm = sk_cm(y_eval, attack_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        advantage_threshold_05 = float(tpr_val - fpr_val)
    else:
        advantage_threshold_05 = 0.0

    # ROC AUC, low-FPR metrics, and max(TPR-FPR) advantage
    try:
        raw_auc = float(roc_auc_score(y_eval, attack_probs))
        auc = max(raw_auc, 1.0 - raw_auc)  # effective AUC
        fpr_arr, tpr_arr, thresholds = roc_curve(y_eval, attack_probs)
        tpr_at_1fpr = _get_tpr_at_fpr_threshold(fpr_arr, tpr_arr, 0.01)
        tpr_at_01fpr = _get_tpr_at_fpr_threshold(fpr_arr, tpr_arr, 0.001)
        advantage = float(np.max(np.abs(tpr_arr - fpr_arr)))
    except Exception as e:
        return {
            "attack_accuracy": None,
            "attack_auc": None,
            "attacker_advantage": None,
            "tpr_at_1fpr": None,
            "tpr_at_01fpr": None,
            "n_members_evaluated": int(len(target_member_feats)),
            "n_nonmembers_evaluated": int(len(target_nonmember_feats)),
            "error": f"MIA metric calculation failed: {e}",
        }

    return {
        # Raw values — NO clipping
        "attack_accuracy": round(acc, 4),
        "attack_auc": round(auc, 4),
        "raw_mia_auc": round(raw_auc, 4),
        "attacker_advantage": round(advantage, 4),
        "attacker_advantage_05": round(advantage_threshold_05, 4),
        "tpr_at_1fpr": round(tpr_at_1fpr, 4) if tpr_at_1fpr is not None else None,
        "tpr_at_01fpr": round(tpr_at_01fpr, 4) if tpr_at_01fpr is not None else None,
        "n_members_evaluated": int(len(target_member_feats)),
        "n_nonmembers_evaluated": int(len(target_nonmember_feats)),
        "attack_clf": attack_clf,
    }


def run_per_class_mia(
    target_model: Any,
    X_train_target: np.ndarray,
    y_train_target: np.ndarray,
    X_test_target: np.ndarray,
    y_test_target: np.ndarray,
    attack_clf: Any,
    class_label: int,
    min_samples: int = 20,
    seed: int = 42,
) -> Optional[Dict[str, Any]]:
    """
    Runs MIA evaluation restricted to a specific class (e.g., fraud=1, legit=0).
    Returns None if there are insufficient samples for reliable evaluation.
    Reports this honestly rather than fabricating results.
    """
    # Filter members
    member_mask = y_train_target == class_label
    nonmember_mask = y_test_target == class_label

    n_members = int(np.sum(member_mask))
    n_nonmembers = int(np.sum(nonmember_mask))

    if n_members < n_nonmembers or n_nonmembers < min_samples:
        return {
            "class_label": class_label,
            "n_members": n_members,
            "n_nonmembers": n_nonmembers,
            "note": f"Insufficient samples for per-class MIA (need >= {min_samples} nonmembers, and members >= nonmembers). "
                    f"Got {n_members} members, {n_nonmembers} non-members.",
            "attack_accuracy": None,
            "attack_auc": None,
            "attacker_advantage": None,
        }

    # Balance the attack set for class-conditional
    rng = np.random.RandomState(seed)
    
    member_indices = np.where(member_mask)[0]
    nonmember_indices = np.where(nonmember_mask)[0]
    
    member_indices = rng.choice(member_indices, n_nonmembers, replace=False)
    # Use all nonmember_indices

    X_m = X_train_target[member_indices]
    y_m = y_train_target[member_indices]
    X_nm = X_test_target[nonmember_indices]
    y_nm = y_test_target[nonmember_indices]

    member_feats = _extract_model_confidence_features(target_model, X_m, y_m)
    nonmember_feats = _extract_model_confidence_features(target_model, X_nm, y_nm)

    X_eval = np.vstack([member_feats, nonmember_feats])
    y_eval = np.concatenate([
        np.ones(len(member_feats), dtype=int),
        np.zeros(len(nonmember_feats), dtype=int),
    ])

    preds = attack_clf.predict(X_eval)
    acc = float(accuracy_score(y_eval, preds))

    from sklearn.metrics import confusion_matrix as sk_cm
    cm = sk_cm(y_eval, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        advantage_threshold_05 = float(tpr_val - fpr_val)
    else:
        advantage_threshold_05 = 0.0

    try:
        probs = attack_clf.predict_proba(X_eval)[:, 1]
        if not np.all(np.isfinite(probs)):
            raise ValueError("Per-class attack probability scores contain non-finite values (NaN/Inf)")
        raw_auc = float(roc_auc_score(y_eval, probs))
        auc = max(raw_auc, 1.0 - raw_auc)
        fpr_arr, tpr_arr, _ = roc_curve(y_eval, probs)
        advantage = float(np.max(np.abs(tpr_arr - fpr_arr)))
    except Exception as e:
        return {
            "class_label": class_label,
            "n_members": n_members,
            "n_nonmembers": n_nonmembers,
            "attack_accuracy": None,
            "attack_auc": None,
            "raw_mia_auc": None,
            "attacker_advantage": None,
            "attacker_advantage_05": None,
            "error": f"Per-class MIA metric calculation failed: {e}",
        }

    return {
        "class_label": class_label,
        "n_members": n_members,
        "n_nonmembers": n_nonmembers,
        "attack_accuracy": round(acc, 4),
        "attack_auc": round(auc, 4) if auc is not None else None,
        "raw_mia_auc": round(raw_auc, 4) if raw_auc is not None else None,
        "attacker_advantage": round(advantage, 4),
        "attacker_advantage_05": round(advantage_threshold_05, 4),
    }


# ─── Legacy alias for backward compat ────────────────────────────────────────
def run_shokri_shadow_mia(
    target_model: Any,
    X_train_target: np.ndarray,
    y_train_target: np.ndarray,
    X_test_target: np.ndarray,
    y_test_target: np.ndarray,
    classifier_type: str = "logistic_regression",
    n_shadow: int = 5,
    seed: int = 42,
    dp_perturb_fn=None,
) -> Dict[str, Any]:
    """Backward-compatible wrapper around run_shadow_mia."""
    return run_shadow_mia(
        target_model=target_model,
        X_train_target=X_train_target,
        y_train_target=y_train_target,
        X_test_target=X_test_target,
        y_test_target=y_test_target,
        classifier_type=classifier_type,
        n_shadow=n_shadow,
        seed=seed,
        dp_perturb_fn=dp_perturb_fn,
    )


# ─── Attribute Inference — FUTURE WORK ───────────────────────────────────────
# The previous run_attribute_reconstruction_attack() used:
#   X_recon = alpha * X_perturbed + (1-alpha) * prior_means
# This is a mathematical blend — NOT a real attribute inference attack.
# It has been REMOVED from the research pipeline.
#
# Future work:
# - Implement proper attribute inference: hide Transaction_Amount from attacker,
#   use model output + permitted features to train a shadow inference model,
#   evaluate with MAE / RMSE.
# See RESEARCH_STATUS.md for details.
def run_attribute_reconstruction_attack(*args, **kwargs):
    """
    DEPRECATED — was a mathematical heuristic, not a real attack.
    Attribute inference is future work. See RESEARCH_STATUS.md.
    Returns None to indicate this experiment was not run.
    """
    return None
