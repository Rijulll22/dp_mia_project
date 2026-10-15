import os

new_code = '''"""
experiment_runner.py — Reproducible Batch Experiment Runner

Results are stored in the SQLite database. Supports safe resume and live progress.
"""

import os
import sys
import json
import uuid
import argparse
import time
import hashlib
import traceback
import logging
from datetime import datetime, timezone
from typing import List, Optional
import numpy as np
import pandas as pd

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from privacy_engine import PrivacyEngine, MECHANISM_LAPLACE_CLIPPED, MECHANISM_LAPLACE, MECHANISM_NONE
from database import init_db, Experiment, SessionLocal

# ─── Configuration ────────────────────────────────────────────────────────

FAST_EPSILONS = [0.1, 1.0, 5.0, 10.0]
FULL_EPSILONS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

FAST_SEEDS = [42, 43]
FULL_SEEDS = list(range(42, 52))  # 42-51

ALL_MECHANISMS = [MECHANISM_LAPLACE_CLIPPED, MECHANISM_LAPLACE, MECHANISM_NONE]
ALL_CLASSIFIERS = ["logistic_regression", "random_forest"]

def get_config_key(dataset_name: str, classifier: str, mechanism: str, epsilon: float, seed: int, n_shadow: int, sensitivity: float) -> str:
    """Canonical deterministic configuration identity."""
    config = {
        'dataset': dataset_name,
        'classifier': classifier,
        'mechanism': mechanism,
        'epsilon': None if mechanism == MECHANISM_NONE else float(epsilon),
        'seed': int(seed),
        'n_shadow': int(n_shadow),
        'sensitivity': float(sensitivity)
    }
    canon = json.dumps(config, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()

def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"session_{session_id}.jsonl")
    
    logger = logging.getLogger("ExperimentRunner")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    
    # File handler
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    
    return logger, log_path

def log_event(logger: logging.Logger, event_type: str, data: dict):
    msg = json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "event": event_type, **data})
    logger.info(msg)

def format_progress(idx, total, clf, mech, eps, seed, status, attempt=""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    eps_str = "non_private" if mech == MECHANISM_NONE else str(eps)
    att_str = f" | attempt={attempt}" if attempt else ""
    return f"[{ts}] [{idx}/{total}] START | classifier={clf} | mechanism={mech} | epsilon={eps_str} | seed={seed} | status={status}{att_str}"


def update_db_record(exp_id: str, status: str, attempt: int, error_msg: str = None, out: dict = None,
                     eps: float = None, mech: str = None, clf: str = None, seed: int = None,
                     dataset_name: str = None, model_path: str = None, export_path: str = None, config_key: str = None):
    with SessionLocal() as db:
        exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
        if not exp:
            exp = Experiment(id=exp_id, config_key=config_key)
            db.add(exp)
            
        exp.status = status
        exp.attempt_count = attempt
        exp.error_message = error_msg
        exp.timestamp = datetime.now(timezone.utc).isoformat()
        
        if out:
            metrics = out.get("metrics", {})
            mia = out.get("mia_result") or {}
            eps_acc = out.get("epsilon_accounting") or {}
            
            exp.filename = dataset_name
            exp.is_simulated = False
            exp.split_seed = str(seed)
            exp.train_size = out.get("train_size")
            exp.test_size = out.get("test_size")
            exp.fraud_count_train = out.get("fraud_count_train")
            exp.fraud_count_test = out.get("fraud_count_test")
            exp.feature_list = json.dumps(out.get("feature_list", []))

            exp.epsilon = eps if eps is not None else float('inf')
            exp.delta = 0.00001
            exp.mechanism = mech
            exp.mechanism_type = mech
            exp.classifier_type = clf
            exp.noise_scale = out.get("noiseScale", 0.0)

            exp.epsilon_record = eps_acc.get("epsilon_record", eps)
            exp.epsilon_feature = eps_acc.get("epsilon_feature")
            exp.feature_count = eps_acc.get("feature_count")
            exp.composition_method = eps_acc.get("composition_method")

            exp.metrics = json.dumps(metrics)
            exp.pr_auc = metrics.get("prAuc")
            exp.fraud_precision = metrics.get("fraud_precision")
            exp.fraud_recall = metrics.get("fraud_recall")
            exp.fraud_f1 = metrics.get("fraud_f1")

            exp.mia_auc = mia.get("attack_auc")
            exp.mia_accuracy = mia.get("attack_accuracy")
            exp.attacker_advantage = mia.get("attacker_advantage")
            exp.attacker_advantage_05 = mia.get("attacker_advantage_05")
            exp.tpr_at_1fpr = mia.get("tpr_at_1fpr")
            exp.tpr_at_01fpr = mia.get("tpr_at_01fpr")
            exp.mia_per_class = json.dumps(out.get("mia_per_class", {}))
            
            exp.baselines = json.dumps(out.get("baselines", {}))
            exp.model_path = model_path
            exp.export_path = export_path
            exp.relative_utility_loss = out.get("relative_utility_loss")
            
        db.commit()


def run_experiments(
    epsilons: List[float],
    seeds: List[int],
    mechanisms: List[str],
    classifiers: List[str],
    n_shadow: int = 5,
    sensitivity: float = 6.0,
    dataset_name: str = "synthetic_benchmark",
    uploads_dir: Optional[str] = None,
    save_export: bool = False,
):
    """
    Runs all combinations of (epsilon, seed, mechanism, classifier).
    Supports safe resume and live progress.
    """
    if uploads_dir is None:
        uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    logger, log_path = setup_logger(log_dir)

    log_event(logger, "startup", {"dataset": dataset_name, "db_path": "backend/privacyguard.db"})

    engine = PrivacyEngine(seed=42)
    
    if dataset_name == "ulb":
        ulb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "creditcard.csv")
        if not os.path.exists(ulb_path):
            raise FileNotFoundError(f"ULB dataset not found at {ulb_path}")
        df = pd.read_csv(ulb_path)
        if "Time" in df.columns:
            df = df.drop(columns=["Time"])
        
        target_col = "Class"
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        X, y = engine.generate_benchmark_dataset(n_samples=5000)

    # Ensure binary labels
    y_unique = y.unique()
    if len(y_unique) > 2:
        top_class = y.mode()[0]
        y = (y != top_class).astype(int)

    run_configs = []
    for seed in seeds:
        for clf in classifiers:
            for mech in mechanisms:
                if mech == MECHANISM_NONE:
                    # Non-private runs do not duplicate across epsilons
                    run_configs.append((seed, clf, mech, float('inf')))
                else:
                    for eps in epsilons:
                        run_configs.append((seed, clf, mech, eps))
                        
    # Deduplicate non-private runs just in case
    run_configs = list(dict.fromkeys(run_configs))

    total = len(run_configs)
    
    stats = {
        "planned": total,
        "completed_this_session": 0,
        "skipped_existing": 0,
        "failed_this_session": 0,
        "retried": 0,
    }

    print(f"\\n{'='*60}")
    print(f"PrivacyGuard Reproducible Experiment Runner (Resumable)")
    print(f"{'='*60}")
    print(f"  Dataset   : {dataset_name} ({len(X)} samples)")
    print(f"  Total planned configurations: {total}")
    print(f"  Log file  : {log_path}")
    print(f"{'='*60}\\n")
    
    start_time = time.time()

    for idx, (seed, clf, mech, eps) in enumerate(run_configs, 1):
        config_key = get_config_key(dataset_name, clf, mech, eps, seed, n_shadow, sensitivity)
        
        # Check database for existing record
        with SessionLocal() as db:
            existing = db.query(Experiment).filter(
                (Experiment.config_key == config_key) | 
                ((Experiment.classifier_type == clf) & (Experiment.mechanism == mech) & 
                 (Experiment.split_seed == str(seed)) & 
                 ((Experiment.epsilon == eps) | ((Experiment.epsilon == None) & (mech == MECHANISM_NONE))))
            ).order_by(Experiment.timestamp.desc()).first()
            
            if existing and existing.status == "completed":
                print(format_progress(idx, total, clf, mech, eps, seed, "skipped"))
                log_event(logger, "skip", {"config_key": config_key, "reason": "already_completed"})
                stats["skipped_existing"] += 1
                continue
                
            attempt = 1
            if existing:
                if existing.attempt_count:
                    attempt = existing.attempt_count + 1
                exp_id = existing.id
                stats["retried"] += 1
            else:
                exp_id = f"EXP-{uuid.uuid4().hex[:6].upper()}"
        
        print(format_progress(idx, total, clf, mech, eps, seed, "running", attempt))
        log_event(logger, "start", {"config_key": config_key, "exp_id": exp_id, "attempt": attempt, "config": {"clf": clf, "mech": mech, "eps": eps, "seed": seed}})
        
        # Mark as running
        update_db_record(exp_id, status="running", attempt=attempt, config_key=config_key)
        
        try:
            out = engine.run_pipeline(
                X=X,
                y=y,
                epsilon=eps,
                delta=0.00001,
                sensitivity=sensitivity,
                mechanism=mech,
                classifier_type=clf,
                test_size=0.20,
                seed=seed,
                n_shadow=n_shadow,
            )

            # Compute epsilon accounting
            if mech != MECHANISM_NONE:
                feature_count = out.get("feature_count", X.shape[1])
                eps_acc = engine.compute_epsilon_accounting(eps, feature_count)
            else:
                eps_acc = {
                    "epsilon_record": float("inf"),
                    "epsilon_feature": float("inf"),
                    "feature_count": X.shape[1],
                    "composition_method": "non-private",
                }
            out["epsilon_accounting"] = eps_acc

            model_path = os.path.join(uploads_dir, f"{exp_id}_model.json")
            if "modelArtifacts" in out:
                with open(model_path, "w") as f:
                    json.dump(out["modelArtifacts"], f, indent=2)

            export_path = None
            if save_export and "perturbedDf" in out:
                export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports", exp_id)
                os.makedirs(export_dir, exist_ok=True)
                df = out["perturbedDf"]
                np.save(os.path.join(export_dir, "noised_data.npy"), df.values)
                export_path = export_dir

            # Mark completed
            update_db_record(
                exp_id, status="completed", attempt=attempt, out=out,
                eps=eps, mech=mech, clf=clf, seed=seed,
                dataset_name=dataset_name, model_path=model_path, export_path=export_path, config_key=config_key
            )
            
            stats["completed_this_session"] += 1
            log_event(logger, "success", {"config_key": config_key, "exp_id": exp_id, "metrics": out.get("metrics")})
            
            metrics = out.get("metrics", {})
            mia = out.get("mia_result") or {}
            print(f"  └─ Success: F1={metrics.get('f1','?'):.4f} | ROC-AUC={metrics.get('rocAuc','?'):.4f} | MIA-AUC={mia.get('attack_auc','?')} | Adv={mia.get('attacker_advantage','?')}")
            
        except Exception as e:
            error_msg = str(e)
            trace = traceback.format_exc()
            update_db_record(exp_id, status="failed", attempt=attempt, error_msg=error_msg, config_key=config_key)
            stats["failed_this_session"] += 1
            log_event(logger, "error", {"config_key": config_key, "exp_id": exp_id, "error": error_msg, "traceback": trace})
            print(f"  └─ Failed: {error_msg}")

    # Ensure stale runs are recovered eventually (for now they are just retried normally on restart)
    
    total_elapsed = time.time() - start_time
    print(f"\\n{'='*60}")
    print(f"Batch complete in {total_elapsed:.0f}s.")
    print(f"Summary: Planned={stats['planned']} | Completed={stats['completed_this_session']} | Skipped={stats['skipped_existing']} | Failed={stats['failed_this_session']} | Retried={stats['retried']}")
    print(f"{'='*60}\\n")


# ─── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="PrivacyGuard Reproducible Experiment Runner"
    )
    parser.add_argument(
        "--mode", choices=["fast", "full"], default="fast",
        help="fast: 2 seeds × 4 epsilons; full: 10 seeds × 6 epsilons"
    )
    parser.add_argument(
        "--epsilons", type=float, nargs="+", default=None,
        help="Custom epsilon values (overrides mode default)"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Custom random seeds (overrides mode default)"
    )
    parser.add_argument(
        "--mechanisms", nargs="+",
        default=None,
        choices=["laplace_clipped", "plain_laplace", "non_private"],
        help="DP mechanisms to test"
    )
    parser.add_argument(
        "--classifiers", nargs="+",
        default=None,
        choices=["logistic_regression", "random_forest"],
        help="Classifiers to test"
    )
    parser.add_argument(
        "--n_shadow", type=int, default=5,
        help="Number of shadow models for MIA (default 5)"
    )
    parser.add_argument(
        "--dataset", type=str, default="ulb",
        help="Dataset to use: 'ulb' or 'synthetic_benchmark'"
    )
    parser.add_argument(
        "--save-export", action="store_true",
        help="Explicitly persist the DP-noised feature matrices for the generated experiments"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Initialize DB
    init_db()

    # Resolve mode defaults
    epsilons = args.epsilons or (FAST_EPSILONS if args.mode == "fast" else FULL_EPSILONS)
    seeds = args.seeds or (FAST_SEEDS if args.mode == "fast" else FULL_SEEDS)
    mechanisms = args.mechanisms or ALL_MECHANISMS
    classifiers = args.classifiers or ALL_CLASSIFIERS

    # Ensure non_private is handled correctly based on mechanism selection
    if "non_private" not in mechanisms and "non_private" in ALL_MECHANISMS:
        # If user explicitly specified mechanisms without non_private, respect it.
        pass

    run_experiments(
        epsilons=epsilons,
        seeds=seeds,
        mechanisms=mechanisms,
        classifiers=classifiers,
        n_shadow=args.n_shadow,
        dataset_name=args.dataset,
        save_export=args.save_export,
    )
'''

with open('backend/experiment_runner.py', 'w', encoding='utf-8') as f:
    f.write(new_code)
