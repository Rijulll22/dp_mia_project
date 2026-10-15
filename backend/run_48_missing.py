import os
import sys
import uuid
import time
from datetime import datetime, timezone
import traceback
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from privacy_engine import PrivacyEngine, MECHANISM_LAPLACE_CLIPPED, MECHANISM_LAPLACE, MECHANISM_NONE
from database import init_db, SessionLocal, Experiment
from experiment_runner import save_to_db

FIX_TIMESTAMP = "2026-09-21T18:30:00+00:00"

def get_verified_db_configs():
    with SessionLocal() as db:
        exps = db.query(Experiment).filter(
            Experiment.is_simulated == False,
            Experiment.timestamp > FIX_TIMESTAMP
        ).all()
        
    configs = set()
    for exp in exps:
        # Ignore files that aren't ULB
        if exp.filename not in ['ulb', 'creditcard.csv']:
            continue
            
        clf = exp.classifier_type
        mech = exp.mechanism
        try:
            seed = int(exp.split_seed)
        except:
            continue
            
        if mech == MECHANISM_NONE:
            configs.add((clf, mech, None, seed))
        else:
            try:
                eps = float(exp.epsilon)
                configs.add((clf, mech, eps, seed))
            except:
                pass
    return configs

def main():
    init_db()
    
    # 1. Define target configs
    classifiers = ["logistic_regression", "random_forest"]
    mechanisms_dp = ["plain_laplace", "laplace_clipped"]
    epsilons = [0.1, 1.0, 5.0, 10.0]
    seeds = [42, 43, 44]
    
    all_configs = []
    for clf in classifiers:
        for seed in seeds:
            all_configs.append((clf, "non_private", None, seed))
            
    for clf in classifiers:
        for mech in mechanisms_dp:
            for eps in epsilons:
                for seed in seeds:
                    all_configs.append((clf, mech, eps, seed))
                    
    # 2. Hardcoded verified smoke tests (these were not saved to DB, so we must skip them manually)
    smoke_test_configs = {
        ("logistic_regression", "non_private", None, 42),
        ("random_forest", "non_private", None, 42),
        ("logistic_regression", "plain_laplace", 1.0, 42),
        ("logistic_regression", "laplace_clipped", 1.0, 42),
        ("random_forest", "plain_laplace", 1.0, 42),
        ("random_forest", "laplace_clipped", 1.0, 42),
    }
    
    # 3. DB verified configs
    db_verified = get_verified_db_configs()
    
    to_run = []
    for c in all_configs:
        if c in smoke_test_configs:
            continue
        if c in db_verified:
            continue
        to_run.append(c)
        
    print(f"Total configs to run: {len(to_run)}")
    if len(to_run) == 0:
        print("Nothing to run.")
        return

    # 4. Load ULB dataset once
    ulb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "creditcard.csv")
    if not os.path.exists(ulb_path):
        raise FileNotFoundError(f"ULB dataset not found at {ulb_path}")
    df = pd.read_csv(ulb_path)
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
        
    target_col = "Class"
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    engine = PrivacyEngine(seed=42)
    
    failures = []
    count = 0
    total = len(to_run)
    
    for clf, mech, eps, seed in to_run:
        count += 1
        eps_val = float('inf') if mech == "non_private" else eps
        eps_for_db = 1.0 if mech == "non_private" else eps  # just a placeholder for DB schema if needed
        
        try:
            print(f"\\n[{count}/{total}] Running: {clf} | {mech} | eps={eps} | seed={seed}")
            
            # Use same shadow logic as runner
            out = engine.run_pipeline(
                X=X,
                y=y,
                epsilon=eps_val,
                delta=0.00001,
                sensitivity=1.0,
                mechanism=mech,
                classifier_type=clf,
                test_size=0.20,
                seed=seed,
                n_shadow=5
            )
            
            # Verification required by prompt
            if mech != MECHANISM_NONE:
                feature_count = out.get("feature_count", X.shape[1])
                eps_acc = engine.compute_epsilon_accounting(eps, feature_count)
                expected_eps_feature = eps / 29.0
                if abs(eps_acc["epsilon_feature"] - expected_eps_feature) > 1e-5:
                    print(f"  [WARNING] epsilon_feature mismatch! Expected {expected_eps_feature}, got {eps_acc['epsilon_feature']}")
            else:
                eps_acc = {
                    "epsilon_record": float("inf"),
                    "epsilon_feature": float("inf"),
                    "feature_count": X.shape[1],
                    "composition_method": "non-private",
                }
            out["epsilon_accounting"] = eps_acc
            
            clf_inst = out["classifier"]
            clf_class = type(clf_inst).__name__
            if clf == "logistic_regression" and clf_class != "LogisticRegression":
                raise ValueError(f"Classifier verification failed: Expected LogisticRegression, got {clf_class}")
            if clf == "random_forest" and clf_class != "RandomForestClassifier":
                raise ValueError(f"Classifier verification failed: Expected RandomForestClassifier, got {clf_class}")
                
            num_features = out.get("feature_count")
            if num_features != 29:
                raise ValueError(f"Feature count verification failed: Expected 29, got {num_features}")
                
            exp_id = f"EXP-{uuid.uuid4().hex[:6].upper()}"
            
            # Save to DB
            save_to_db(
                experiment_id=exp_id,
                epsilon=eps_for_db,
                mechanism=mech,
                classifier_type=clf,
                seed=seed,
                pipeline_output=out,
                dataset_name="creditcard.csv",
                perturbed_path=None,
                model_path=None,
                export_path=None
            )
            
            print(f"[{count}/{total}]")
            print(f"experiment_id: {exp_id}")
            print(f"classifier: {clf}")
            print(f"mechanism: {mech}")
            print(f"epsilon: {eps}")
            print(f"seed: {seed}")
            print(f"COMPLETED")
            
        except Exception as e:
            traceback.print_exc()
            print(f"  [FAILED] {clf} | {mech} | eps={eps} | seed={seed}")
            failures.append({
                "config": f"{clf} | {mech} | eps={eps} | seed={seed}",
                "error": str(e)
            })

    print(f"\\n{'='*40}\\nEXECUTION FINISHED\\n{'='*40}")
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f" - {f['config']}: {f['error']}")
    else:
        print("ALL RUNS SUCCEEDED.")

if __name__ == "__main__":
    main()
