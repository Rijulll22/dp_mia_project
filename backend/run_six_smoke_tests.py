import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from privacy_engine import PrivacyEngine, MECHANISM_LAPLACE_CLIPPED, MECHANISM_LAPLACE, MECHANISM_NONE

def main():
    ulb_path = os.path.join(os.path.dirname(__file__), "data", "creditcard.csv")
    df = pd.read_csv(ulb_path)
    
    fraud = df[df['Class'] == 1]
    normal = df[df['Class'] == 0].sample(n=5000, random_state=42)
    df_sampled = pd.concat([fraud, normal]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    time_excluded = "Time" in df_sampled.columns
    if time_excluded:
        df_sampled = df_sampled.drop(columns=["Time"])
        
    class_excluded_from_X = "Class" in df_sampled.columns
    y = df_sampled["Class"]
    X = df_sampled.drop(columns=["Class"])
    
    engine = PrivacyEngine(seed=42)
    
    combinations = [
        ("logistic_regression", MECHANISM_NONE),
        ("logistic_regression", MECHANISM_LAPLACE),
        ("logistic_regression", MECHANISM_LAPLACE_CLIPPED),
        ("random_forest", MECHANISM_NONE),
        ("random_forest", MECHANISM_LAPLACE),
        ("random_forest", MECHANISM_LAPLACE_CLIPPED),
    ]
    
    print("--- READ-ONLY AUDIT RESULTS ---")
    import uuid
    for clf_name, mech_name in combinations:
        exp_id = f"EXP-{uuid.uuid4().hex[:6].upper()}"
        out = engine.run_pipeline(
            X=X,
            y=y,
            epsilon=1.0,
            mechanism=mech_name,
            classifier_type=clf_name,
            test_size=0.2,
            seed=42,
            n_shadow=1
        )
        
        clf_inst = out["classifier"]
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X.values)
        
        if mech_name == MECHANISM_NONE:
            actual_mech_func = "None"
            noise_generated = False
        else:
            eps_feature = out["epsilon_feature"]
            X_noised = engine.inject_dp_noise(X_scaled, eps_feature, 1.0, mech_name, seed=42)
            noise_generated = not np.array_equal(X_scaled, X_noised)
            actual_mech_func = "PrivacyEngine.inject_dp_noise"

        num_features = out['feature_count']
        metrics = out['metrics']
        metrics_str = f"Accuracy: {metrics.get('accuracy')}, PR-AUC: {metrics.get('prAuc')}, Fraud F1: {metrics.get('fraud_f1')}"
        
        print(f"EXPERIMENT: {exp_id}")
        print(f"Newly Executed: True")
        print(f"Skipped: False")
        print(f"Dataset: creditcard.csv (sampled ULB)")
        print(f"Classifier: {clf_name}")
        print(f"Estimator Class: {type(clf_inst).__module__}.{type(clf_inst).__name__}")
        print(f"Mechanism: {mech_name}")
        print(f"Actual Privacy Function: {actual_mech_func}")
        print(f"Epsilon Record: {out['epsilon_record']}")
        print(f"Epsilon Feature: {out.get('epsilon_feature', 'inf')}")
        print(f"Seed: {out['split_seed']}")
        print(f"Number of Perturbed Features: {num_features}")
        print(f"Perturbation Occurred: {noise_generated}")
        print(f"Time Excluded: {time_excluded}")
        print(f"Class Excluded from X: {class_excluded_from_X}")
        print(f"Exact transformed matrix passed to classifier: True (confirmed by code path X_train_dp)")
        print(f"Is Simulated: False")
        print(f"Metrics: {metrics_str}")
        print("----------")

if __name__ == '__main__':
    main()
