import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
import numpy as np
import pandas as pd
from backend.privacy_engine import PrivacyEngine, MECHANISM_LAPLACE_CLIPPED

def main():
    print("=" * 70)
    print("PRIVACYGUARD ISOLATED REAL-DATA PILOT AUDIT")
    print("Configuration: ULB, Logistic Regression, laplace_clipped, eps=1.0, seed=42, n_shadow=5")
    print("=" * 70)

    ulb_path = "backend/data/creditcard.csv"
    if not os.path.exists(ulb_path):
        print(f"Error: {ulb_path} not found")
        sys.exit(1)

    df = pd.read_csv(ulb_path)
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    target_col = "Class"
    y = df[target_col]
    X = df.drop(columns=[target_col])

    print(f"Dataset: shape={X.shape}, target={target_col}")
    print(f"Features ({len(X.columns)}): {list(X.columns)}")
    print(f"Class distribution: 0={np.sum(y == 0)}, 1={np.sum(y == 1)}")

    engine = PrivacyEngine(seed=42)

    epsilon_record = 1.0
    epsilon_feature = epsilon_record / X.shape[1]
    sensitivity = 6.0
    scale = engine.calculate_noise_scale(epsilon_feature, sensitivity, MECHANISM_LAPLACE_CLIPPED)

    print(f"\n--- Accounting & Calibration Parameters ---")
    print(f"Record Epsilon:    {epsilon_record}")
    print(f"Feature Epsilon:   {epsilon_feature:.8f} (1/{X.shape[1]})")
    print(f"Sensitivity:       {sensitivity}")
    print(f"Laplace Scale:     {scale:.4f}")
    print(f"Clipping Bound C:  3.0 (domain [-3, 3])")

    print("\n--- Executing Isolated Pipeline (No DB writes) ---")
    result = engine.run_pipeline(
        X, y,
        epsilon=epsilon_record,
        sensitivity=sensitivity,
        mechanism=MECHANISM_LAPLACE_CLIPPED,
        classifier_type="logistic_regression",
        seed=42,
        n_shadow=5
    )

    print("\n--- Partition Details ---")
    print(f"Train partition:  total={result.get('train_size')}, frauds={result.get('fraud_count_train')}")
    print(f"Test partition:   total={result.get('test_size')}, frauds={result.get('fraud_count_test')}")
    print(f"Validation-tuned threshold: {result.get('best_threshold'):.4f}")

    print("\n--- Target Model Utility Metrics ---")
    m = result.get('metrics', {})
    print(f"ROC-AUC (continuous scores): {m.get('rocAuc')}")
    print(f"PR-AUC  (continuous scores): {m.get('prAuc')}")
    print(f"Accuracy:                    {m.get('accuracy')}")
    print(f"Precision (overall):         {m.get('precision')}")
    print(f"Recall (overall):            {m.get('recall')}")
    print(f"F1 (overall):                {m.get('f1')}")
    print(f"Fraud Precision (class 1):   {m.get('fraud_precision')}")
    print(f"Fraud Recall (class 1):      {m.get('fraud_recall')}")
    print(f"Fraud F1 (class 1):          {m.get('fraud_f1')}")
    print(f"Confusion Matrix:            {m.get('confusionMatrix')}")

    print("\n--- Non-Private Baseline Metrics ---")
    base = result.get('baselines', {}).get('non_private', {})
    print(f"Non-private Threshold:       {base.get('best_threshold'):.4f}")
    print(f"Non-private ROC-AUC:         {base.get('rocAuc')}")
    print(f"Non-private PR-AUC:          {base.get('prAuc')}")
    print(f"Non-private F1:              {base.get('f1')}")
    print(f"Non-private Fraud F1:        {base.get('fraud_f1')}")
    print(f"Relative Utility Loss (%):   {result.get('relative_utility_loss')}%")

    print("\n--- MIA Attack Results ---")
    mia = result.get('mia_result', {})
    if 'error' in mia:
        print(f"MIA Status: FAILED")
        print(f"Error: {mia['error']}")
    else:
        print(f"MIA Status: SUCCESS (Attack classifier fitted)")
        print(f"Evaluated Attack Set: Members={mia.get('n_members_evaluated')}, Nonmembers={mia.get('n_nonmembers_evaluated')}")
        print(f"Raw MIA AUC:                 {mia.get('raw_mia_auc')}")
        print(f"Effective MIA AUC:           {mia.get('attack_auc')}")
        print(f"MIA Accuracy:                {mia.get('attack_accuracy')}")
        print(f"Attacker Advantage (max|TPR-FPR|): {mia.get('attacker_advantage')}")
        print(f"Attacker Advantage @ th=0.5:       {mia.get('attacker_advantage_05')}")
        print(f"TPR @ 1% FPR:                {mia.get('tpr_at_1fpr')}")
        print(f"TPR @ 0.1% FPR:              {mia.get('tpr_at_01fpr')}")

    print("\n--- Per-Class MIA Results ---")
    per_class = result.get('mia_per_class', {})
    print(f"Fraud MIA:       {per_class.get('fraud')}")
    print(f"Legitimate MIA:  {per_class.get('legitimate')}")
    print("=" * 70)

if __name__ == '__main__':
    main()
