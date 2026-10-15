import os
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from privacy_engine import PrivacyEngine


# ============================================================
# FINAL BATCH
# ============================================================

BATCH_ID = "FINAL_ULB_20261007"

DATASET_PATH = os.path.join(
    ROOT,
    "data",
    "creditcard.csv"
)

RESULT_DIR = os.path.join(
    ROOT,
    "final_results"
)

os.makedirs(RESULT_DIR, exist_ok=True)

RESULT_CSV = os.path.join(
    RESULT_DIR,
    f"{BATCH_ID}_results.csv"
)

SUMMARY_FILE = os.path.join(
    RESULT_DIR,
    f"{BATCH_ID}_summary.txt"
)


CLASSIFIERS = [
    "logistic_regression",
    "random_forest",
]

MECHANISMS = [
    "non_private",
    "plain_laplace",
    "laplace_clipped",
]

EPSILONS = [
    0.1,
    1.0,
    5.0,
    10.0,
]

SEEDS = [
    42,
    43,
    44,
]

DELTA = 0.00001
SENSITIVITY = 1.0
TEST_SIZE = 0.20
N_SHADOW = 5


# ============================================================
# BUILD EXACT 54 CONFIGURATIONS
# ============================================================

def build_configs():

    configs = []

    # --------------------------------------------------------
    # 6 non-private baseline configurations
    # --------------------------------------------------------

    for classifier in CLASSIFIERS:
        for seed in SEEDS:

            configs.append({
                "classifier": classifier,
                "mechanism": "non_private",
                "epsilon": float("inf"),
                "seed": seed,
            })

    # --------------------------------------------------------
    # 48 DP configurations
    # --------------------------------------------------------

    for classifier in CLASSIFIERS:
        for mechanism in [
            "plain_laplace",
            "laplace_clipped",
        ]:
            for epsilon in EPSILONS:
                for seed in SEEDS:

                    configs.append({
                        "classifier": classifier,
                        "mechanism": mechanism,
                        "epsilon": epsilon,
                        "seed": seed,
                    })

    return configs


# ============================================================
# LOAD REAL ULB DATASET
# ============================================================

def load_dataset():

    print("\n" + "=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"ULB dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print("Dataset:", DATASET_PATH)
    print("Shape:", df.shape)

    expected_columns = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount", "Class"]
    )

    if list(df.columns) != expected_columns:
        raise ValueError(
            "Unexpected dataset columns."
        )

    if len(df) != 284807:
        raise ValueError(
            f"Expected 284807 rows, got {len(df)}"
        )

    fraud_count = int(df["Class"].sum())
    legitimate_count = int(
        (df["Class"] == 0).sum()
    )

    print("Legitimate:", legitimate_count)
    print("Fraud:", fraud_count)
    print(
        "Fraud rate:",
        round(
            fraud_count / len(df) * 100,
            4
        ),
        "%"
    )

    if fraud_count != 492:
        raise ValueError(
            f"Expected 492 fraud records, got {fraud_count}"
        )

    # Time is excluded.
    # Class is target.
    X = df.drop(
        columns=["Time", "Class"]
    )

    y = df["Class"]

    if X.shape[1] != 29:
        raise ValueError(
            f"Expected 29 features, got {X.shape[1]}"
        )

    print("Model features:", X.shape[1])
    print("Excluded features: Time")
    print("Target: Class")

    return X, y


# ============================================================
# EXTRACT ACTUAL PIPELINE RESULTS
# ============================================================

def extract_result(
    output,
    config,
    elapsed
):

    metrics = output.get(
        "metrics",
        {}
    )

    mia = output.get(
        "mia_result",
        {}
    )

    epsilon = config["epsilon"]

    return {

        # ------------------------------
        # Batch identity
        # ------------------------------

        "batch_id": BATCH_ID,

        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        # ------------------------------
        # Configuration
        # ------------------------------

        "classifier": config[
            "classifier"
        ],

        "mechanism": config[
            "mechanism"
        ],

        "epsilon": epsilon,

        "epsilon_feature": (
            epsilon / 29
            if np.isfinite(epsilon)
            else float("inf")
        ),

        "delta": DELTA,

        "seed": config[
            "seed"
        ],

        # ------------------------------
        # Utility
        # ------------------------------

        "accuracy": metrics.get(
            "accuracy"
        ),

        "precision": metrics.get(
            "precision"
        ),

        "recall": metrics.get(
            "recall"
        ),

        "f1": metrics.get(
            "f1"
        ),

        "roc_auc": metrics.get(
            "roc_auc"
        ),

        "prAuc": metrics.get(
            "prAuc"
        ),

        # ------------------------------
        # Fraud metrics
        # ------------------------------

        "fraud_precision": metrics.get(
            "fraud_precision"
        ),

        "fraud_recall": metrics.get(
            "fraud_recall"
        ),

        "fraud_f1": metrics.get(
            "fraud_f1"
        ),

        # ------------------------------
        # MIA
        # ------------------------------

        "mia_accuracy": mia.get(
            "attack_accuracy"
        ),

        "mia_auc": mia.get(
            "attack_auc"
        ),

        "attacker_advantage": mia.get(
            "attacker_advantage"
        ),

        "tpr_at_1pct_fpr": mia.get(
            "tpr_at_1pct_fpr"
        ),

        "tpr_at_01pct_fpr": mia.get(
            "tpr_at_01pct_fpr"
        ),

        # ------------------------------
        # Pipeline metadata
        # ------------------------------

        "feature_count": output.get(
            "feature_count"
        ),

        "train_size": output.get(
            "train_size"
        ),

        "test_size": output.get(
            "test_size"
        ),

        "fraud_count_train": output.get(
            "fraud_count_train"
        ),

        "fraud_count_test": output.get(
            "fraud_count_test"
        ),

        "noise_scale": output.get(
            "noiseScale"
        ),

        "composition_method": output.get(
            "composition_method"
        ),

        # ------------------------------
        # Runtime
        # ------------------------------

        "elapsed_seconds": round(
            elapsed,
            2
        ),

        "success": True,

        "error": "",
    }


# ============================================================
# RUN ONE CONFIGURATION
# ============================================================

def run_one(
    engine,
    X,
    y,
    config
):

    epsilon = config["epsilon"]

    print(
        f"\n"
        f"Classifier : {config['classifier']}\n"
        f"Mechanism  : {config['mechanism']}\n"
        f"Epsilon    : {epsilon}\n"
        f"Seed       : {config['seed']}"
    )

    start = time.time()

    output = engine.run_pipeline(

        X=X,

        y=y,

        epsilon=epsilon,

        delta=DELTA,

        sensitivity=SENSITIVITY,

        mechanism=config[
            "mechanism"
        ],

        classifier_type=config[
            "classifier"
        ],

        test_size=TEST_SIZE,

        seed=config[
            "seed"
        ],

        n_shadow=N_SHADOW,
    )

    elapsed = time.time() - start

    result = extract_result(
        output,
        config,
        elapsed
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("PRIVACYGUARD FINAL 54-RUN EXPERIMENT")
    print("=" * 80)
    print("Batch:", BATCH_ID)
    print("=" * 80)

    configs = build_configs()

    if len(configs) != 54:
        raise RuntimeError(
            f"Expected 54 configs, got {len(configs)}"
        )

    print(
        f"\nConfirmed experiment matrix: "
        f"{len(configs)} configurations"
    )

    X, y = load_dataset()

    print("\nCreating fresh PrivacyEngine...")

    engine = PrivacyEngine(
        seed=42
    )

    results = []

    total_start = time.time()

    # ========================================================
    # RUN ALL 54
    # ========================================================

    for index, config in enumerate(
        configs,
        start=1
    ):

        print("\n")
        print("#" * 80)
        print(
            f"EXPERIMENT {index}/54"
        )
        print("#" * 80)

        try:

            result = run_one(
                engine,
                X,
                y,
                config
            )

            results.append(result)

            print(
                "\nSUCCESS"
            )

            print(
                "F1:",
                result["f1"]
            )

            print(
                "PR-AUC:",
                result["prAuc"]
            )

            print(
                "MIA-AUC:",
                result["mia_auc"]
            )

            print(
                "Attacker Advantage:",
                result[
                    "attacker_advantage"
                ]
            )

            print(
                "Runtime:",
                result[
                    "elapsed_seconds"
                ],
                "seconds"
            )

        except Exception as exc:

            print(
                "\n!!! EXPERIMENT FAILED !!!"
            )

            traceback.print_exc()

            results.append({

                "batch_id": BATCH_ID,

                "timestamp":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "classifier":
                    config[
                        "classifier"
                    ],

                "mechanism":
                    config[
                        "mechanism"
                    ],

                "epsilon":
                    config[
                        "epsilon"
                    ],

                "epsilon_feature":
                    (
                        config["epsilon"] / 29
                        if np.isfinite(
                            config["epsilon"]
                        )
                        else float("inf")
                    ),

                "delta":
                    DELTA,

                "seed":
                    config["seed"],

                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
                "prAuc": None,

                "fraud_precision": None,
                "fraud_recall": None,
                "fraud_f1": None,

                "mia_accuracy": None,
                "mia_auc": None,
                "attacker_advantage": None,
                "tpr_at_1pct_fpr": None,
                "tpr_at_01pct_fpr": None,

                "feature_count": None,
                "train_size": None,
                "test_size": None,
                "fraud_count_train": None,
                "fraud_count_test": None,
                "noise_scale": None,
                "composition_method": None,

                "elapsed_seconds": None,

                "success": False,

                "error":
                    f"{type(exc).__name__}: {exc}",
            })

        # ----------------------------------------------------
        # SAVE AFTER EVERY RUN
        # ----------------------------------------------------

        pd.DataFrame(
            results
        ).to_csv(
            RESULT_CSV,
            index=False
        )

        print(
            f"\nSaved progress: "
            f"{len(results)}/54"
        )

    # ========================================================
    # FINAL AUDIT
    # ========================================================

    df = pd.DataFrame(
        results
    )

    total_runtime = (
        time.time()
        - total_start
    )

    successful = int(
        df["success"].sum()
    )

    failed = (
        len(df)
        - successful
    )

    unique_configs = (
        df[
            [
                "classifier",
                "mechanism",
                "epsilon",
                "seed"
            ]
        ]
        .drop_duplicates()
    )

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "prAuc",
        "fraud_precision",
        "fraud_recall",
        "fraud_f1",
        "mia_accuracy",
        "mia_auc",
        "attacker_advantage",
    ]

    missing_metric_rows = df[
        df[
            metric_columns
        ]
        .isnull()
        .any(axis=1)
    ]

    duplicate_counts = (
        df.groupby(
            [
                "classifier",
                "mechanism",
                "epsilon",
                "seed"
            ]
        )
        .size()
    )

    duplicates = duplicate_counts[
        duplicate_counts > 1
    ]

    print("\n")
    print("=" * 80)
    print("FINAL AUDIT")
    print("=" * 80)

    print(
        "Expected configurations:",
        54
    )

    print(
        "Rows produced:",
        len(df)
    )

    print(
        "Successful:",
        successful
    )

    print(
        "Failed:",
        failed
    )

    print(
        "Unique configurations:",
        len(unique_configs)
    )

    print(
        "Rows with missing metrics:",
        len(missing_metric_rows)
    )

    print(
        "Duplicate configurations:",
        len(duplicates)
    )

    print(
        "Total runtime:",
        round(
            total_runtime / 60,
            2
        ),
        "minutes"
    )

    # ========================================================
    # SUMMARY FILE
    # ========================================================

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "PrivacyGuard Final Experiment\n"
        )

        f.write(
            f"Batch ID: {BATCH_ID}\n\n"
        )

        f.write(
            f"Expected: 54\n"
        )

        f.write(
            f"Rows produced: {len(df)}\n"
        )

        f.write(
            f"Successful: {successful}\n"
        )

        f.write(
            f"Failed: {failed}\n"
        )

        f.write(
            f"Unique configurations: "
            f"{len(unique_configs)}\n"
        )

        f.write(
            f"Missing metric rows: "
            f"{len(missing_metric_rows)}\n"
        )

        f.write(
            f"Duplicate configurations: "
            f"{len(duplicates)}\n"
        )

        f.write(
            f"Runtime minutes: "
            f"{total_runtime / 60:.2f}\n"
        )

    print("\n")
    print("=" * 80)

    if (
        len(df) == 54
        and successful == 54
        and len(unique_configs) == 54
        and len(missing_metric_rows) == 0
        and len(duplicates) == 0
    ):

        print(
            "✓ FINAL 54-RUN BATCH COMPLETE"
        )

        print(
            "✓ 54/54 successful"
        )

        print(
            "✓ 54 unique configurations"
        )

        print(
            "✓ All metrics present"
        )

        print(
            "✓ READY FOR REPORT ANALYSIS"
        )

    else:

        print(
            "⚠ FINAL BATCH INCOMPLETE"
        )

        print(
            "Check the audit above."
        )

    print("=" * 80)

    print(
        "\nResults:",
        RESULT_CSV
    )

    print(
        "Summary:",
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()
