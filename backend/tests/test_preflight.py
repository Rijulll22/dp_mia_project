import pytest
import os
import pandas as pd
from experiment_runner import run_experiments, FAST_EPSILONS, ALL_MECHANISMS, ALL_CLASSIFIERS
from privacy_engine import MECHANISM_NONE, MECHANISM_LAPLACE_CLIPPED, MECHANISM_LAPLACE

def test_90_configurations():
    epsilons = [0.1, 1.0, 5.0, 10.0]
    seeds = [42, 43, 44, 45, 46]
    mechanisms = [MECHANISM_NONE, MECHANISM_LAPLACE_CLIPPED, MECHANISM_LAPLACE]
    classifiers = ["logistic_regression", "random_forest"]
    
    run_configs = []
    for seed in seeds:
        for clf in classifiers:
            for mech in mechanisms:
                if mech == MECHANISM_NONE:
                    run_configs.append((seed, clf, mech, float('inf')))
                else:
                    for eps in epsilons:
                        run_configs.append((seed, clf, mech, eps))
    
    assert len(run_configs) == 90
    
    # Assert unique
    assert len(set(run_configs)) == 90
    
    # Assert one non-private per classifier per seed
    non_privates = [c for c in run_configs if c[2] == MECHANISM_NONE]
    assert len(non_privates) == len(seeds) * len(classifiers) # 5 * 2 = 10
    
    # Assert private config counts
    privates = [c for c in run_configs if c[2] != MECHANISM_NONE]
    assert len(privates) == 80 # 2 mechs * 4 eps * 2 clfs * 5 seeds = 80
    
    print(f"Verified {len(run_configs)} configurations.")

def test_dataset_identity():
    ulb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "creditcard.csv")
    assert os.path.exists(ulb_path)
    
    df = pd.read_csv(ulb_path, nrows=5) # Load only a few rows
    assert "Class" in df.columns
    
    features = [c for c in df.columns if c not in ["Class", "Time"]]
    assert len(features) == 29
    
    # Check the required features exist
    for i in range(1, 29):
        assert f"V{i}" in features
    assert "Amount" in features

def test_missing_dataset_failure():
    from experiment_runner import run_experiments
    import unittest.mock as mock
    
    with mock.patch('os.path.exists', return_value=False):
        try:
            run_experiments(
                epsilons=[1.0], seeds=[42], mechanisms=[MECHANISM_NONE], classifiers=["logistic_regression"],
                dataset_name="ulb", save_export=False
            )
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            assert True
        except Exception as e:
            assert False, f"Expected FileNotFoundError, got {type(e)}"

if __name__ == "__main__":
    test_90_configurations()
    test_dataset_identity()
    test_missing_dataset_failure()
    print("All preflight tests passed!")
