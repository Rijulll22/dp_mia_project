import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_runner import get_config_key, run_experiments, FAST_EPSILONS, FAST_SEEDS, FULL_EPSILONS, FULL_SEEDS, ALL_MECHANISMS, ALL_CLASSIFIERS
from database import Experiment, SessionLocal, init_db
import experiment_runner

def test_stable_keys():
    k1 = get_config_key("ulb", "dummy_hash", "logistic_regression", "laplace_clipped", 1.0, 42, 5, 6.0)
    k2 = get_config_key("ulb", "dummy_hash", "logistic_regression", "laplace_clipped", 1.0, 42, 5, 6.0)
    assert k1 == k2

def test_key_differences():
    k_base = get_config_key("ulb", "dummy_hash", "logistic_regression", "laplace_clipped", 1.0, 42, 5, 6.0)
    assert k_base != get_config_key("synthetic", "dummy_hash", "logistic_regression", "laplace_clipped", 1.0, 42, 5, 6.0)
    assert k_base != get_config_key("ulb", "dummy_hash", "random_forest", "laplace_clipped", 1.0, 42, 5, 6.0)
    assert k_base != get_config_key("ulb", "dummy_hash", "logistic_regression", "plain_laplace", 1.0, 42, 5, 6.0)
    assert k_base != get_config_key("ulb", "dummy_hash", "logistic_regression", "laplace_clipped", 5.0, 42, 5, 6.0)
    assert k_base != get_config_key("ulb", "dummy_hash", "logistic_regression", "laplace_clipped", 1.0, 43, 5, 6.0)

def test_90_unique_configurations():
    run_configs = []
    for seed in FULL_SEEDS:
        for clf in ALL_CLASSIFIERS:
            for mech in ALL_MECHANISMS:
                if mech == "non_private":
                    run_configs.append((seed, clf, mech, float('inf')))
                else:
                    for eps in FULL_EPSILONS:
                        run_configs.append((seed, clf, mech, eps))
    
    # Deduplicate non_private runs (they don't depend on epsilon, so we shouldn't iterate eps for them)
    run_configs = list(dict.fromkeys(run_configs))
    
    assert len(run_configs) == 90
    
    non_private = [r for r in run_configs if r[2] == "non_private"]
    private = [r for r in run_configs if r[2] != "non_private"]
    
    assert len(non_private) == 10
    assert len(private) == 80

def test_legacy_rows_not_matched():
    # Insert a legacy row with NO config_key
    with SessionLocal() as db:
        db.query(Experiment).filter(Experiment.id == "EXP-LEGACY").delete()
        legacy_exp = Experiment(
            id="EXP-LEGACY",
            status="completed",
            filename="synthetic_benchmark",
            classifier_type="logistic_regression",
            mechanism="laplace_clipped",
            epsilon=1.0,
            split_seed="42",
            is_simulated=True,
            metrics="{}"
        )
        db.add(legacy_exp)
        db.commit()
    
    k = get_config_key("synthetic_benchmark", "dummy_hash", "logistic_regression", "laplace_clipped", 1.0, 42, 5, 6.0)
    
    # The runner's query ONLY checks config_key
    with SessionLocal() as db:
        existing = db.query(Experiment).filter(
            Experiment.config_key == k
        ).order_by(Experiment.timestamp.desc()).first()
        
        assert existing is None  # Legacy row without config_key is ignored!

if __name__ == "__main__":
    pytest.main([__file__])
