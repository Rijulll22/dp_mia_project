import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import Experiment, SessionLocal
from experiment_runner import run_experiments, ALL_MECHANISMS, ALL_CLASSIFIERS

def test_stale_running_recovery():
    # Insert a stale running record
    stale_id = "EXP-STALE-TEST"
    test_key = "fake_stale_key_for_test"
    with SessionLocal() as db:
        # cleanup old test
        db.query(Experiment).filter(Experiment.config_key == test_key).delete()
        
        stale_exp = Experiment(
            id=stale_id,
            config_key=test_key,
            status="running",
            attempt_count=1,
            classifier_type="logistic_regression",
            mechanism="laplace_clipped",
            split_seed="999",
            epsilon=1.0,
            metrics="{}"
        )
        db.add(stale_exp)
        db.commit()

    # If the runner encounters a "running" status for the config key, it should increment attempt to 2.
    # In run_experiments, it does:
    # if existing:
    #    if existing.attempt_count: attempt = existing.attempt_count + 1
    #    exp_id = existing.id

    # Simulate running the runner for this specific config.
    # We will mock `PrivacyEngine.run_pipeline` to raise an error so it finishes fast.
    with patch("privacy_engine.PrivacyEngine.run_pipeline") as mock_run:
        mock_run.side_effect = Exception("Simulated crash during model training")
        
        with patch("experiment_runner.get_config_key", return_value=test_key):
            run_experiments(
                epsilons=[1.0],
                seeds=[999],
                mechanisms=["laplace_clipped"],
                classifiers=["logistic_regression"],
                dataset_name="synthetic_benchmark"
            )
            
    # Now check the DB
    with SessionLocal() as db:
        exp = db.query(Experiment).filter(Experiment.config_key == test_key).first()
        assert exp is not None
        assert exp.id == stale_id
        assert exp.status == "failed"
        assert exp.attempt_count == 2
        assert "Simulated crash" in exp.error_message

if __name__ == "__main__":
    pytest.main(["-v", __file__])
