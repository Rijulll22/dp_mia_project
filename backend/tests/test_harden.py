import os
import sys
import pytest
import sqlite3
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import Experiment, ExperimentAttempt, SessionLocal, init_db
from experiment_runner import get_config_key, run_experiments

def test_duplicate_key_rejection():
    # Attempt to insert two records with the same non-null config_key
    with SessionLocal() as db:
        test_key = "test_duplicate_key_123"
        
        # Cleanup
        db.query(Experiment).filter(Experiment.config_key == test_key).delete()
        
        exp1 = Experiment(id="EXP-DUP-1", config_key=test_key, epsilon=1.0, metrics="{}")
        db.add(exp1)
        db.commit()
        
        exp2 = Experiment(id="EXP-DUP-2", config_key=test_key, epsilon=1.0, metrics="{}")
        db.add(exp2)
        
        with pytest.raises(Exception) as excinfo:
            db.commit()
        assert "UNIQUE constraint failed" in str(excinfo.value)
        db.rollback()

def test_dataset_fingerprint_changes():
    # Verify that changing the dataset hash changes the config_key
    k1 = get_config_key("ulb", "hash1", "logistic_regression", "laplace_clipped", 1.0, 42, 5, 6.0)
    k2 = get_config_key("ulb", "hash2", "logistic_regression", "laplace_clipped", 1.0, 42, 5, 6.0)
    assert k1 != k2

def test_retry_history_preservation():
    # Verify that updating an experiment adds a new ExperimentAttempt row
    test_key = "test_retry_history_123"
    
    with SessionLocal() as db:
        db.query(ExperimentAttempt).filter(ExperimentAttempt.config_key == test_key).delete()
        db.query(Experiment).filter(Experiment.config_key == test_key).delete()
        db.commit()
    
    # Run the experiment twice (simulating a retry after failure)
    from experiment_runner import update_db_record
    
    update_db_record("EXP-RETRY", "failed", attempt=1, error_msg="Error 1", config_key=test_key, eps=1.0, out={})
    update_db_record("EXP-RETRY", "running", attempt=2, error_msg=None, config_key=test_key, eps=1.0, out={})
    update_db_record("EXP-RETRY", "completed", attempt=2, error_msg=None, config_key=test_key, eps=1.0, out={})
    
    with SessionLocal() as db:
        attempts = db.query(ExperimentAttempt).filter(ExperimentAttempt.config_key == test_key).order_by(ExperimentAttempt.attempt_number).all()
        assert len(attempts) == 3
        assert attempts[0].status == "failed"
        assert attempts[1].status == "running"
        assert attempts[2].status == "completed"

def test_concurrent_runner_refusal():
    # Manually acquire the lock file to simulate another runner
    lock_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment_runner.lock")
    lock_file = open(lock_path, "w")
    import msvcrt
    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    
    try:
        with pytest.raises(SystemExit) as excinfo:
            run_experiments(
                epsilons=[1.0],
                seeds=[999],
                mechanisms=["laplace_clipped"],
                classifiers=["logistic_regression"],
                dataset_name="synthetic_benchmark"
            )
        assert excinfo.value.code == 1
    finally:
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        lock_file.close()

if __name__ == "__main__":
    pytest.main(["-v", __file__])
