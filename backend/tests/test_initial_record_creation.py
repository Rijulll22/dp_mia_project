import os
import sys
import json
import pytest
from unittest.mock import patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Base, Experiment, ExperimentAttempt
import database
import experiment_runner
from experiment_runner import update_db_record, get_config_key, MECHANISM_LAPLACE_CLIPPED, MECHANISM_NONE

@pytest.fixture
def isolated_db(tmp_path):
    """
    Creates an isolated temporary SQLite database for each test.
    Ensures backend/privacyguard.db is completely untouched.
    """
    db_file = tmp_path / "test_isolated.db"
    test_db_url = f"sqlite:///{db_file}"
    test_engine = create_engine(test_db_url, connect_args={"check_same_thread": False})
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    # Create tables and unique index
    Base.metadata.create_all(bind=test_engine)
    with test_engine.connect() as conn:
        from sqlalchemy import text
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_experiments_config_key ON experiments(config_key) WHERE config_key IS NOT NULL;"))
        conn.commit()

    with patch.object(database, "SessionLocal", TestSessionLocal), \
         patch.object(experiment_runner, "SessionLocal", TestSessionLocal):
        yield TestSessionLocal

    test_engine.dispose()

def test_new_running_record_insertion_constraints(isolated_db):
    """
    Verifies that creating a new record with status='running' satisfies all
    database schema NOT NULL constraints (specifically epsilon and metrics).
    """
    exp_id = "EXP-NEW-001"
    config_key = "test_config_key_running_001"
    
    # 1. Private run
    update_db_record(
        exp_id=exp_id,
        status="running",
        attempt=1,
        config_key=config_key,
        eps=1.0,
        mech=MECHANISM_LAPLACE_CLIPPED,
        clf="logistic_regression",
        seed=42,
        dataset_name="ulb"
    )

    with isolated_db() as db:
        rec = db.query(Experiment).filter(Experiment.id == exp_id).first()
        assert rec is not None
        assert rec.config_key == config_key
        assert rec.status == "running"
        assert rec.attempt_count == 1
        assert rec.epsilon == 1.0
        assert rec.metrics == "{}"
        assert rec.classifier_type == "logistic_regression"
        assert rec.mechanism == MECHANISM_LAPLACE_CLIPPED
        assert rec.split_seed == "42"
        assert rec.filename == "ulb"
        assert rec.error_message is None

        # Verify attempt record was logged
        att = db.query(ExperimentAttempt).filter(ExperimentAttempt.experiment_id == exp_id).first()
        assert att is not None
        assert att.status == "running"
        assert att.attempt_number == 1

    # 2. Non-private run (infinite epsilon)
    exp_id_np = "EXP-NEW-002"
    config_key_np = "test_config_key_running_002"
    update_db_record(
        exp_id=exp_id_np,
        status="running",
        attempt=1,
        config_key=config_key_np,
        eps=float('inf'),
        mech=MECHANISM_NONE,
        clf="random_forest",
        seed=43,
        dataset_name="ulb"
    )

    with isolated_db() as db:
        rec_np = db.query(Experiment).filter(Experiment.id == exp_id_np).first()
        assert rec_np is not None
        assert rec_np.epsilon == float('inf')
        assert rec_np.metrics == "{}"
        assert rec_np.mechanism == MECHANISM_NONE
        assert rec_np.classifier_type == "random_forest"

def test_update_running_record_to_completed_persists_actual_metrics(isolated_db):
    """
    Verifies that when a running record completes, actual metrics from pipeline output
    replace the initial empty metrics and all metadata is preserved.
    """
    exp_id = "EXP-COMPLETE-001"
    config_key = "test_config_key_complete_001"

    # Step 1: Initial running record creation
    update_db_record(
        exp_id=exp_id,
        status="running",
        attempt=1,
        config_key=config_key,
        eps=5.0,
        mech=MECHANISM_LAPLACE_CLIPPED,
        clf="logistic_regression",
        seed=44,
        dataset_name="ulb"
    )

    # Step 2: Pipeline completes with actual metrics
    pipeline_out = {
        "metrics": {
            "accuracy": 0.9982,
            "precision": 0.8521,
            "recall": 0.7245,
            "f1": 0.7831,
            "rocAuc": 0.9412,
            "prAuc": 0.7105,
            "fraud_f1": 0.7831
        },
        "mia_result": {
            "attack_auc": 0.5012,
            "attack_accuracy": 0.5005,
            "attacker_advantage": 0.0024
        },
        "train_size": 113922,
        "test_size": 56962,
        "fraud_count_train": 197,
        "fraud_count_test": 98,
        "feature_list": ["V1", "V2", "Amount"]
    }

    update_db_record(
        exp_id=exp_id,
        status="completed",
        attempt=1,
        error_msg=None,
        out=pipeline_out,
        eps=5.0,
        mech=MECHANISM_LAPLACE_CLIPPED,
        clf="logistic_regression",
        seed=44,
        dataset_name="ulb",
        config_key=config_key
    )

    with isolated_db() as db:
        rec = db.query(Experiment).filter(Experiment.id == exp_id).first()
        assert rec.status == "completed"
        assert rec.epsilon == 5.0
        
        # Verify metrics is no longer empty "{}"
        metrics = json.loads(rec.metrics)
        assert metrics["f1"] == 0.7831
        assert metrics["rocAuc"] == 0.9412
        assert rec.pr_auc == 0.7105
        assert rec.mia_auc == 0.5012
        assert rec.train_size == 113922
        assert rec.test_size == 56962

        # Verify attempts history has 2 records
        attempts = db.query(ExperimentAttempt).filter(ExperimentAttempt.experiment_id == exp_id).order_by(ExperimentAttempt.start_time).all()
        assert len(attempts) == 2
        assert attempts[0].status == "running"
        assert attempts[1].status == "completed"

def test_failed_attempt_recoverable_without_duplicates(isolated_db):
    """
    Verifies that when a run fails and is subsequently resumed:
    1. The existing experiment ID is reused.
    2. attempt_count increments.
    3. No duplicate canonical records are created (respecting UNIQUE config_key).
    4. Attempt history preserves the failure and the subsequent recovery.
    """
    test_key = "test_config_key_recover_001"
    exp_id = "EXP-RECOVER-001"

    # Run attempt 1: running then failed
    update_db_record(
        exp_id=exp_id, status="running", attempt=1, config_key=test_key,
        eps=1.0, mech=MECHANISM_LAPLACE_CLIPPED, clf="logistic_regression", seed=42, dataset_name="ulb"
    )
    update_db_record(
        exp_id=exp_id, status="failed", attempt=1, error_msg="Simulated OOM crash", config_key=test_key,
        eps=1.0, mech=MECHANISM_LAPLACE_CLIPPED, clf="logistic_regression", seed=42, dataset_name="ulb"
    )

    with isolated_db() as db:
        rec = db.query(Experiment).filter(Experiment.config_key == test_key).first()
        assert rec.status == "failed"
        assert rec.attempt_count == 1
        assert "Simulated OOM" in rec.error_message

    # Simulate runner restarting for the same config_key
    with isolated_db() as db:
        existing = db.query(Experiment).filter(Experiment.config_key == test_key).first()
        assert existing is not None
        assert existing.status != "completed"
        # Increment attempt and reuse ID
        next_attempt = (existing.attempt_count or 1) + 1
        resumed_exp_id = existing.id
        assert resumed_exp_id == exp_id

    # Run attempt 2: mark running then completed
    update_db_record(
        exp_id=resumed_exp_id, status="running", attempt=next_attempt, config_key=test_key,
        eps=1.0, mech=MECHANISM_LAPLACE_CLIPPED, clf="logistic_regression", seed=42, dataset_name="ulb"
    )
    success_out = {"metrics": {"f1": 0.81}, "train_size": 113922}
    update_db_record(
        exp_id=resumed_exp_id, status="completed", attempt=next_attempt, error_msg=None, out=success_out,
        eps=1.0, mech=MECHANISM_LAPLACE_CLIPPED, clf="logistic_regression", seed=42, dataset_name="ulb", config_key=test_key
    )

    with isolated_db() as db:
        # Check that ONLY ONE record exists in experiments table
        all_recs = db.query(Experiment).filter(Experiment.config_key == test_key).all()
        assert len(all_recs) == 1
        rec = all_recs[0]
        assert rec.id == exp_id
        assert rec.status == "completed"
        assert rec.attempt_count == 2
        assert rec.error_message is None

        # Check full history in experiment_attempts (running, failed, running, completed)
        atts = db.query(ExperimentAttempt).filter(ExperimentAttempt.config_key == test_key).order_by(ExperimentAttempt.start_time).all()
        assert len(atts) == 4
        assert [a.status for a in atts] == ["running", "failed", "running", "completed"]

def test_completed_records_skipped(isolated_db):
    """
    Verifies that already completed configurations are recognized and skipped.
    """
    test_key = "test_config_key_skip_001"
    exp_id = "EXP-SKIP-001"

    # Insert completed record
    update_db_record(
        exp_id=exp_id, status="running", attempt=1, config_key=test_key,
        eps=1.0, mech=MECHANISM_LAPLACE_CLIPPED, clf="logistic_regression", seed=42, dataset_name="ulb"
    )
    update_db_record(
        exp_id=exp_id, status="completed", attempt=1, out={"metrics": {"f1": 0.90}},
        eps=1.0, mech=MECHANISM_LAPLACE_CLIPPED, clf="logistic_regression", seed=42, dataset_name="ulb", config_key=test_key
    )

    # Runner check
    with isolated_db() as db:
        existing = db.query(Experiment).filter(Experiment.config_key == test_key).first()
        assert existing is not None
        assert existing.status == "completed"

        # Simulating runner skip condition
        should_skip = (existing and existing.status == "completed")
        assert should_skip is True

def test_non_private_configuration_safety(isolated_db):
    """
    Verifies that non-private configurations with epsilon=float('inf'):
    1. Satisfy SQLite schema constraints and can be stored/read back as IEEE 754 infinity.
    2. Produce deterministic config_key identical across representations.
    3. Export and import cleanly to/from CSV via pandas with infinite floats intact.
    4. Support pandas and SQL aggregations without error.
    5. Serialize safely to valid RFC-compliant JSON via FastAPI endpoints without crashing Starlette.
    """
    import math
    import pandas as pd
    from io import StringIO
    from fastapi.testclient import TestClient
    import main

    exp_id = "EXP-NONPRIV-001"
    config_key = get_config_key("ulb", "hash123", "logistic_regression", MECHANISM_NONE, float('inf'), 42, 5, 6.0)

    # 1. Initial insert into isolated SQLite
    update_db_record(
        exp_id=exp_id,
        status="running",
        attempt=1,
        config_key=config_key,
        eps=float('inf'),
        mech=MECHANISM_NONE,
        clf="logistic_regression",
        seed=42,
        dataset_name="ulb"
    )

    with isolated_db() as db:
        rec = db.query(Experiment).filter(Experiment.id == exp_id).first()
        assert rec is not None
        assert math.isinf(rec.epsilon)
        assert rec.metrics == "{}"

    # 2. Update to completed with actual metrics
    pipeline_out = {
        "metrics": {"accuracy": 0.999, "f1": 0.85, "rocAuc": 0.98},
        "mia_result": {"attack_auc": 0.52, "attack_accuracy": 0.51, "attacker_advantage": 0.02},
        "epsilon_accounting": {"composition_method": "non-private"},
        "train_size": 113922,
        "test_size": 56962
    }
    update_db_record(
        exp_id=exp_id,
        status="completed",
        attempt=1,
        out=pipeline_out,
        eps=float('inf'),
        mech=MECHANISM_NONE,
        clf="logistic_regression",
        seed=42,
        dataset_name="ulb",
        config_key=config_key
    )

    with isolated_db() as db:
        rec = db.query(Experiment).filter(Experiment.id == exp_id).first()
        assert rec.status == "completed"
        assert math.isinf(rec.epsilon)
        assert json.loads(rec.metrics)["f1"] == 0.85

    # 3. Canonical config_key invariance
    k_inf = get_config_key("ulb", "hash123", "logistic_regression", MECHANISM_NONE, float('inf'), 42, 5, 6.0)
    k_none = get_config_key("ulb", "hash123", "logistic_regression", MECHANISM_NONE, None, 42, 5, 6.0)
    assert k_inf == k_none == config_key

    # 4. CSV export & pandas aggregation
    with isolated_db() as db:
        rows = db.query(Experiment.id, Experiment.mechanism, Experiment.epsilon, Experiment.status).all()
        df = pd.DataFrame(rows, columns=["id", "mechanism", "epsilon", "status"])
        
    csv_buf = StringIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    df_read = pd.read_csv(csv_buf)
    assert df_read.loc[df_read["id"] == exp_id, "epsilon"].apply(math.isinf).iloc[0]

    # Metrics aggregation
    grouped = df.groupby(["mechanism", "epsilon"]).size()
    assert (MECHANISM_NONE, float('inf')) in grouped.index

    # 5. JSON serialization via FastAPI TestClient
    with patch.object(main, "SessionLocal", isolated_db):
        client = TestClient(main.app)
        
        # Test list endpoint
        res_list = client.get("/api/experiments")
        assert res_list.status_code == 200, f"Failed: {res_list.text}"
        items = res_list.json()
        assert len(items) == 1
        item = items[0]
        assert item["id"] == exp_id
        # In JSON response, non-finite float is represented as null / None
        assert item["epsilon"] is None
        assert item["mechanism"] == MECHANISM_NONE

        # Test single item endpoint
        res_detail = client.get(f"/api/results/{exp_id}")
        assert res_detail.status_code == 200, f"Failed: {res_detail.text}"
        detail = res_detail.json()
        assert detail["experimentId"] == exp_id
        assert detail["epsilon"] is None
        assert detail["epsilon_accounting"]["epsilon_record"] is None

