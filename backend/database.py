import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, String, Float, Text, Boolean, Integer, text
from sqlalchemy.orm import declarative_base, sessionmaker

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "privacyguard.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ExperimentAttempt(Base):
    __tablename__ = "experiment_attempts"
    
    id = Column(String, primary_key=True, index=True)
    experiment_id = Column(String, index=True)
    config_key = Column(String, index=True)
    attempt_number = Column(Integer)
    status = Column(String)  # running, failed, completed
    error_message = Column(Text, nullable=True)
    start_time = Column(String)
    end_time = Column(String, nullable=True)

class Experiment(Base):
    __tablename__ = "experiments"

    # Identity
    id = Column(String, primary_key=True, index=True)
    timestamp = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())
    filename = Column(String, default="synthetic_benchmark")
    is_simulated = Column(Boolean, default=False)  # True for old/fake records

    # Run state and deterministic configuration identity
    config_key = Column(String, nullable=True, index=True)
    status = Column(String, default="completed") # pending, running, completed, failed
    error_message = Column(Text, nullable=True)
    attempt_count = Column(Integer, default=1)

    # Dataset metadata
    split_seed = Column(String, nullable=True)          # JSON list of seeds used
    train_size = Column(Float, nullable=True)
    test_size = Column(Float, nullable=True)
    fraud_count_train = Column(Float, nullable=True)
    fraud_count_test = Column(Float, nullable=True)
    feature_list = Column(Text, nullable=True)          # JSON list of feature names

    # DP configuration
    epsilon = Column(Float, nullable=False)
    delta = Column(Float, default=0.00001)
    mechanism = Column(String, default="laplace_clipped")
    mechanism_type = Column(String, default="laplace_clipped")
    classifier_type = Column(String, default="logistic_regression")
    noise_scale = Column(Float, default=1.0)

    # Epsilon accounting (new — correct)
    epsilon_record = Column(Float, nullable=True)       # Total record-level epsilon
    epsilon_feature = Column(Float, nullable=True)      # Per-feature epsilon
    feature_count = Column(Float, nullable=True)        # Number of perturbed features
    composition_method = Column(String, nullable=True)  # e.g. "basic sequential composition"

    # Utility metrics (JSON blob)
    metrics = Column(Text, nullable=False)              # Full metrics dict

    # Individual metric columns for quick queries
    pr_auc = Column(Float, nullable=True)               # Precision-Recall AUC
    fraud_precision = Column(Float, nullable=True)      # Fraud class precision
    fraud_recall = Column(Float, nullable=True)         # Fraud class recall
    fraud_f1 = Column(Float, nullable=True)             # Fraud class F1

    # MIA metrics
    mia_auc = Column(Float, nullable=True)
    mia_accuracy = Column(Float, nullable=True)
    attacker_advantage = Column(Float, nullable=True)
    attacker_advantage_05 = Column(Float, nullable=True)
    tpr_at_1fpr = Column(Float, nullable=True)          # TPR @ 1% FPR
    tpr_at_01fpr = Column(Float, nullable=True)         # TPR @ 0.1% FPR

    # Per-class MIA (stored as JSON)
    mia_per_class = Column(Text, nullable=True)         # JSON: {all, legitimate, fraud}

    # Attribute inference (future work — stored as NULL until implemented)
    attribute_inference_mae = Column(Float, nullable=True)
    attribute_inference_rmse = Column(Float, nullable=True)

    # Reconstruction (REMOVED as a research claim — column kept for schema compat)
    reconstruction_nmse = Column(Float, nullable=True)  # DEPRECATED — do not use for research

    # Experiment curves
    pareto_curve = Column(Text, nullable=True)          # JSON list of TradeoffPoints
    baselines = Column(Text, nullable=True)             # JSON baselines

    # File artifacts
    perturbed_dataset_path = Column(String, nullable=True) # Deprecated, not populated automatically
    model_path = Column(String, nullable=True)
    export_path = Column(String, nullable=True) # Explicit export of noised matrix

    # Legacy columns kept for backward compat
    relative_utility_loss = Column(Float, nullable=True)
    espinosa_benchmark = Column(Text, nullable=True)    # DEPRECATED — do not populate


def init_db():
    """Create all SQLite database tables and apply backward-compatible column migrations."""
    Base.metadata.create_all(bind=engine)

    # Apply backward-compatible migrations for existing databases
    new_columns = [
        ("config_key", "TEXT"),
        ("status", "VARCHAR DEFAULT 'completed'"),
        ("error_message", "TEXT"),
        ("attempt_count", "INTEGER DEFAULT 1"),
        ("is_simulated", "BOOLEAN DEFAULT 0"),
        ("split_seed", "TEXT"),
        ("train_size", "FLOAT"),
        ("test_size", "FLOAT"),
        ("fraud_count_train", "FLOAT"),
        ("fraud_count_test", "FLOAT"),
        ("feature_list", "TEXT"),
        ("epsilon_record", "FLOAT"),
        ("epsilon_feature", "FLOAT"),
        ("feature_count", "FLOAT"),
        ("composition_method", "VARCHAR"),
        ("pr_auc", "FLOAT"),
        ("fraud_precision", "FLOAT"),
        ("fraud_recall", "FLOAT"),
        ("fraud_f1", "FLOAT"),
        ("mia_accuracy", "FLOAT"),
        ("attacker_advantage_05", "FLOAT"),
        ("tpr_at_1fpr", "FLOAT"),
        ("tpr_at_01fpr", "FLOAT"),
        ("mia_per_class", "TEXT"),
        ("export_path", "VARCHAR"),
        ("attribute_inference_mae", "FLOAT"),
        ("attribute_inference_rmse", "FLOAT"),
        # Old columns that may exist
        ("mechanism_type", "VARCHAR DEFAULT 'laplace_clipped'"),
        ("classifier_type", "VARCHAR DEFAULT 'logistic_regression'"),
        ("mia_auc", "FLOAT"),
        ("attacker_advantage", "FLOAT"),
        ("reconstruction_nmse", "FLOAT"),
        ("relative_utility_loss", "FLOAT"),
        ("baselines", "TEXT"),
        ("espinosa_benchmark", "TEXT"),
    ]

    with engine.connect() as conn:
        try:
            result = conn.execute(text("PRAGMA table_info(experiments);"))
            existing_cols = {row[1] for row in result.fetchall()}

            for col_name, col_def in new_columns:
                if col_name not in existing_cols:
                    conn.execute(text(f"ALTER TABLE experiments ADD COLUMN {col_name} {col_def};"))

            # Mark all pre-existing experiments as simulated (they were generated with fake data)
            conn.execute(text(
                "UPDATE experiments SET is_simulated = 1 WHERE is_simulated IS NULL OR is_simulated = 0"
                " AND id IN ('EXP-001','EXP-002','EXP-003','EXP-004','EXP-005')"
            ))
            # Add unique index for config_key
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_experiments_config_key ON experiments(config_key) WHERE config_key IS NOT NULL;"))

            conn.commit()
        except Exception as e:
            print("Database migration check note:", e)


def get_db():
    """FastAPI database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
