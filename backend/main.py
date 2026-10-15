"""
main.py — PrivacyGuard FastAPI Backend

Scientific correctness guarantees:
- No simulated or hard-coded research results returned from any endpoint.
- If the DB is empty, /api/experiments returns an empty list (not fake seeds).
- /api/results/{id} returns 404 if experiment not found (not fabricated values).
- All metrics come from actual executed experiments.
- AES/Fernet encryption is NOT claimed — data is processed in plaintext in RAM.
"""

import os
import json
import shutil
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from schemas import (
    HealthResponse,
    UploadResponse,
    PrivacyProcessRequest,
    PrivacyProcessResponse,
    UtilityMetrics,
    ConfusionMatrix,
    TradeoffPoint,
    EpsilonAccounting,
    MIAMetrics,
    ExperimentListItem,
    ExperimentRunRequest,
)
from privacy_engine import PrivacyEngine, MECHANISM_LAPLACE_CLIPPED, MECHANISM_NONE
from database import init_db, Experiment, SessionLocal

# Initialize SQLite tables (with migrations)
init_db()

# Initialize app
app = FastAPI(
    title="PrivacyGuard API",
    description=(
        "Privacy-Preserving Fraud Detection Research Backend. "
        "All metrics come from real experiments — no simulation."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Shared in-memory state
privacy_engine = PrivacyEngine(seed=42)

# Default synthetic dataset (clearly labeled as synthetic)
default_X, default_y = privacy_engine.generate_benchmark_dataset(n_samples=5000)

current_dataset: Dict[str, Any] = {
    "id": "DS-SYNTHETIC-BENCHMARK",
    "name": "synthetic_benchmark",
    "X": default_X,
    "y": default_y,
    "target_col": "Class",
    "is_synthetic": True,
}

# In-memory cache for recent experiment responses
experiments_cache: Dict[str, Dict[str, Any]] = {}


# ─── Helpers ──────────────────────────────────────────────────────────────

def _format_pct(val: Optional[float]) -> Optional[str]:
    """Format a float metric as percentage string, or None if missing."""
    if val is None:
        return None
    return f"{val * 100:.1f}%"


def _format_num(val: Optional[float], decimals: int = 4) -> Optional[str]:
    """Format a float as string with given decimals, or None if missing."""
    if val is None:
        return None
    return f"{val:.{decimals}f}"


def _save_experiment_to_db(
    experiment_id: str,
    req: PrivacyProcessRequest,
    pipeline_output: Dict[str, Any],
    model_json_path: str,
    export_path: Optional[str] = None,
):
    """Persist experiment results to SQLite. Never overwrites existing records."""
    metrics = pipeline_output.get("metrics", {})
    mia = pipeline_output.get("mia_result") or {}
    eps_acc = pipeline_output.get("epsilon_accounting") or {}

    with SessionLocal() as db:
        exp_record = Experiment(
            id=experiment_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            filename=current_dataset["name"],
            is_simulated=False,

            # Dataset metadata
            split_seed=str(pipeline_output.get("split_seed", req.seed or 42)),
            train_size=pipeline_output.get("train_size"),
            test_size=pipeline_output.get("test_size"),
            fraud_count_train=pipeline_output.get("fraud_count_train"),
            fraud_count_test=pipeline_output.get("fraud_count_test"),
            feature_list=json.dumps(pipeline_output.get("feature_list", [])),

            # DP config
            epsilon=req.epsilon,
            delta=req.delta or 0.00001,
            mechanism=pipeline_output.get("mechanism_type", MECHANISM_LAPLACE_CLIPPED),
            mechanism_type=pipeline_output.get("mechanism_type", MECHANISM_LAPLACE_CLIPPED),
            classifier_type=pipeline_output.get("classifier_type", "logistic_regression"),
            noise_scale=pipeline_output.get("noiseScale", 0.0),

            # Epsilon accounting
            epsilon_record=eps_acc.get("epsilon_record", req.epsilon),
            epsilon_feature=eps_acc.get("epsilon_feature"),
            feature_count=eps_acc.get("feature_count"),
            composition_method=eps_acc.get("composition_method"),

            # Metrics (full JSON)
            metrics=json.dumps(metrics),

            # Individual metric columns
            pr_auc=metrics.get("prAuc"),
            fraud_precision=metrics.get("fraud_precision"),
            fraud_recall=metrics.get("fraud_recall"),
            fraud_f1=metrics.get("fraud_f1"),

            # MIA
            mia_auc=mia.get("attack_auc"),
            mia_accuracy=mia.get("attack_accuracy"),
            attacker_advantage=mia.get("attacker_advantage"),
            tpr_at_1fpr=mia.get("tpr_at_1fpr"),
            tpr_at_01fpr=mia.get("tpr_at_01fpr"),
            mia_per_class=json.dumps(pipeline_output.get("mia_per_class", {})),

            # Attribute inference (future work)
            attribute_inference_mae=None,
            attribute_inference_rmse=None,

            # Curves
            pareto_curve=json.dumps(pipeline_output.get("tradeoffCurve", [])),
            baselines=json.dumps(pipeline_output.get("baselines", {})),

            # File artifacts
            perturbed_dataset_path=None, # Deprecated
            model_path=model_json_path,
            export_path=export_path,

            # Other
            relative_utility_loss=pipeline_output.get("relative_utility_loss"),
        )
        db.add(exp_record)
        db.commit()


# ─── Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        message=(
            "PrivacyGuard Differential Privacy Research API is running. "
            "All metrics come from real experiments."
        ),
        version="2.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Handles CSV dataset upload and metadata extraction.
    Note: Data is processed in RAM (plaintext). No encryption is applied.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are accepted.",
        )

    file_id = f"DS-{uuid.uuid4().hex[:8].upper()}"
    saved_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size_bytes = os.path.getsize(saved_path)
        file_size_mb = f"{file_size_bytes / (1024 * 1024):.1f} MB"

        df = pd.read_csv(saved_path)
        records_count = len(df)
        columns_count = len(df.columns)
        column_names = list(df.columns)

        # Detect target column
        target_candidate = None
        for col in ["Class", "class", "target", "label", "is_fraud", "fraud"]:
            if col in df.columns:
                target_candidate = col
                break
        if not target_candidate:
            target_candidate = column_names[-1]

        y_series = df[target_candidate]
        X_df = df.drop(columns=[target_candidate])

        X_numeric = X_df.select_dtypes(include=["number"])
        if X_numeric.shape[1] == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset must contain numeric features.",
            )
        X_clean = X_numeric.fillna(X_numeric.median())

        # Update shared dataset state
        current_dataset["id"] = file_id
        current_dataset["name"] = file.filename
        current_dataset["X"] = X_clean
        current_dataset["y"] = y_series
        current_dataset["target_col"] = target_candidate
        current_dataset["is_synthetic"] = False

        preview_df = df.head(5).fillna("")
        preview_rows = preview_df.astype(str).to_dict(orient="records")

        return UploadResponse(
            success=True,
            message=(
                "Dataset uploaded and parsed successfully. "
                "Data is stored in RAM (plaintext) for processing."
            ),
            datasetId=file_id,
            filename=file.filename,
            filesize=file_size_mb,
            records=records_count,
            columns=columns_count,
            columnNames=column_names,
            targetColumn=target_candidate,
            missingValues=int(df.isnull().sum().sum()),
            previewRows=preview_rows,
        )

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process dataset: {str(e)}",
        )


@app.post("/api/process-privacy", response_model=PrivacyProcessResponse)
async def process_privacy(req: PrivacyProcessRequest):
    """
    Executes the DP training pipeline and returns real experiment results.

    Correct epsilon accounting:
    - Input epsilon is record-level epsilon_record.
    - Per-feature epsilon = epsilon_record / feature_count.
    - scale = sensitivity / epsilon_feature.

    No simulated values are returned.
    """
    X = current_dataset["X"]
    y = current_dataset["y"]

    # Ensure binary labels
    y_unique = y.unique()
    if len(y_unique) > 2:
        top_class = y.mode()[0]
        y_binary = (y != top_class).astype(int)
    else:
        y_binary = y.astype(int)

    # Normalize mechanism and classifier names
    mech_raw = req.mechanism_type or req.mechanism or MECHANISM_LAPLACE_CLIPPED
    clf_type = req.classifier_type or "logistic_regression"
    seed = req.seed or 42
    n_shadow = req.n_shadow or 5

    try:
        pipeline_output = privacy_engine.run_pipeline(
            X=X,
            y=y_binary,
            epsilon=req.epsilon,
            delta=req.delta or 0.00001,
            sensitivity=req.sensitivity,
            mechanism=mech_raw,
            classifier_type=clf_type,
            test_size=0.20,
            seed=seed,
            n_shadow=n_shadow,
        )

        # Generate tradeoff curve (lightweight, 2 shadow models per epsilon)
        tradeoff_curve = privacy_engine.generate_tradeoff_curve(
            X=X,
            y=y_binary,
            sensitivity=req.sensitivity,
            mechanism=mech_raw,
            delta=req.delta or 0.00001,
            classifier_type=clf_type,
            seed=seed,
        )
        pipeline_output["tradeoffCurve"] = tradeoff_curve
        pipeline_output["pareto_curve"] = tradeoff_curve

        experiment_id = f"EXP-{uuid.uuid4().hex[:6].upper()}"

        # Save artifacts
        model_json_path = os.path.join(UPLOAD_DIR, f"{experiment_id}_model.json")
        export_path = None

        # For single interactive UI runs, we'll optionally save the export if they want it later.
        # However, to avoid disk bloat, we won't automatically save the massive ULB CSV here unless specifically requested.
        # For this implementation, we will skip writing the 284k CSV to disk automatically.

        if "modelArtifacts" in pipeline_output:
            with open(model_json_path, "w") as f:
                json.dump(pipeline_output["modelArtifacts"], f, indent=2)

        # Build response objects
        raw_metrics = pipeline_output.get("metrics", {})
        cm_raw = raw_metrics.get("confusionMatrix", {})

        metrics_obj = UtilityMetrics(
            accuracy=raw_metrics.get("accuracy"),
            precision=raw_metrics.get("precision"),
            recall=raw_metrics.get("recall"),
            f1=raw_metrics.get("f1"),
            rocAuc=raw_metrics.get("rocAuc"),
            prAuc=raw_metrics.get("prAuc"),
            fraud_precision=raw_metrics.get("fraud_precision"),
            fraud_recall=raw_metrics.get("fraud_recall"),
            fraud_f1=raw_metrics.get("fraud_f1"),
            fraud_support=raw_metrics.get("fraud_support"),
            confusionMatrix=ConfusionMatrix(
                tn=cm_raw.get("tn", 0),
                fp=cm_raw.get("fp", 0),
                fn=cm_raw.get("fn", 0),
                tp=cm_raw.get("tp", 0),
            ) if cm_raw else None,
            miaRisk=pipeline_output.get("mia_result", {}).get("attack_accuracy"),
            reconRisk=None,
        )

        eps_acc = pipeline_output.get("epsilon_accounting") or {}
        if eps_acc:
            eps_accounting_obj = EpsilonAccounting(
                epsilon_record=eps_acc["epsilon_record"],
                epsilon_feature=eps_acc["epsilon_feature"],
                feature_count=eps_acc["feature_count"],
                composition_method=eps_acc["composition_method"],
            )
        else:
            eps_accounting_obj = None

        mia_raw = pipeline_output.get("mia_result") or {}
        mia_obj = MIAMetrics(
            attack_accuracy=mia_raw.get("attack_accuracy"),
            attack_auc=mia_raw.get("attack_auc"),
            attacker_advantage=mia_raw.get("attacker_advantage"),
            tpr_at_1fpr=mia_raw.get("tpr_at_1fpr"),
            tpr_at_01fpr=mia_raw.get("tpr_at_01fpr"),
            n_members_evaluated=mia_raw.get("n_members_evaluated"),
            n_nonmembers_evaluated=mia_raw.get("n_nonmembers_evaluated"),
            error=mia_raw.get("error"),
        ) if mia_raw else None

        tradeoff_objs = [
            TradeoffPoint(
                epsilon=pt["epsilon"],
                epsilon_feature=pt.get("epsilon_feature"),
                accuracy=pt.get("accuracy"),
                precision=pt.get("precision"),
                recall=pt.get("recall"),
                f1=pt.get("f1"),
                rocAuc=pt.get("rocAuc"),
                prAuc=pt.get("prAuc"),
                fraud_f1=pt.get("fraud_f1"),
                fraud_recall=pt.get("fraud_recall"),
                miaAccuracy=pt.get("miaAccuracy"),
                miaAuc=pt.get("miaAuc"),
                attackerAdvantage=pt.get("attackerAdvantage"),
                tprAt1Fpr=pt.get("tprAt1Fpr"),
                miaRisk=pt.get("miaRisk"),
                reconRisk=None,
            )
            for pt in tradeoff_curve
        ]

        noise_scale = pipeline_output.get("noiseScale", 0.0)

        response = PrivacyProcessResponse(
            success=True,
            experimentId=experiment_id,
            is_simulated=False,
            epsilon=req.epsilon,
            delta=req.delta or 0.00001,
            sensitivity=req.sensitivity,
            mechanism=pipeline_output.get("mechanism_type", MECHANISM_LAPLACE_CLIPPED),
            mechanism_type=pipeline_output.get("mechanism_type"),
            mechanism_label=pipeline_output.get("mechanism_label"),
            classifier_type=pipeline_output.get("classifier_type"),
            noiseScale=noise_scale,
            epsilon_accounting=eps_accounting_obj,
            split_seed=pipeline_output.get("split_seed"),
            train_size=pipeline_output.get("train_size"),
            test_size=pipeline_output.get("test_size"),
            fraud_count_train=pipeline_output.get("fraud_count_train"),
            fraud_count_test=pipeline_output.get("fraud_count_test"),
            feature_list=pipeline_output.get("feature_list"),
            metrics=metrics_obj,
            mia_result=mia_obj,
            mia_per_class=pipeline_output.get("mia_per_class"),
            tradeoffCurve=tradeoff_objs,
            pareto_curve=tradeoff_curve,
            baselines=pipeline_output.get("baselines"),
            model_comparison=pipeline_output.get("model_comparison"),
            processed_rows=pipeline_output.get("processed_rows"),
            relative_utility_loss=pipeline_output.get("relative_utility_loss"),
            confusion_matrix=pipeline_output.get("confusion_matrix"),
            # Legacy compat
            empirical_attacks={
                "mia": mia_raw,
                "reconstruction": None,  # REMOVED
            },
        )

        # Cache in memory
        experiments_cache[experiment_id] = response.model_dump()

        # Persist to SQLite
        pipeline_output["epsilon_accounting"] = eps_acc
        try:
            _save_experiment_to_db(
                experiment_id, req, pipeline_output,
                model_json_path, export_path
            )
        except Exception as db_err:
            print(f"[PrivacyGuard DB Error] Failed to persist: {db_err}")

        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Experiment execution error: {str(e)}",
        )


@app.get("/api/experiments")
async def list_experiments():
    """
    Retrieve all real experiment runs from SQLite.
    Returns empty list if no experiments have been run.
    Does NOT return fake/seeded experiments.
    """
    results = []
    try:
        with SessionLocal() as db:
            records = db.query(Experiment).order_by(Experiment.timestamp.desc()).all()
            for rec in records:
                try:
                    metrics_data = json.loads(rec.metrics) if rec.metrics else {}
                except Exception:
                    metrics_data = {}

                # MIA AUC for risk label
                mia_auc_val = rec.mia_auc
                if mia_auc_val is not None:
                    if mia_auc_val <= 0.60:
                        risk_label = "LOW"
                    elif mia_auc_val <= 0.70:
                        risk_label = "MEDIUM"
                    else:
                        risk_label = "HIGH"
                else:
                    risk_label = None

                import math
                eps_val = None if (rec.epsilon is not None and math.isinf(rec.epsilon)) else rec.epsilon
                eps_rec = getattr(rec, "epsilon_record", rec.epsilon)
                if eps_rec is not None and math.isinf(eps_rec): eps_rec = None
                
                eps_feat = getattr(rec, "epsilon_feature", None)
                if eps_feat is not None and math.isinf(eps_feat): eps_feat = None
                
                results.append({
                    "id": rec.id,
                    "dataset": rec.filename or "synthetic_benchmark",
                    "epsilon": eps_val,
                    "epsilon_record": eps_rec,
                    "epsilon_feature": eps_feat,
                    "feature_count": getattr(rec, "feature_count", None),
                    "composition_method": getattr(rec, "composition_method", None),
                    "delta": rec.delta,
                    "mechanism": rec.mechanism,
                    "classifier_type": getattr(rec, "classifier_type", None),
                    "date": rec.timestamp[:10] if rec.timestamp else "",
                    "timestamp": rec.timestamp,
                    # Utility
                    "accuracy": _format_pct(metrics_data.get("accuracy")),
                    "precision": _format_pct(metrics_data.get("precision")),
                    "recall": _format_pct(metrics_data.get("recall")),
                    "f1": _format_pct(metrics_data.get("f1")),
                    "rocAuc": _format_pct(metrics_data.get("rocAuc")),
                    "prAuc": _format_pct(metrics_data.get("prAuc")),
                    # Fraud class
                    "fraud_f1": _format_pct(metrics_data.get("fraud_f1")),
                    "fraud_recall": _format_pct(metrics_data.get("fraud_recall")),
                    # MIA
                    "mia_accuracy": _format_num(getattr(rec, "mia_accuracy", None)),
                    "mia_auc": _format_num(mia_auc_val),
                    "attacker_advantage": _format_num(getattr(rec, "attacker_advantage", None)),
                    "tpr_at_1fpr": _format_num(getattr(rec, "tpr_at_1fpr", None)),
                    "tpr_at_01fpr": _format_num(getattr(rec, "tpr_at_01fpr", None)),
                    # Status
                    "privacyRisk": risk_label,
                    "status": "Completed",
                    "is_simulated": bool(getattr(rec, "is_simulated", False)),
                    "hasDataset": bool(getattr(rec, "export_path", False)),
                    "hasModel": os.path.exists(rec.model_path) if rec.model_path else False,
                })
    except Exception as e:
        print(f"[PrivacyGuard DB Error] Query failed: {e}")

    # Return empty list if no experiments — do NOT return fake seeds
    return results


@app.get("/api/results/{experiment_id}")
async def get_experiment_results(experiment_id: str):
    """
    Retrieve stored experiment result by ID.
    Returns 404 if not found — does NOT fabricate results.
    """
    if experiment_id in experiments_cache:
        return experiments_cache[experiment_id]

    with SessionLocal() as db:
        rec = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not rec:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found. Run an experiment first.",
            )

        try:
            metrics_dict = json.loads(rec.metrics) if rec.metrics else {}
        except Exception:
            metrics_dict = {}

        try:
            pareto_list = json.loads(rec.pareto_curve) if rec.pareto_curve else []
        except Exception:
            pareto_list = []

        try:
            baselines_dict = json.loads(rec.baselines) if getattr(rec, "baselines", None) else None
        except Exception:
            baselines_dict = None

        try:
            mia_per_class = json.loads(rec.mia_per_class) if getattr(rec, "mia_per_class", None) else None
        except Exception:
            mia_per_class = None

        cm_dict = metrics_dict.get("confusionMatrix", {})
        cm_matrix = [
            [cm_dict.get("tn", 0), cm_dict.get("fp", 0)],
            [cm_dict.get("fn", 0), cm_dict.get("tp", 0)],
        ]
        processed = (
            cm_dict.get("tn", 0) + cm_dict.get("fp", 0) +
            cm_dict.get("fn", 0) + cm_dict.get("tp", 0)
        )

        mia_auc_val = getattr(rec, "mia_auc", None)
        mia_acc_val = getattr(rec, "mia_accuracy", None)
        adv_val = getattr(rec, "attacker_advantage", None)

        import math
        eps_val = None if (rec.epsilon is not None and math.isinf(rec.epsilon)) else rec.epsilon
        eps_rec = getattr(rec, "epsilon_record", rec.epsilon)
        if eps_rec is not None and math.isinf(eps_rec):
            eps_rec = None
        eps_feat = getattr(rec, "epsilon_feature", None)
        if eps_feat is not None and math.isinf(eps_feat):
            eps_feat = None

        return {
            "success": True,
            "experimentId": rec.id,
            "is_simulated": bool(getattr(rec, "is_simulated", False)),
            "epsilon": eps_val,
            "delta": rec.delta,
            "mechanism": rec.mechanism,
            "mechanism_type": getattr(rec, "mechanism_type", None),
            "classifier_type": getattr(rec, "classifier_type", None),
            "noiseScale": rec.noise_scale,
            # Epsilon accounting
            "epsilon_accounting": {
                "epsilon_record": eps_rec,
                "epsilon_feature": eps_feat,
                "feature_count": getattr(rec, "feature_count", None),
                "composition_method": getattr(rec, "composition_method", None),
            },
            # Dataset split
            "split_seed": getattr(rec, "split_seed", None),
            "train_size": getattr(rec, "train_size", None),
            "test_size": getattr(rec, "test_size", None),
            "fraud_count_train": getattr(rec, "fraud_count_train", None),
            "fraud_count_test": getattr(rec, "fraud_count_test", None),
            # Metrics
            "metrics": metrics_dict,
            # MIA
            "mia_result": {
                "attack_accuracy": mia_acc_val,
                "attack_auc": mia_auc_val,
                "attacker_advantage": adv_val,
                "tpr_at_1fpr": getattr(rec, "tpr_at_1fpr", None),
                "tpr_at_01fpr": getattr(rec, "tpr_at_01fpr", None),
            },
            "mia_per_class": mia_per_class,
            # Curves
            "tradeoffCurve": pareto_list,
            "pareto_curve": pareto_list,
            # Other
            "confusion_matrix": cm_matrix,
            "processed_rows": processed,
            "baselines": baselines_dict,
            "relative_utility_loss": getattr(rec, "relative_utility_loss", None),
            # Legacy compat
            "empirical_attacks": {
                "mia": {
                    "attack_accuracy": mia_acc_val,
                    "attacker_advantage": adv_val,
                    "attack_auc": mia_auc_val,
                },
                "reconstruction": None,  # REMOVED
            },
        }


@app.post("/api/run-experiment")
async def run_experiment_batch(
    req: ExperimentRunRequest,
    background_tasks: BackgroundTasks,
):
    """
    Runs a batch of experiments (multiple seeds × epsilon grid × mechanisms × classifiers).
    Returns immediately with a job ID; experiments run in background.

    Use /api/experiments to poll for results.
    """
    job_id = f"JOB-{uuid.uuid4().hex[:6].upper()}"

    # Resolve mode defaults
    if req.mode == "fast":
        default_epsilons = [0.1, 1.0, 5.0, 10.0]
        default_seeds = [42, 43]
    else:  # full
        default_epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        default_seeds = list(range(42, 52))  # 42-51

    epsilon_grid = req.epsilon_grid or default_epsilons
    seeds = req.seeds or default_seeds
    mechanisms = req.mechanisms or ["laplace_clipped", "plain_laplace"]
    classifier_types = req.classifier_types or ["logistic_regression", "random_forest"]
    n_shadow = req.n_shadow or 5

    X = current_dataset["X"]
    y = current_dataset["y"]
    y_unique = y.unique()
    if len(y_unique) > 2:
        top_class = y.mode()[0]
        y_binary = (y != top_class).astype(int)
    else:
        y_binary = y.astype(int)

    def _run_batch():
        print(f"[{job_id}] Starting batch: {len(seeds)} seeds × {len(epsilon_grid)} epsilons × "
              f"{len(mechanisms)} mechs × {len(classifier_types)} classifiers = "
              f"{len(seeds)*len(epsilon_grid)*len(mechanisms)*len(classifier_types)} experiments")

        for seed in seeds:
            for eps in epsilon_grid:
                for mech in mechanisms:
                    for clf_type in classifier_types:
                        try:
                            out = privacy_engine.run_pipeline(
                                X=X, y=y_binary,
                                epsilon=eps,
                                mechanism=mech,
                                classifier_type=clf_type,
                                seed=seed,
                                n_shadow=n_shadow,
                            )
                            exp_id = f"EXP-{uuid.uuid4().hex[:6].upper()}"
                            model_path = os.path.join(UPLOAD_DIR, f"{exp_id}_model.json")
                            if "modelArtifacts" in out:
                                with open(model_path, "w") as f:
                                    json.dump(out["modelArtifacts"], f, indent=2)

                            # Build a minimal req for DB save
                            from schemas import PrivacyProcessRequest as PPR
                            dummy_req = PPR(
                                epsilon=eps,
                                mechanism=mech,
                                classifier_type=clf_type,
                                seed=seed,
                                n_shadow=n_shadow,
                            )
                            eps_acc = privacy_engine.compute_epsilon_accounting(
                                eps, out.get("feature_count", 1)
                            ) if mech != MECHANISM_NONE else {}
                            out["epsilon_accounting"] = eps_acc
                            _save_experiment_to_db(exp_id, dummy_req, out, model_path, None)
                            print(f"[{job_id}] Saved {exp_id}: eps={eps}, mech={mech}, clf={clf_type}, seed={seed}")
                        except Exception as e:
                            print(f"[{job_id}] Error at eps={eps}, mech={mech}, clf={clf_type}, seed={seed}: {e}")

        print(f"[{job_id}] Batch complete.")

    background_tasks.add_task(_run_batch)

    return {
        "job_id": job_id,
        "message": f"Batch experiment started in background ({req.mode} mode).",
        "config": {
            "mode": req.mode,
            "seeds": seeds,
            "epsilon_grid": epsilon_grid,
            "mechanisms": mechanisms,
            "classifier_types": classifier_types,
            "n_shadow": n_shadow,
            "total_experiments": len(seeds) * len(epsilon_grid) * len(mechanisms) * len(classifier_types),
        },
    }


@app.get("/api/download-dataset/{experiment_id}")
async def download_perturbed_dataset(experiment_id: str):
    """Download the privacy-perturbed dataset CSV for an experiment."""
    csv_path = os.path.join(UPLOAD_DIR, f"{experiment_id}_perturbed.csv")

    if not os.path.exists(csv_path):
        with SessionLocal() as db:
            rec = db.query(Experiment).filter(Experiment.id == experiment_id).first()
            if rec and getattr(rec, "export_path", None):
                export_csv = os.path.join(rec.export_path, "noised_data.csv")
                if os.path.exists(export_csv):
                    csv_path = export_csv

    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No persisted noised matrix exists for this experiment.",
        )

    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=f"{experiment_id}_noised_data.csv",
        headers={"Content-Disposition": f'attachment; filename="{experiment_id}_noised_data.csv"'},
    )

@app.get("/api/experiments/{experiment_id}/noised-data")
async def get_noised_data(experiment_id: str):
    """Retrieve the DP-noised matrix if it was explicitly exported."""
    with SessionLocal() as db:
        rec = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not rec or not getattr(rec, "export_path", None):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No persisted noised matrix exists for this experiment."
            )
        
        export_csv = os.path.join(rec.export_path, "noised_data.csv")
        if not os.path.exists(export_csv):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No persisted noised matrix exists for this experiment."
            )
        
        return FileResponse(
            path=export_csv,
            media_type="text/csv",
            filename=f"{experiment_id}_noised_data.csv",
            headers={"Content-Disposition": f'attachment; filename="{experiment_id}_noised_data.csv"'},
        )


@app.get("/api/download-model/{experiment_id}")
async def download_model_weights(experiment_id: str):
    """Download trained model weights as JSON."""
    json_path = os.path.join(UPLOAD_DIR, f"{experiment_id}_model.json")

    if not os.path.exists(json_path):
        with SessionLocal() as db:
            rec = db.query(Experiment).filter(Experiment.id == experiment_id).first()
            if rec and rec.model_path and os.path.exists(rec.model_path):
                json_path = rec.model_path

    if not os.path.exists(json_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model weights for experiment {experiment_id} not found.",
        )

    return FileResponse(
        path=json_path,
        media_type="application/json",
        filename=f"{experiment_id}_model_weights.json",
        headers={"Content-Disposition": f'attachment; filename="{experiment_id}_model_weights.json"'},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
