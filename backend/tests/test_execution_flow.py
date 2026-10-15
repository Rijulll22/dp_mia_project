import sys
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from privacy_engine import PrivacyEngine, MECHANISM_LAPLACE_CLIPPED, MECHANISM_LAPLACE, MECHANISM_NONE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def test_feature_perturbation():
    engine = PrivacyEngine(seed=42)
    # create dummy df with V1, V2, Amount (40 rows so 4-way partition succeeds)
    n_dummy = 40
    X = pd.DataFrame({
        "V1": np.linspace(0.1, 0.9, n_dummy),
        "V2": np.linspace(0.2, 0.8, n_dummy),
        "Amount": np.linspace(10, 100, n_dummy)
    })
    y = pd.Series([0, 1] * (n_dummy // 2))
    
    out = engine.run_pipeline(
        X=X,
        y=y,
        epsilon=1.0,
        mechanism=MECHANISM_LAPLACE_CLIPPED,
        classifier_type="logistic_regression",
        test_size=0.2,
        seed=42
    )
    
    assert out["feature_list"] == ["V1", "V2", "Amount"]
    assert out["perturbedDf"].shape[1] == 3, "Only V1, V2, and Amount should be perturbed."

def test_classifier_selection():
    engine = PrivacyEngine(seed=42)
    
    clf1 = engine._create_classifier("logistic_regression")
    assert isinstance(clf1, LogisticRegression)
    
    clf2 = engine._create_classifier("random_forest")
    assert isinstance(clf2, RandomForestClassifier)

def test_api_payload_preservation():
    from fastapi.testclient import TestClient
    from main import app
    from schemas import PrivacyProcessRequest
    
    client = TestClient(app)
    
    # We send a request to /api/process-privacy
    # We mock the engine.run_pipeline to avoid actual execution
    with patch('main.privacy_engine.run_pipeline') as mock_run:
        mock_run.return_value = {
            "metrics": {},
            "epsilon_accounting": {},
            "mechanism_type": "laplace_clipped",
            "classifier_type": "random_forest"
        }
        
        # Also mock tradeoff generation
        with patch('main.privacy_engine.generate_tradeoff_curve') as mock_curve:
            mock_curve.return_value = []
            
            payload = {
                "epsilon": 1.0,
                "delta": 0.00001,
                "sensitivity": 1.0,
                "mechanism": "laplace_clipped",
                "mechanism_type": "laplace_clipped",
                "classifier_type": "random_forest",
                "seed": 42
            }
            
            response = client.post("/api/process-privacy", json=payload)
            
            # Assert the API received and preserved the values
            assert response.status_code == 200
            
            # Verify the call to run_pipeline
            mock_run.assert_called_once()
            _, kwargs = mock_run.call_args
            assert kwargs["epsilon"] == 1.0
            assert kwargs["mechanism"] == "laplace_clipped"
            assert kwargs["classifier_type"] == "random_forest"
            assert kwargs["seed"] == 42
