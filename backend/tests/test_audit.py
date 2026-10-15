import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from privacy_engine import PrivacyEngine, MECHANISM_LAPLACE_CLIPPED, DEFAULT_SENSITIVITY
from attack_engine import run_shadow_mia
import attack_engine

def test_noise_scale_formula():
    engine = PrivacyEngine(seed=42)
    # The prescribed calibration
    # epsilon_feature = epsilon_record / 29 = 1.0 / 29
    # sensitivity = 6.0
    epsilon_feature = 1.0 / 29.0
    scale = engine.calculate_noise_scale(epsilon_feature=epsilon_feature, sensitivity=6.0, mechanism=MECHANISM_LAPLACE_CLIPPED)
    assert np.isclose(scale, 174.0)

def test_genuine_auc_0_5():
    from sklearn.metrics import roc_auc_score
    # If a model outputs 0.5 for everything
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.5, 0.5, 0.5, 0.5])
    auc = roc_auc_score(y_true, y_scores)
    assert auc == 0.5
    
    # In attack_engine, this should be returned as attack_auc: 0.5
    # An unavailable result returns {"error": ...} or None for auc
    from sklearn.base import BaseEstimator, ClassifierMixin
    class MockAttackModel(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            pass
        def predict(self, X):
            return np.zeros(len(X))
        def predict_proba(self, X):
            return np.ones((len(X), 2)) * 0.5
            
    res = run_shadow_mia(
        target_model=MockAttackModel(),
        X_train_target=np.random.randn(20, 2),
        y_train_target=np.random.randint(0, 2, 20),
        X_test_target=np.random.randn(20, 2),
        y_test_target=np.random.randint(0, 2, 20),
        X_shadow=np.random.randn(100, 2),
        y_shadow=np.random.randint(0, 2, 100),
        n_shadow=2, seed=42
    )
    assert res['attack_auc'] == 0.5
    assert 'error' not in res or res['error'] is None

def test_pilot_imports():
    # Read pilot.py and assert it imports PrivacyEngine from local path
    pilot_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pilot.py")
    with open(pilot_path, "r") as f:
        content = f.read()
        assert "from privacy_engine import PrivacyEngine" in content
        assert "sys.path.insert(0" in content

if __name__ == "__main__":
    pytest.main([__file__])
