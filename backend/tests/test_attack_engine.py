import pytest
import numpy as np
from unittest.mock import Mock
from attack_engine import (
    _get_tpr_at_fpr_threshold,
    _extract_model_confidence_features,
    run_shadow_mia,
    run_per_class_mia
)

def test_tpr_at_fpr_threshold():
    # FPRs sorted naturally by ROC curve, TPRs corresponding
    # Let's say FPR is [0.0, 0.0005, 0.005, 0.01, 0.05, 1.0]
    # TPR is         [0.0, 0.1,    0.2,   0.3,  0.8,  1.0]
    fpr_arr = np.array([0.0, 0.0005, 0.005, 0.01, 0.05, 1.0])
    tpr_arr = np.array([0.0, 0.1,    0.2,   0.3,  0.8,  1.0])
    
    tpr_01 = _get_tpr_at_fpr_threshold(fpr_arr, tpr_arr, 0.001)
    assert np.isclose(tpr_01, 0.1)
    
    tpr_1 = _get_tpr_at_fpr_threshold(fpr_arr, tpr_arr, 0.01)
    assert np.isclose(tpr_1, 0.3)
    
    # If no FPR is <= target, it should be 0.0 (though roc curve always starts with 0.0)
    # But let's say target is -1
    tpr_none = _get_tpr_at_fpr_threshold(fpr_arr, tpr_arr, -1.0)
    assert np.isclose(tpr_none, 0.0)

def test_extract_model_confidence_features():
    # Mock model
    class MockModel:
        def predict_proba(self, X):
            return X # Passing probabilities directly as X for testing
    
    model = MockModel()
    # 2 classes: 0 and 1
    # Probabilities for 2 records:
    # Rec 1: [0.1, 0.9], True label: 1
    # Rec 2: [0.9999999999999, 0.0000000000001], True label: 1
    X = np.array([
        [0.1, 0.9],
        [0.9999999999999, 0.0000000000001]
    ])
    y = np.array([1, 1])
    
    feats = _extract_model_confidence_features(model, X, y)
    
    # Rec 1 features
    # true_class_prob = 0.9
    # max_prob = 0.9
    # entropy = -(0.1*log(0.1) + 0.9*log(0.9)) = 0.32508297
    # ce_loss = -log(0.9) = 0.1053605
    assert np.isclose(feats[0, 0], 0.9)
    assert np.isclose(feats[0, 1], 0.9)
    expected_entropy = -(0.1*np.log(0.1) + 0.9*np.log(0.9))
    assert np.isclose(feats[0, 2], expected_entropy)
    assert np.isclose(feats[0, 3], -np.log(0.9))
    
    # Rec 2 features (Numerical stability test)
    # true_class_prob = 1e-13 (clipped to eps_log 1e-12 by ce_loss)
    # max_prob = 0.9999999999999
    assert feats[1, 0] <= 1e-12
    # CE loss is -log(1e-12)
    assert np.isclose(feats[1, 3], -np.log(1e-12))
    # Entropy is practically 0
    assert feats[1, 2] >= 0.0
    assert feats[1, 2] < 1e-10

class MockTargetModel:
    def predict_proba(self, X):
        return np.ones((len(X), 2)) * 0.5
    def fit(self, X, y):
        pass

def test_shadow_mia_balancing_and_reproducibility():
    X_train = np.random.randn(100, 5)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(30, 5) # 30 non-members
    y_test = np.random.randint(0, 2, 30)
    X_shadow = np.random.randn(200, 5)
    y_shadow = np.random.randint(0, 2, 200)
    
    target_model = MockTargetModel()
    
    res1 = run_shadow_mia(
        target_model, X_train, y_train, X_test, y_test, X_shadow, y_shadow,
        n_shadow=2, seed=42
    )
    
    # Balancing check
    # 100 members, 30 non-members => should evaluate on all 30 non-members, and 30 downsampled members
    # "n_members_evaluated": int(len(target_member_feats)), which is 30
    assert res1['n_members_evaluated'] == 30
    assert res1['n_nonmembers_evaluated'] == 30
    assert res1.get('error') is None

def test_shadow_mia_insufficient_members():
    X_train = np.random.randn(20, 5) # 20 members
    y_train = np.random.randint(0, 2, 20)
    X_test = np.random.randn(30, 5) # 30 non-members
    y_test = np.random.randint(0, 2, 30)
    X_shadow = np.random.randn(200, 5)
    y_shadow = np.random.randint(0, 2, 200)
    
    target_model = MockTargetModel()
    
    res = run_shadow_mia(
        target_model, X_train, y_train, X_test, y_test, X_shadow, y_shadow,
        n_shadow=2, seed=42
    )
    assert res['attack_accuracy'] is None
    assert "Insufficient members" in res['error']

def test_shadow_mia_dp_perturb_fn():
    # Verify that the perturb function is actually called
    X_shadow = np.random.randn(50, 2)
    y_shadow = np.random.randint(0, 2, 50)
    
    called = []
    def mock_perturb(X):
        called.append(True)
        return X
    
    target_model = MockTargetModel()
    run_shadow_mia(
        target_model, np.random.randn(10,2), np.random.randint(0,2,10), 
        np.random.randn(10,2), np.random.randint(0,2,10), 
        X_shadow, y_shadow, n_shadow=1, seed=42, dp_perturb_fn=mock_perturb
    )
    
    assert len(called) > 0, "DP perturb function was not called on shadow training data"

def test_shadow_mia_insufficient_data():
    X_shadow = np.random.randn(2, 2)
    y_shadow = np.random.randint(0, 2, 2)
    target_model = MockTargetModel()
    
    res = run_shadow_mia(
        target_model, np.random.randn(10,2), np.random.randint(0,2,10), 
        np.random.randn(10,2), np.random.randint(0,2,10), 
        X_shadow, y_shadow, n_shadow=1, seed=42
    )
    
    assert res['error'] == "Insufficient shadow training data"
    assert res['attack_accuracy'] is None

def test_run_per_class_mia():
    target_model = MockTargetModel()
    attack_clf = MockTargetModel() # doesn't matter what it is as long as predict_proba works
    attack_clf.predict = lambda X: np.zeros(len(X))
    
    # 35 members of class 1, 30 non-members of class 1 -> balanced to 30 members, 30 non-members
    X_train = np.random.randn(50, 2)
    y_train = np.concatenate([np.ones(35), np.zeros(15)])
    
    X_test = np.random.randn(60, 2)
    y_test = np.concatenate([np.ones(30), np.zeros(30)])
    
    res = run_per_class_mia(
        target_model, X_train, y_train, X_test, y_test, attack_clf, class_label=1, min_samples=20
    )
    
    assert res['n_members'] == 35
    assert res['n_nonmembers'] == 30
    assert res['note'] is None if 'note' in res else True
    # Evaluated on 30 and 30 => attack_accuracy 0.5 because balanced and predict all 0
    assert np.isclose(res['attack_accuracy'], 0.5)
    assert np.isclose(res['attack_auc'], 0.5)

def test_run_per_class_mia_insufficient():
    target_model = MockTargetModel()
    attack_clf = MockTargetModel()
    
    # 10 members, 30 non-members -> should fail because members < nonmembers
    X_train = np.random.randn(10, 2)
    y_train = np.ones(10)
    
    X_test = np.random.randn(30, 2)
    y_test = np.ones(30)
    
    res = run_per_class_mia(
        target_model, X_train, y_train, X_test, y_test, attack_clf, class_label=1, min_samples=20
    )
    
    assert "Insufficient samples" in res['note']
    assert res['attack_accuracy'] is None

if __name__ == '__main__':
    test_tpr_at_fpr_threshold()
    test_extract_model_confidence_features()
    test_shadow_mia_balancing_and_reproducibility()
    test_shadow_mia_dp_perturb_fn()
    test_shadow_mia_insufficient_data()
    test_run_per_class_mia()
    test_run_per_class_mia_insufficient()
    print("All MIA tests passed successfully!")
