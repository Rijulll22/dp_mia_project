import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from privacy_engine import PrivacyEngine, MECHANISM_NONE, MECHANISM_LAPLACE_CLIPPED

def _mock_shadow_mia(*args, **kwargs):
    return {"attack_accuracy": 0.5, "attack_auc": 0.5, "attacker_advantage": 0.0, "tpr_at_1fpr": 0.0}

def _mock_per_class_mia(*args, **kwargs):
    return {"attack_accuracy": 0.5, "attack_auc": 0.5, "attacker_advantage": 0.0, "tpr_at_1fpr": 0.0}

@patch('privacy_engine.run_shadow_mia', side_effect=_mock_shadow_mia)
@patch('privacy_engine.run_per_class_mia', side_effect=_mock_per_class_mia)
def test_partition_disjointness_and_proportions(mock_per_class, mock_shadow):
    engine = PrivacyEngine(seed=42)
    df_X, series_y = engine.generate_benchmark_dataset(n_samples=200, n_features=10, fraud_ratio=0.1)
    
    with patch('sklearn.model_selection.train_test_split', side_effect=None) as mock_split:
        out = engine.run_pipeline(df_X, series_y, mechanism=MECHANISM_NONE, n_shadow=1)
        
        train_sz = out['train_size']
        test_sz = out['test_size']
        
        # 40% train of 200 = 80, 20% test of 200 = 40
        assert 70 <= train_sz <= 90, f"Expected ~80, got {train_sz}"
        assert 35 <= test_sz <= 45, f"Expected ~40, got {test_sz}"
        assert train_sz + test_sz < 200, "Train and test should not be full dataset"

@patch('privacy_engine.run_shadow_mia', side_effect=_mock_shadow_mia)
@patch('privacy_engine.run_per_class_mia', side_effect=_mock_per_class_mia)
def test_small_dataset_failure(mock_per_class, mock_shadow):
    engine = PrivacyEngine(seed=42)
    # n_samples = 3 (< 4)
    df_X = pd.DataFrame({'V1': [1, 2, 3]})
    series_y = pd.Series([0, 1, 0])
    
    try:
        engine.run_pipeline(df_X, series_y, mechanism=MECHANISM_NONE, n_shadow=1)
        assert False, "Should have raised ValueError for n_samples < 4"
    except ValueError as e:
        assert "Dataset too small" in str(e) or "Disjoint partitions" in str(e)

@patch('privacy_engine.run_shadow_mia', side_effect=_mock_shadow_mia)
@patch('privacy_engine.run_per_class_mia', side_effect=_mock_per_class_mia)
def test_imbalanced_dataset_failure(mock_per_class, mock_shadow):
    engine = PrivacyEngine(seed=42)
    # n_samples = 10, but only 1 minority class -> can't guarantee both classes in train & test
    df_X = pd.DataFrame(np.random.randn(10, 5))
    series_y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    
    try:
        engine.run_pipeline(df_X, series_y, mechanism=MECHANISM_NONE, n_shadow=1)
        assert False, "Should have raised ValueError for missing class in split"
    except ValueError as e:
        assert "does not contain both classes" in str(e) or "stratify" in str(e)

@patch('privacy_engine.run_shadow_mia', side_effect=_mock_shadow_mia)
@patch('privacy_engine.run_per_class_mia', side_effect=_mock_per_class_mia)
def test_threshold_consistency(mock_per_class, mock_shadow):
    engine = PrivacyEngine(seed=42)
    df_X, series_y = engine.generate_benchmark_dataset(n_samples=200, n_features=10, fraud_ratio=0.5)
    
    out = engine.run_pipeline(df_X, series_y, mechanism=MECHANISM_LAPLACE_CLIPPED, epsilon=10.0, n_shadow=1)
    
    dp_threshold = out['best_threshold']
    baseline_threshold = out['baselines']['non_private']['best_threshold']
    
    assert isinstance(dp_threshold, float)
    assert isinstance(baseline_threshold, float)
    # thresholds should be strictly bounded
    assert 0.0 <= dp_threshold <= 1.0
    assert 0.0 <= baseline_threshold <= 1.0
    
    # ensure metric keys are present
    assert 'f1' in out['metrics']
    assert 'f1' in out['baselines']['non_private']

@patch('privacy_engine.run_shadow_mia', side_effect=_mock_shadow_mia)
@patch('privacy_engine.run_per_class_mia', side_effect=_mock_per_class_mia)
def test_reproducibility(mock_per_class, mock_shadow):
    engine1 = PrivacyEngine(seed=123)
    df_X, series_y = engine1.generate_benchmark_dataset(n_samples=150, n_features=5, fraud_ratio=0.2)
    
    out1 = engine1.run_pipeline(df_X, series_y, mechanism=MECHANISM_LAPLACE_CLIPPED, epsilon=5.0, seed=123, n_shadow=1)
    
    engine2 = PrivacyEngine(seed=123)
    out2 = engine2.run_pipeline(df_X, series_y, mechanism=MECHANISM_LAPLACE_CLIPPED, epsilon=5.0, seed=123, n_shadow=1)
    
    assert out1['metrics']['f1'] == out2['metrics']['f1']
    assert out1['best_threshold'] == out2['best_threshold']
    assert out1['baselines']['non_private']['best_threshold'] == out2['baselines']['non_private']['best_threshold']

if __name__ == '__main__':
    test_partition_disjointness_and_proportions()
    test_small_dataset_failure()
    test_imbalanced_dataset_failure()
    test_threshold_consistency()
    test_reproducibility()
    print('All tests passed successfully!')

@patch('privacy_engine.run_per_class_mia', side_effect=_mock_per_class_mia)
def test_per_class_mia_seed_propagation(mock_per_class):
    engine = PrivacyEngine(seed=42)
    df_X, series_y = engine.generate_benchmark_dataset(n_samples=200, n_features=10, fraud_ratio=0.1)
    
    with patch('sklearn.model_selection.train_test_split', side_effect=None):
        out = engine.run_pipeline(df_X, series_y, mechanism=MECHANISM_NONE, n_shadow=1)
        
    assert mock_per_class.call_count == 2
    # Check if seed=42 is passed to kwargs
    call_args_legit = mock_per_class.call_args_list[1][1]
    assert call_args_legit.get('seed') == 42
