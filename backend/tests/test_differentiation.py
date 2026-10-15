import sys
import os
import numpy as np

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from privacy_engine import PrivacyEngine, MECHANISM_LAPLACE_CLIPPED, MECHANISM_LAPLACE, MECHANISM_NONE

def test_non_private_unchanged():
    engine = PrivacyEngine(seed=42)
    # 5 rows, 3 columns scaled in [0, 1]
    X_input = np.array([
        [0.1, 0.5, 0.9],
        [0.2, 0.6, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5]
    ])
    
    # Epsilon = 1.0, feature_count = 3 => epsilon_feature = 1.0/3
    X_output = engine.inject_dp_noise(
        X_input, 
        epsilon_feature=1.0/3, 
        sensitivity=1.0, 
        mechanism=MECHANISM_NONE, 
        seed=42
    )
    
    assert np.array_equal(X_input, X_output), "non_private should not modify the input matrix"

def test_plain_laplace_perturbed_and_unclipped():
    engine = PrivacyEngine(seed=42)
    X_input = np.array([
        [0.1, 0.5, 0.9],
        [0.2, 0.6, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5]
    ])
    
    # Epsilon is small to generate large noise
    epsilon_feature = 0.01 / 3
    X_output = engine.inject_dp_noise(
        X_input, 
        epsilon_feature=epsilon_feature, 
        sensitivity=1.0, 
        mechanism=MECHANISM_LAPLACE, 
        seed=42
    )
    
    assert not np.array_equal(X_input, X_output), "plain_laplace should perturb the input matrix"
    
    # Assert that there are values outside [0, 1] due to large noise on boundaries
    assert np.any(X_output < 0) or np.any(X_output > 1), "plain_laplace should NOT be clipped to [0,1]"

def test_laplace_clipped_perturbed_and_clipped():
    engine = PrivacyEngine(seed=42)
    X_input = np.array([
        [0.1, 0.5, 0.9],
        [0.2, 0.6, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5]
    ])
    
    # Epsilon is small to generate large noise that hits the boundaries
    epsilon_feature = 0.01 / 3
    X_output = engine.inject_dp_noise(
        X_input, 
        epsilon_feature=epsilon_feature, 
        sensitivity=1.0, 
        mechanism=MECHANISM_LAPLACE_CLIPPED, 
        seed=42
    )
    
    assert not np.array_equal(X_input, X_output), "laplace_clipped should perturb the input matrix"
    
    # Assert clipping
    assert np.all(X_output >= 0.0) and np.all(X_output <= 1.0), "laplace_clipped should strictly bound output to [0,1]"

def test_epsilon_scaling_differentiates_output():
    engine = PrivacyEngine(seed=42)
    X_input = np.array([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    ])
    
    eps_01 = 0.1 / 3
    eps_1 = 1.0 / 3
    eps_10 = 10.0 / 3
    
    X_out_01 = engine.inject_dp_noise(X_input, epsilon_feature=eps_01, sensitivity=1.0, mechanism=MECHANISM_LAPLACE, seed=42)
    X_out_1 = engine.inject_dp_noise(X_input, epsilon_feature=eps_1, sensitivity=1.0, mechanism=MECHANISM_LAPLACE, seed=42)
    X_out_10 = engine.inject_dp_noise(X_input, epsilon_feature=eps_10, sensitivity=1.0, mechanism=MECHANISM_LAPLACE, seed=42)
    
    assert not np.array_equal(X_out_01, X_out_1), "Different epsilons should produce different noisy outputs"
    assert not np.array_equal(X_out_1, X_out_10), "Different epsilons should produce different noisy outputs"
    
    diff_01 = np.abs(X_out_01 - 0.5).mean()
    diff_1 = np.abs(X_out_1 - 0.5).mean()
    diff_10 = np.abs(X_out_10 - 0.5).mean()
    
    assert diff_10 < diff_1 < diff_01, "Higher epsilon should result in lower average perturbation calibration"

if __name__ == "__main__":
    test_non_private_unchanged()
    test_plain_laplace_perturbed_and_unclipped()
    test_laplace_clipped_perturbed_and_clipped()
    test_epsilon_scaling_differentiates_output()
    print("All differentiation tests passed successfully.")
