import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from privacy_engine import PrivacyEngine, MECHANISM_LAPLACE_CLIPPED, MECHANISM_LAPLACE, MECHANISM_NONE, CLIPPING_BOUND, DEFAULT_SENSITIVITY
import numpy as np

def test_noise_calibration():
    run_tests()

def run_tests():
    engine = PrivacyEngine(seed=42)
    
    # 1. epsilon_feature = epsilon_record / 29
    acc = engine.compute_epsilon_accounting(1.0, 29)
    assert np.isclose(acc['epsilon_feature'], 1.0 / 29), f"Expected {1.0/29}, got {acc['epsilon_feature']}"
    
    # 2. Scale with C=3 and epsilon=1 is 174
    # b = 6.0 / (1.0 / 29) = 6.0 * 29 = 174
    scale = engine.calculate_noise_scale(acc['epsilon_feature'], DEFAULT_SENSITIVITY, MECHANISM_LAPLACE_CLIPPED)
    assert np.isclose(scale, 174.0), f"Expected 174.0, got {scale}"
    
    # 3. For epsilon values 0.1, 1, 5, and 10, scales are 1740, 174, 34.8, 17.4
    epsilons = [0.1, 1.0, 5.0, 10.0]
    expected_scales = [1740.0, 174.0, 34.8, 17.4]
    for eps, expected in zip(epsilons, expected_scales):
        acc = engine.compute_epsilon_accounting(eps, 29)
        scale = engine.calculate_noise_scale(acc['epsilon_feature'], DEFAULT_SENSITIVITY, MECHANISM_LAPLACE_CLIPPED)
        assert np.isclose(scale, expected), f"For eps {eps}, expected {expected}, got {scale}"
        
    # 4. Clipped mechanism respects output bounds
    # Input has extreme values to test clipping before and after
    X_test = np.array([[10.0, -10.0], [0.5, 0.5]])
    acc = engine.compute_epsilon_accounting(1.0, 2) # just 2 features for this isolated test
    X_clipped_pert = engine.inject_dp_noise(X_test, acc['epsilon_feature'], DEFAULT_SENSITIVITY, MECHANISM_LAPLACE_CLIPPED)
    assert np.all(X_clipped_pert <= CLIPPING_BOUND) and np.all(X_clipped_pert >= -CLIPPING_BOUND), "Clipped mechanism failed output bounds test"
    
    # 5. Plain mechanism is not artificially clipped
    X_plain_pert = engine.inject_dp_noise(X_test, acc['epsilon_feature'], DEFAULT_SENSITIVITY, MECHANISM_LAPLACE)
    # The unbounded Laplace should exceed bounds for some values, especially with 10.0 and large scale
    assert np.any(X_plain_pert > CLIPPING_BOUND) or np.any(X_plain_pert < -CLIPPING_BOUND), "Plain mechanism appears artificially clipped"
    
    # 6. Non-private mechanism returns unchanged copy
    X_none = engine.inject_dp_noise(X_test, acc['epsilon_feature'], DEFAULT_SENSITIVITY, MECHANISM_NONE)
    assert np.array_equal(X_none, X_test), "Non-private mechanism modified input"
    
    # 7. Reproducibility
    X_rep1 = engine.inject_dp_noise(X_test, acc['epsilon_feature'], DEFAULT_SENSITIVITY, MECHANISM_LAPLACE_CLIPPED, seed=123)
    X_rep2 = engine.inject_dp_noise(X_test, acc['epsilon_feature'], DEFAULT_SENSITIVITY, MECHANISM_LAPLACE_CLIPPED, seed=123)
    assert np.array_equal(X_rep1, X_rep2), "Reproducibility failed"
    
    # 8. Shadow model perturb function
    fn = engine.make_dp_perturb_fn(acc['epsilon_feature'], DEFAULT_SENSITIVITY, MECHANISM_LAPLACE_CLIPPED, seed=123)
    X_fn = fn(X_test)
    assert np.array_equal(X_fn, X_rep1), "Shadow model perturb fn doesn't match direct injection"
    
    print('All calibration and mechanism tests passed!')

if __name__ == '__main__':
    run_tests()
