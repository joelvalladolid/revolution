import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lab.kelly_optimizer import optimize_convex_kelly_allocation

def run_test():
    print("Testing Módulo 6...")
    expected_returns = np.array([0.10, 0.05, 0.08])
    covariance_matrix = np.array([
        [0.04, 0.01, 0.02],
        [0.01, 0.03, 0.01],
        [0.02, 0.01, 0.05]
    ])
    
    result = optimize_convex_kelly_allocation(
        expected_returns, 
        covariance_matrix, 
        kelly_fraction=0.5, 
        max_concentration_per_asset=0.50
    )
    
    print("Status:", result['status'])
    print("Weights:", result['optimal_weights'])
    
    assert result['status'] == 'SUCCESS'
    assert abs(np.sum(result['optimal_weights']) - 1.0) < 1e-5
    assert np.all(result['optimal_weights'] <= 0.5)
    print("Módulo 6 test superado.")

if __name__ == '__main__':
    run_test()
