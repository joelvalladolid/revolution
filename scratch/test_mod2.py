import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lab.backtest_combined import compute_core_metrics_vectorized, compute_advanced_risk_metrics

def run_test():
    print("Testing Módulo 2...")
    pnl = np.array([0.05, -0.02, 0.03, -0.01, 0.04])
    core = compute_core_metrics_vectorized(pnl)
    print("Core:", core)
    
    rets = np.array([0.01, 0.02, -0.01, 0.01, -0.02, 0.03, 0.01])
    risk = compute_advanced_risk_metrics(rets)
    print("Risk:", risk)

    assert core['n_signals'] == 5
    assert core['win_rate'] == 0.6
    print("Módulo 2 test superado.")

if __name__ == '__main__':
    run_test()
