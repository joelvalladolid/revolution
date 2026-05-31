import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lab.parallel_engine import execute_parallel_optimization_grid

def run_test():
    print("Testing Módulo 1...")
    
    # Create fake df
    dates = pd.date_range('2020-01-01', periods=100)
    data_aapl = np.random.randn(100)
    data_msft = np.random.randn(100)
    df = pd.DataFrame({'AAPL': data_aapl, 'MSFT': data_msft}, index=dates)
    
    param_space = {
        'fast_ema': [10, 20],
        'slow_ema': [50]
    }
    universe = ['AAPL', 'MSFT']
    
    # run parallel
    results = execute_parallel_optimization_grid(df, universe, param_space, max_cores=2)
    
    print("Results count:", len(results))
    print("First result:", results[0])
    
    assert len(results) == 4, "Debería haber 4 combinaciones (2 tickers * 2 params)"
    assert all(r['status'] == 'SUCCESS' for r in results)
    print("Módulo 1 test superado.")

if __name__ == '__main__':
    run_test()
