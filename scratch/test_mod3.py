import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lab.sltp_engine import vectorized_atr_backtest_engine

def run_test():
    print("Testing Módulo 3 (Numba)...")
    
    dates = pd.date_range('2020-01-01', periods=5)
    df = pd.DataFrame({
        'High': [100, 110, 120, 105, 100],
        'Low':  [90, 95, 100, 90, 85],
        'EntryPrice': [100, np.nan, np.nan, np.nan, np.nan],
        'TP_Price': [115, np.nan, np.nan, np.nan, np.nan],
        'SL_Price': [88, np.nan, np.nan, np.nan, np.nan]
    }, index=dates)
    
    res = vectorized_atr_backtest_engine(df)
    
    # El TP de 115 se alcanza en el tercer día (High = 120)
    print("PnL:", res['PnL'].values)
    print("ExitDate:", res['ExitDate'].values)
    
    assert res['PnL'].iloc[0] == 15.0 # (115 - 100)
    print("Módulo 3 test superado.")

if __name__ == '__main__':
    run_test()
