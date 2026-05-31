import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import function instead of streamlit which breaks without context
from app import compute_optimized_correlation_matrix

def run_test():
    print("Testing Módulo 4...")
    
    # Fake prices
    df = pd.DataFrame({
        'AAPL': [100, 101, 102, 103],
        'MSFT': [200, 199, 198, 197],
        'GOOG': [50, 50, 50, 50]
    })
    
    json_payload = df.to_json(orient='split')
    corr_matrix, labels = compute_optimized_correlation_matrix(json_payload)
    
    print("Labels:", labels)
    print("Correlation:\n", corr_matrix)
    
    assert len(labels) == 3
    # AAPL and MSFT are perfectly negatively correlated (-1)
    # The upper triangle should be nan
    assert np.isnan(corr_matrix[0, 1])
    # The diagonal should be 1
    assert abs(corr_matrix[0, 0] - 1.0) < 1e-5
    
    print("Módulo 4 test superado.")

if __name__ == '__main__':
    run_test()
