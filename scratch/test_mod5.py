import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lab.score_engine import MasterScoreNormalizer

def run_test():
    print("Testing Módulo 5...")
    normalizer = MasterScoreNormalizer()
    
    metrics = {
        'win_rate': 0.60,
        'ev': 0.5,
        'profit_factor': 1.5,
        'n_signals': 100,
        'p_value': 0.01,
        'max_consec_loss': 5,
        'edge_crash': 1.0,
        'edge_bull': 1.0,
        'edge_bear': 1.0
    }
    
    boundaries = {
        'win_rate': {'min': 0.40, 'max': 0.70},
        'ev': {'min': -1.0, 'max': 2.0}
    }
    
    result = normalizer.compute_strategy_score(metrics, boundaries)
    print("Result:", result)
    assert not result['is_disqualified'], "Estrategia buena fue descalificada"
    assert result['final_score'] > 0, "Score final debe ser positivo"
    assert 'All-Weather Bonus [Impulso +15%]' in result['audit_trail']
    print("Módulo 5 test superado.")

if __name__ == '__main__':
    run_test()
