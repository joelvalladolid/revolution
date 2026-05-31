import multiprocessing as mp
import pandas as pd
import numpy as np
import logging
import gc
from typing import List, Dict, Any, Tuple
from itertools import product

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_SHARED_HISTORICAL_DATA: pd.DataFrame = pd.DataFrame()

def _pool_worker_initializer(shared_df: pd.DataFrame) -> None:
    global _SHARED_HISTORICAL_DATA
    _SHARED_HISTORICAL_DATA = shared_df
    gc.disable()

def _strategy_evaluation_task(params: Dict[str, Any]) -> Dict[str, Any]:
    global _SHARED_HISTORICAL_DATA
    
    ticker = params.get('ticker')
    try:
        # Check if ticker is in columns
        # To handle both multi-index and flat columns safely
        is_multi = isinstance(_SHARED_HISTORICAL_DATA.columns, pd.MultiIndex)
        if is_multi:
            # Assuming format like ('AAPL', 'Close')
            cols = _SHARED_HISTORICAL_DATA.columns.get_level_values(0).unique()
            if ticker in cols:
                ticker_data = _SHARED_HISTORICAL_DATA[ticker]
            else:
                raise KeyError(f"Identificador bursátil {ticker} ausente en el bloque de memoria.")
        else:
            if ticker in _SHARED_HISTORICAL_DATA.columns:
                ticker_data = _SHARED_HISTORICAL_DATA[ticker]
            else:
                raise KeyError(f"Identificador bursátil {ticker} ausente en el bloque de memoria.")
        
        return {
            'ticker': ticker,
            'params': params,
            'status': 'SUCCESS',
            'ev': float(np.random.rand()) # Placeholder 
        }
    except Exception as exc:
        logger.error(f"Fallo de segmentación/ejecución en {params}: {str(exc)}")
        return {
            'ticker': ticker,
            'params': params,
            'status': 'FAILED',
            'error': str(exc)
        }

def execute_parallel_optimization_grid(
    historical_df: pd.DataFrame, 
    universe_tickers: List[str], 
    param_space: Dict[str, List[Any]],
    max_cores: int = max(1, mp.cpu_count() - 1),
    eval_func=None
) -> List[Dict]:
    keys, values = zip(*param_space.items())
    base_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    tasks = []
    for ticker in universe_tickers:
        for combo in base_combinations:
            task_instance = combo.copy()
            task_instance['ticker'] = ticker
            tasks.append(task_instance)
            
    total_tasks = len(tasks)
    logger.info(f"Orquestando clúster: {total_tasks} nodos de cálculo sobre {max_cores} núcleos lógicos.")
    
    optimal_chunk = max(1, int(total_tasks / (max_cores * 4)))
    
    func_to_use = eval_func if eval_func is not None else _strategy_evaluation_task
    
    results = []
    with mp.Pool(
        processes=max_cores, 
        initializer=_pool_worker_initializer, 
        initargs=(historical_df,)
    ) as pool:
        for res in pool.imap_unordered(func_to_use, tasks, chunksize=optimal_chunk):
            results.append(res)
            
    return results
