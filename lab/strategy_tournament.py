import sys
import os
import time
import argparse
import pandas as pd
import numpy as np
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lab.parallel_engine import execute_parallel_optimization_grid
import lab.parallel_engine as parallel_engine
from lab.indicators import rsi
from lab.sltp_engine import vectorized_atr_backtest_engine
from lab.backtest_combined import compute_core_metrics_vectorized, PERIODS
from lab.score_engine import MasterScoreNormalizer
from lab.kelly_optimizer import optimize_convex_kelly_allocation
import yfinance as yf

def _real_evaluation_task(params):
    ticker = params.get('ticker')
    try:
        shared_df = parallel_engine._SHARED_HISTORICAL_DATA
        if isinstance(shared_df.columns, pd.MultiIndex):
            # yfinance MultiIndex is usually (PriceType, Ticker)
            df = shared_df.xs(ticker, level=1, axis=1).copy()
        else:
            # If it's a single ticker, yfinance returns flat columns
            df = shared_df.copy()
            
        if len(df) < 30:
            return {'ticker': ticker, 'params': params, 'status': 'FAILED', 'error': 'Not enough data'}
            
        df['RSI'] = rsi(df, period=14)
        rsi_th = params.get('rsi_threshold', 30)
        signal = df['RSI'] < rsi_th
        
        df['EntryPrice'] = np.where(signal.shift(1).fillna(False), df['Open'], np.nan)
        
        sl_pct = params.get('sl_pct', 0.05)
        tp_pct = params.get('tp_pct', 0.10)
        
        df['SL_Price'] = df['EntryPrice'] * (1 - sl_pct)
        df['TP_Price'] = df['EntryPrice'] * (1 + tp_pct)
        
        df_result = vectorized_atr_backtest_engine(df)
        
        df_trades = df_result.dropna(subset=['PnL']).copy()
        if len(df_trades) == 0:
            metrics = compute_core_metrics_vectorized(np.array([]))
            return {'ticker': ticker, 'params': params, 'status': 'SUCCESS', 'metrics': metrics, 'daily_returns': {}}
            
        df_trades['ReturnPct'] = df_trades['PnL'] / df_trades['EntryPrice']
        metrics = compute_core_metrics_vectorized(df_trades['ReturnPct'].values)
        
        daily_returns = df_trades.groupby(df_trades['ExitDate'].dt.date)['ReturnPct'].sum()
        
        return {
            'ticker': ticker,
            'params': params,
            'status': 'SUCCESS',
            'metrics': metrics,
            'daily_returns': daily_returns.to_dict()
        }
        
    except Exception as exc:
        return {
            'ticker': ticker,
            'params': params,
            'status': 'FAILED',
            'error': str(exc) + " | " + traceback.format_exc()
        }

def run_tournament(tickers, period, max_combos, output_file):
    start, end = PERIODS.get(period, ("2022-01-01", "2022-12-31"))
    print(f"Descargando datos para {len(tickers)} tickers desde {start} a {end}...")
    df_history = yf.download(tickers, start=start, end=end, progress=False)
    
    if df_history.empty:
        print("Error: No se pudieron descargar datos.")
        return
             
    # Generar exactamente 18 combinaciones (<= 20)
    param_space = {
        'rsi_threshold': [20, 30],
        'sl_pct': [0.03, 0.05, 0.08],
        'tp_pct': [0.05, 0.10, 0.15]
    }
    
    print(f"Iniciando cluster paralelo...")
    start_t = time.time()
    
    results = execute_parallel_optimization_grid(
        historical_df=df_history,
        universe_tickers=tickers,
        param_space=param_space,
        max_cores=max(1, os.cpu_count() - 1),
        eval_func=_real_evaluation_task
    )
    
    parallel_time = time.time() - start_t
    
    successful_results = [r for r in results if r['status'] == 'SUCCESS']
    failed_results = [r for r in results if r['status'] == 'FAILED']
    
    print(f"Paralelo completado en {parallel_time:.2f}s. Tareas exitosas: {len(successful_results)}, Fallidas: {len(failed_results)}")
    
    if failed_results:
        print(f"Sample error: {failed_results[0].get('error')}")
    
    # Calcular boundaries para ScoreEngine
    all_metrics = [r['metrics'] for r in successful_results]
    boundaries = {}
    if all_metrics:
        for k in all_metrics[0].keys():
            vals = [m[k] for m in all_metrics if not np.isnan(m[k]) and not np.isinf(m[k])]
            if vals:
                boundaries[k] = {'min': min(vals), 'max': max(vals)}
    
    normalizer = MasterScoreNormalizer()
    for r in successful_results:
        score_res = normalizer.compute_strategy_score(r['metrics'], boundaries)
        r['final_score'] = score_res['final_score']
    
    # Seleccionar Top 3
    top_3 = sorted(successful_results, key=lambda x: x.get('final_score', 0.0), reverse=True)[:3]
    
    # Preparar inputs para Kelly Optimizer
    if len(top_3) > 0:
        daily_returns_list = []
        expected_returns = []
        for i, strat in enumerate(top_3):
            dr = pd.Series(strat['daily_returns'])
            dr.name = f"Strat_{i}"
            daily_returns_list.append(dr)
            expected_returns.append(strat['metrics']['ev'])
            
        df_returns = pd.concat(daily_returns_list, axis=1).fillna(0.0)
        cov_matrix = df_returns.cov().values
        cov_matrix = np.nan_to_num(cov_matrix) + np.eye(len(cov_matrix)) * 1e-6
        exp_ret_arr = np.array(expected_returns)
        
        kelly_res = optimize_convex_kelly_allocation(exp_ret_arr, cov_matrix, max_concentration_per_asset=1.0)
        optimal_weights = kelly_res.get('optimal_weights', [])
    else:
        optimal_weights = []
        
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== REPORTE DE TORNEO INTEGRADO ===\n")
        f.write(f"Tiempo paralelo: {parallel_time:.2f} segundos\n")
        f.write(f"Tareas exitosas: {len(successful_results)}\n")
        f.write("\n=== TOP 3 ESTRATEGIAS ===\n")
        for i, strat in enumerate(top_3):
            m = strat['metrics']
            f.write(f"Puesto {i+1}: {strat['ticker']} | {strat['params']}\n")
            f.write(f"   Score: {strat['final_score']:.2f}/100\n")
            f.write(f"   Win Rate: {m.get('win_rate', 0)*100:.2f}%\n")
            f.write(f"   EV (Porcentaje por trade): {m.get('ev', 0)*100:.2f}%\n")
            f.write(f"   Trades Totales: {m.get('n_signals', 0)}\n\n")
            
        f.write("=== KELLY OPTIMIZER ===\n")
        f.write(f"Pesos resultantes: {optimal_weights}\n")
        sum_w = sum(optimal_weights) if len(optimal_weights) > 0 else 0
        f.write(f"Suma total: {sum_w:.6f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs='+', required=True)
    parser.add_argument("--period", type=str, required=True)
    parser.add_argument("--max-combos", type=int, default=20)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    run_tournament(args.tickers, args.period, args.max_combos, args.output)
