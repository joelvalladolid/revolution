import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
import datetime
from lab.indicators import rsi, bollinger_pctB, stochastic_k, ema_discount
from lab.regime_detector import classify_regime
from lab.monte_carlo import simulate_price_paths

def run_historical_backtest(tickers, days_back=90, capital=10000, max_positions=10, state_dict=None):
    """
    Ejecuta un backtest histórico vectorial/diario.
    Se conecta al diccionario de estado (state_dict) para emitir progreso a la UI.
    """
    if state_dict is None:
        state_dict = {}

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days_back + 400) # +400 para asegurar 252 días de mercado
    
    state_dict['status'] = 'Descargando datos históricos (esto puede tardar 1-2 min)...'
    state_dict['progress'] = 0.0
    
    # 1. Bajar VIX
    try:
        tk = yf.Ticker("^VIX")
        vix_df = tk.history(start=start_date, end=end_date)
        if vix_df.empty:
            raise ValueError("Yahoo Finance no devolvió datos para el VIX (Posible Rate Limit).")
            
        vix_df = vix_df.dropna(subset=['Close'])
        vix_df.index = vix_df.index.tz_localize(None) # Igualar formato de fechas
        regimes = classify_regime(vix_df['Close'])
        vix_history = pd.DataFrame({'Close': vix_df['Close'], 'Regime': regimes}, index=vix_df.index)
    except Exception as e:
        logging.error(f"Error bajando VIX: {e}")
        state_dict['status'] = f'Error en VIX: {e}'
        state_dict['error'] = True
        return None

    # 2. Bajar S&P 500
    state_dict['status'] = 'Descargando precios de S&P 500...'
    try:
        # Descarga bulk (mucho mas veloz que individual)
        df_all = yf.download(tickers, start=start_date, end=end_date, progress=False, ignore_tz=True)
        # Si df_all tiene MultiIndex (yf behavior for multiple tickers), we reshape it
        if isinstance(df_all.columns, pd.MultiIndex):
            df_close = df_all['Close']
            df_open = df_all['Open']
            df_high = df_all['High']
            df_low = df_all['Low']
            df_vol = df_all['Volume']
        else:
            state_dict['status'] = 'Error de formato en Yahoo Finance.'
            return None
    except Exception as e:
        logging.error(f"Error bajando tickers: {e}")
        state_dict['status'] = f'Error: {e}'
        state_dict['error'] = True
        return None

    # Recortamos las fechas del VIX a los últimos N días operativos
    eval_dates = vix_history.index[-days_back:]
    
    portfolio = capital
    equity_curve = []
    trades = []
    
    # Pre-calcular indicadores (muy costoso hacerlo día por día, lo hacemos vectorizado por ticker)
    state_dict['status'] = 'Pre-calculando indicadores técnicos...'
    precomputed = {}
    total_t = len(df_close.columns)
    c = 0
    
    for t in df_close.columns:
        c += 1
        if c % 50 == 0:
            state_dict['progress'] = c / total_t * 0.3 # Primer 30% del progreso
            
        try:
            df_t = pd.DataFrame({
                'Open': df_open[t],
                'High': df_high[t],
                'Low': df_low[t],
                'Close': df_close[t],
                'Volume': df_vol[t]
            }).dropna()
            
            if len(df_t) < 252:
                continue
                
            df_t['EMA200_disc'] = ema_discount(df_t)
            df_t['BB_pctB'] = bollinger_pctB(df_t)
            df_t['Stoch_K'] = stochastic_k(df_t)
            df_t['RSI'] = rsi(df_t)
            df_t['Volume_Avg'] = df_t['Volume'].rolling(20).mean()
            
            precomputed[t] = df_t
        except Exception:
            pass

    state_dict['status'] = 'Simulando día a día...'
    
    # Variables globales para simular Monte Carlo rapido y Filtros
    total_dates = len(eval_dates)
    d_count = 0
    
    for i, date in enumerate(eval_dates):
        d_count += 1
        state_dict['progress'] = 0.3 + (d_count / total_dates * 0.7)
        state_dict['status'] = f'Simulando {date.strftime("%Y-%m-%d")}...'
        
        todays_cash = portfolio
        candidates = []
        
        for t, df in precomputed.items():
            if date not in df.index: continue
            
            idx = df.index.get_loc(date)
            # Necesitamos datos hasta el día anterior (T-1)
            if idx < 252: continue
            
            df_hist = df.iloc[:idx]
            yesterday_close = df_hist['Close'].iloc[-1]
            
            # Corrección: Como compramos en Open y vendemos en Close, usar retornos INTRADAY históricos.
            rets = (df_hist['Close'] / df_hist['Open'] - 1).dropna().tail(252)
            
            # Simulación rápida sin EVT (mirando a T)
            mc = simulate_price_paths(yesterday_close, rets, horizon_days=1, n_simulations=500, fast_mode=True)
            
            score = mc['expected_value']
            
            # Añadimos a candidatos (compramos al Open de HOY, vendemos al Close de HOY)
            # Para evitar 0 o divisiones
            open_price = df.iloc[idx]['Open']
            close_price = df.iloc[idx]['Close']
            if open_price > 0:
                candidates.append({
                    'ticker': t,
                    'buy_price': open_price,
                    'sell_price': close_price,
                    'score': score,
                    'prob_positive': mc['prob_positive'],
                    'expected_value': score,
                    'mean_pos': mc['mean_pos'],
                    'mean_neg': mc['mean_neg']
                })
                    
        # Filtro de calidad mínima: Esperanza Matemática positiva
        pos_candidates = [c for c in candidates if c['expected_value'] > 0]
        if len(pos_candidates) >= max_positions:
            candidates = pos_candidates
            
        # Comprar top 10
        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:max_positions]
        
        todays_profit = 0
        if candidates and todays_cash > 0:
            alloc_per_ticker = todays_cash / len(candidates)
            for c in candidates:
                shares = alloc_per_ticker / c['buy_price']
                profit_cash = shares * c['sell_price']
                todays_profit += profit_cash
                
                pnl = (c['sell_price'] - c['buy_price']) / c['buy_price']
                trades.append({
                    'ticker': c['ticker'],
                    'buy_date': date.strftime('%Y-%m-%d'),
                    'sell_date': date.strftime('%Y-%m-%d'),
                    'buy_price': c['buy_price'],
                    'sell_price': c['sell_price'],
                    'pnl_pct': pnl * 100,
                    'profit_usd': profit_cash - alloc_per_ticker,
                    'expected_value': c['expected_value'],
                    'prob_positive': c['prob_positive'],
                    'mean_pos': c['mean_pos'],
                    'mean_neg': c['mean_neg']
                })
            
            portfolio = todays_profit
            
        current_equity = portfolio
        equity_curve.append({'date': date.strftime('%Y-%m-%d'), 'equity': current_equity})

    # Finalizar y armar reporte
    state_dict['status'] = 'Generando reporte...'
    df_equity = pd.DataFrame(equity_curve)
    df_trades = pd.DataFrame(trades)
    
    total_return = ((current_equity / capital) - 1) * 100
    win_rate = (len(df_trades[df_trades['pnl_pct'] > 0]) / len(df_trades) * 100) if len(df_trades) > 0 else 0
    
    res = {
        'equity_curve': df_equity,
        'trades': df_trades,
        'initial_capital': capital,
        'final_equity': current_equity,
        'total_return_pct': total_return,
        'win_rate': win_rate,
        'total_trades': len(df_trades)
    }
    state_dict['result'] = res
    state_dict['status'] = 'completed'
    return res
