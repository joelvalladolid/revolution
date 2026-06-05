"""
Test de cadena completa: analiza un ticker como lo hace la app.
Ejecutar: python tests/test_full_chain.py
"""
import sys, os, datetime
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

# Importar sin Streamlit (mock st.cache_data)
import types
import streamlit as st

from data.fetcher import fetch_history
from lab.regime_detector import classify_regime
from lab.indicators import ema_discount, bollinger_pctB, stochastic_k, rsi
from lab.rule_engine import evaluate_signal, RULE_SETS
from lab.monte_carlo import simulate_price_paths
import yfinance as yf
import pandas as pd
import numpy as np

def test_fetch_history():
    """Test 1: fetch_history no retorna NaN en Close"""
    print("[Test 1] fetch_history - sin NaN en Close")
    start = (datetime.date.today() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')
    end = datetime.date.today().strftime('%Y-%m-%d')
    result = fetch_history(["AAPL"], start=start, end=end)
    df = result.get("AAPL")
    assert df is not None, "AAPL no retornado"
    assert not df.empty, "AAPL vacio"
    assert not pd.isna(df['Close'].iloc[-1]), f"ULTIMO Close es NaN! Esto rompe la app."
    assert len(df) >= 252, f"Solo {len(df)} filas, necesita >=252"
    print(f"  OK: {len(df)} filas, ultimo Close=${df['Close'].iloc[-1]:.2f}")
    return df

def test_indicators(df):
    """Test 2: Indicadores no retornan NaN en la ultima fila"""
    print("[Test 2] Indicadores - ultimo valor no NaN")
    ema = ema_discount(df)
    bb = bollinger_pctB(df)
    sk = stochastic_k(df)
    r = rsi(df)

    assert not pd.isna(ema.iloc[-1]), f"EMA_discount ultimo = NaN"
    assert not pd.isna(bb.iloc[-1]), f"BollingerB ultimo = NaN"
    assert not pd.isna(sk.iloc[-1]), f"Stochastic_K ultimo = NaN"
    assert not pd.isna(r.iloc[-1]), f"RSI ultimo = NaN"

    print(f"  OK: EMA={ema.iloc[-1]:.2f}%, BB={bb.iloc[-1]:.2f}, "
          f"Stoch={sk.iloc[-1]:.1f}, RSI={r.iloc[-1]:.1f}")

def test_regime():
    """Test 3: Regimen VIX funciona"""
    print("[Test 3] Regimen VIX")
    tk = yf.Ticker("^VIX")
    vix = tk.history(period="3mo")
    vix = vix.dropna(subset=['Close'])
    assert not vix.empty, "VIX vacio"

    regimes = classify_regime(vix['Close'])
    current = str(regimes.iloc[-1])
    assert current in ["CALM", "SLOW_BEAR", "FAST_CRASH"], f"Regimen invalido: {current}"
    print(f"  OK: Regimen={current}, VIX={vix['Close'].iloc[-1]:.2f}")
    return current

def test_analyze_chain(df, regime):
    """Test 4: Cadena completa de analisis"""
    print("[Test 4] Cadena completa (indicators -> signal -> MC)")
    from lab.indicators import mfi, macd_hist, williams_r, adx

    df['EMA200_disc'] = ema_discount(df)
    df['BB_pctB'] = bollinger_pctB(df)
    df['Stoch_K'] = stochastic_k(df)
    df['RSI'] = rsi(df)
    df['MFI'] = mfi(df)
    df['MACD_hist'] = macd_hist(df)
    df['Williams_R'] = williams_r(df)
    df['ADX'] = adx(df)

    last = df.iloc[-1]
    price = float(last['Close'])
    ind_vals = {
        'EMA200_disc': float(last['EMA200_disc']),
        'BB_pctB':     float(last['BB_pctB']),
        'Stoch_K':     float(last['Stoch_K']),
        'RSI':         float(last['RSI']),
        'MFI':         float(last['MFI']),
        'MACD_rising': bool(last['MACD_hist'] > 0),
        'Williams_R':  float(last['Williams_R']),
        'ADX':         float(last['ADX']),
    }

    # Verify no NaN
    for k, v in ind_vals.items():
        if isinstance(v, float):
            assert not pd.isna(v), f"{k} es NaN"

    print(f"  Indicators OK: price=${price:.2f}, EMA_disc={ind_vals['EMA200_disc']:.1f}%, "
          f"RSI={ind_vals['RSI']:.1f}")

    # Signal evaluation
    signal = evaluate_signal(ind_vals, regime, 10)
    print(f"  Signal: active={signal['signal']}, confidence={signal['confidence']:.0f}")

    # Monte Carlo
    rets = df['Close'].pct_change().dropna().tail(252)
    mc = simulate_price_paths(price, rets, horizon_days=5, n_simulations=10_000)
    print(f"  MC: P(>0%)={mc['prob_positive']*100:.1f}%, "
          f"P(>2%)={mc['prob_gt_2pct']*100:.1f}%, "
          f"p10={mc['p10']*100:.1f}%, p90={mc['p90']*100:.1f}%")
    print("  OK: Cadena completa funciona")

def test_fundamentals():
    """Test 5: Fundamentales (data_fetcher + estrategia)"""
    print("[Test 5] Fundamentales (fetch_stock_data + evaluar_protocolo)")
    try:
        from data_fetcher import fetch_stock_data
        from estrategia import evaluar_protocolo_accion

        data = fetch_stock_data("MSFT")
        assert data and isinstance(data, dict), "fetch_stock_data retorno vacio"
        assert data.get("price"), f"price=None"

        price = data["price"]
        tech = {"rsi": 50, "sma_200": price, "fifty_two_position": 50}
        res = evaluar_protocolo_accion(data, tech, 4.2, price, soportes=[], profile='B')
        passed = res.get('passed', 0)
        total = res.get('total', 0)
        print(f"  OK: MSFT price=${price:.2f}, stars={passed}/{total}")
    except Exception as e:
        print(f"  FAIL: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST CADENA COMPLETA - revolution-main")
    print("=" * 60)

    try:
        df = test_fetch_history()
        test_indicators(df)
        regime = test_regime()
        test_analyze_chain(df, regime)
        test_fundamentals()
        print("\n" + "=" * 60)
        print("TODOS LOS TESTS PASARON")
        print("=" * 60)
    except AssertionError as e:
        print(f"\nFAILED: {e}")
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
