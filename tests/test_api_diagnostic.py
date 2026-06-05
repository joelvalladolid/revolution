"""
Test diagnostico de APIs - identifica que capa falla.
Ejecutar: python tests/test_api_diagnostic.py
"""
import sys, os, datetime, traceback

# Fix encoding for Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

PASS = "[OK]"
FAIL = "[FAIL]"

results = []

def log(name, ok, detail=""):
    status = PASS if ok else FAIL
    results.append((name, ok, detail))
    print(f"  {status} {name}: {detail}")


print("=" * 70)
print("DIAGNÓSTICO DE APIs — revolution-main")
print("=" * 70)

# ── 1. yfinance básico ──────────────────────────────────────────────────────
print("\n[1] yfinance — descarga precio")
try:
    import yfinance as yf
    print(f"    yfinance version: {yf.__version__}")
    tk = yf.Ticker("AAPL")
    hist = tk.history(period="5d")
    if hist.empty:
        log("yfinance.history(AAPL, 5d)", False, "DataFrame vacío")
    else:
        price = float(hist['Close'].iloc[-1])
        log("yfinance.history(AAPL, 5d)", True, f"precio=${price:.2f}, filas={len(hist)}")
except Exception as e:
    log("yfinance.history(AAPL, 5d)", False, f"EXCEPCIÓN: {e}")

# ── 2. yfinance.download ────────────────────────────────────────────────────
print("\n[2] yfinance.download — descarga histórica larga")
try:
    start = (datetime.date.today() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')
    df = yf.download("AAPL", start=start, progress=False)
    if df.empty:
        log("yf.download(AAPL, 2y)", False, "DataFrame vacío")
    else:
        # Verificar si columnas son MultiIndex
        import pandas as pd
        if isinstance(df.columns, pd.MultiIndex):
            log("yf.download MultiIndex", True, f"columns={list(df.columns[:3])}")
            df.columns = df.columns.droplevel(1)
        log("yf.download(AAPL, 2y)", True, f"filas={len(df)}, cols={list(df.columns)}")
except Exception as e:
    log("yf.download(AAPL, 2y)", False, f"EXCEPCIÓN: {e}")

# ── 3. yfinance — VIX ───────────────────────────────────────────────────────
print("\n[3] yfinance — VIX (régimen)")
try:
    vix_tk = yf.Ticker("^VIX")
    vix_hist = vix_tk.history(period="3mo")
    if vix_hist.empty:
        log("VIX history", False, "DataFrame vacío")
    else:
        vix_val = float(vix_hist['Close'].iloc[-1])
        log("VIX history", True, f"VIX={vix_val:.2f}, filas={len(vix_hist)}")
except Exception as e:
    log("VIX history", False, f"EXCEPCIÓN: {e}")

# ── 4. yfinance.info (puede fallar por rate limit) ──────────────────────────
print("\n[4] yfinance — info/fast_info")
try:
    tk = yf.Ticker("AAPL")
    try:
        fi = tk.fast_info
        log("fast_info", True, f"last_price={fi.last_price}, market_cap={fi.market_cap}")
    except Exception as e:
        log("fast_info", False, f"{e}")

    try:
        info = tk.info
        if info and isinstance(info, dict) and len(info) > 5:
            log("tk.info", True, f"keys={len(info)}, sector={info.get('sector','?')}")
        else:
            log("tk.info", False, f"info vacío o insuficiente: {type(info)}")
    except Exception as e:
        log("tk.info", False, f"EXCEPCIÓN: {e}")
except Exception as e:
    log("yfinance info/fast_info", False, f"EXCEPCIÓN global: {e}")

# ── 5. yahooquery ────────────────────────────────────────────────────────────
print("\n[5] yahooquery — fundamentales")
try:
    from yahooquery import Ticker as YQTicker
    import yahooquery
    print(f"    yahooquery version: {yahooquery.__version__}")
    yq = YQTicker("AAPL", asynchronous=False, validate=True)

    fd = yq.financial_data
    if isinstance(fd, dict) and "AAPL" in fd:
        val = fd["AAPL"]
        if isinstance(val, dict):
            log("yahooquery financial_data", True, f"keys={list(val.keys())[:5]}")
        elif isinstance(val, str):
            log("yahooquery financial_data", False, f"string error: {val[:100]}")
        else:
            log("yahooquery financial_data", False, f"tipo inesperado: {type(val)}")
    else:
        log("yahooquery financial_data", False, f"AAPL no encontrado en: {type(fd)}")

    ks = yq.key_stats
    if isinstance(ks, dict) and "AAPL" in ks:
        val = ks["AAPL"]
        if isinstance(val, dict):
            log("yahooquery key_stats", True, f"PE={val.get('forwardPE')}, PEG={val.get('pegRatio')}")
        else:
            log("yahooquery key_stats", False, f"tipo: {type(val)}")
    else:
        log("yahooquery key_stats", False, f"no data")

except ImportError:
    log("yahooquery import", False, "No instalado")
except Exception as e:
    log("yahooquery", False, f"EXCEPCIÓN: {e}")

# ── 6. data.fetcher — fetch_history ──────────────────────────────────────────
print("\n[6] data.fetcher — fetch_history")
try:
    from data.fetcher import fetch_history
    start = (datetime.date.today() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')
    end = datetime.date.today().strftime('%Y-%m-%d')
    result = fetch_history(["AAPL"], start=start, end=end)
    df = result.get("AAPL")
    if df is None or df.empty:
        log("fetch_history(AAPL)", False, "retornó None/vacío")
    else:
        log("fetch_history(AAPL)", True, f"filas={len(df)}, cols={list(df.columns)}")
        if len(df) < 252:
            log("fetch_history suficiente", False, f"solo {len(df)} filas, necesita >=252")
        else:
            log("fetch_history suficiente", True, f"{len(df)} >= 252 filas OK")
except Exception as e:
    log("fetch_history", False, f"EXCEPCIÓN: {e}\n{traceback.format_exc()}")

# ── 7. data_fetcher — fetch_stock_data ───────────────────────────────────────
print("\n[7] data_fetcher — fetch_stock_data (fundamentales completos)")
try:
    from data_fetcher import fetch_stock_data
    data = fetch_stock_data("AAPL")
    if data and isinstance(data, dict) and data.get("price"):
        log("fetch_stock_data(AAPL)", True,
            f"price={data.get('price')}, sector={data.get('sector')}, "
            f"PE={data.get('trailing_pe')}, ROE={data.get('roe')}")
    elif data and isinstance(data, dict):
        log("fetch_stock_data(AAPL)", False,
            f"dict con {len(data)} keys pero price=None. Keys: {list(data.keys())[:10]}")
    else:
        log("fetch_stock_data(AAPL)", False, f"retornó: {type(data)}")
except Exception as e:
    log("fetch_stock_data", False, f"EXCEPCIÓN: {e}\n{traceback.format_exc()}")

# ── 8. lab.indicators ───────────────────────────────────────────────────────
print("\n[8] lab.indicators — cálculos técnicos")
try:
    from lab.indicators import ema_discount, bollinger_pctB, stochastic_k, rsi as rsi_fn
    from data.fetcher import fetch_history
    start = (datetime.date.today() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')
    end = datetime.date.today().strftime('%Y-%m-%d')
    result = fetch_history(["AAPL"], start=start, end=end)
    df = result.get("AAPL")
    if df is not None and len(df) >= 252:
        ema = ema_discount(df)
        bb = bollinger_pctB(df)
        sk = stochastic_k(df)
        r = rsi_fn(df)
        import pandas as pd
        last_ema = float(ema.iloc[-1]) if pd.notna(ema.iloc[-1]) else None
        last_rsi = float(r.iloc[-1]) if pd.notna(r.iloc[-1]) else None
        log("indicators", True, f"EMA_disc={last_ema:.2f}%, RSI={last_rsi:.1f}")
    else:
        log("indicators", False, "sin datos suficientes para calcular")
except Exception as e:
    log("indicators", False, f"EXCEPCIÓN: {e}")

# ── 9. lab.regime_detector ──────────────────────────────────────────────────
print("\n[9] lab.regime_detector — clasificación régimen")
try:
    from lab.regime_detector import classify_regime
    vix_tk = yf.Ticker("^VIX")
    vix_hist = vix_tk.history(period="3mo")
    if not vix_hist.empty:
        regimes = classify_regime(vix_hist['Close'])
        current = str(regimes.iloc[-1])
        log("regime_detector", True, f"régimen actual={current}")
    else:
        log("regime_detector", False, "VIX vacío")
except Exception as e:
    log("regime_detector", False, f"EXCEPCIÓN: {e}")

# ── 10. estrategia + data_fetcher combinados ────────────────────────────────
print("\n[10] estrategia — evaluar_protocolo_accion")
try:
    from estrategia import evaluar_protocolo_accion
    from data_fetcher import fetch_stock_data
    data = fetch_stock_data("MSFT")
    if data and data.get("price"):
        price = data["price"]
        tech_real = {"rsi": 50, "sma_200": price, "fifty_two_position": 50}
        res = evaluar_protocolo_accion(data, tech_real, 4.2, price, soportes=[], profile='B')
        log("evaluar_protocolo", True,
            f"passed={res.get('passed')}/{res.get('total')}, "
            f"verdicts={len(res.get('verdicts',[]))}")
    else:
        log("evaluar_protocolo", False, "fetch_stock_data retornó sin precio")
except ImportError as e:
    log("evaluar_protocolo", False, f"ImportError: {e}")
except Exception as e:
    log("evaluar_protocolo", False, f"EXCEPCIÓN: {e}")

# ── 11. lab_tickers — Wikipedia S&P 500 ─────────────────────────────────────
print("\n[11] lab_tickers — S&P 500 lista")
try:
    # No importar desde lab_tickers directamente porque usa @st.cache_data
    import requests, io
    import pandas as pd
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0",
    }
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    html = io.StringIO(resp.text)
    tables = pd.read_html(html, attrs={"id": "constituents"}, flavor="lxml")
    tickers = [str(t).replace(".", "-") for t in tables[0]["Symbol"].tolist()]
    log("Wikipedia SP500", True, f"{len(tickers)} tickers descargados")
except Exception as e:
    log("Wikipedia SP500", False, f"EXCEPCIÓN: {e}")


# ── RESUMEN ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESUMEN DIAGNÓSTICO")
print("=" * 70)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"  {PASS} Pasaron: {passed}")
print(f"  {FAIL} Fallaron: {failed}")

if failed > 0:
    print(f"\n  Fallos detectados:")
    for name, ok, detail in results:
        if not ok:
            print(f"    {FAIL} {name}: {detail}")
else:
    print(f"\n  Todo funciona correctamente.")

print("=" * 70)
