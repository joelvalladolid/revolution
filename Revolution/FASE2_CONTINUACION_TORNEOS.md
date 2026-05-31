# CONTINUACIÓN — FASE 2 EXTENDIDA + TORNEOS DE ESTRATEGIAS
## Documento de investigación e implementación para Gemini
### Sistema: Francotirador de Liquidez — Trading Cuantitativo

---

## 🔷 CONTEXTO DEL PROYECTO (leer antes de todo)

Este prompt es la **continuación directa** del documento de Fase 2 enviado anteriormente.
El sistema ya existe y está parcialmente construido. No partir de cero.

### Lo que ya está implementado y validado (no reinventar):

**Proyecto 1 — Inversión largo plazo** (`francotirador-inversion/`)
- Motor de análisis fundamental: 16 checks reales con datos de Yahoo Finance
- Tres perfiles validados: A (ETFs, RSI≤30), B (balanceado, F11_CashIntRSI), C (alta convicción, F21_UltraQ)
- Sistema de pisos con confirmación de volumen (1.2× MA20)
- Resultados del torneo de 400+ modelos sobre S&P 500 completo:
  - Ganador TEST (2021-2024): F23_CashKing +83.2% vs Dumb DCA +32.4%
  - Ganador TRAIN (2014-2020): F23_CashKing +252% vs Dumb DCA +112.3%
  - Hit rate: 81-95% según período

**Proyecto 2 — Trading corto plazo** (`trading_project/`)
- `lab/indicators.py`: RSI, MACD, Bollinger %B, MFI, Williams %R, CCI,
  Stochastic K, ATR normalizado, EMA discount, OBV slope, VWAP deviation,
  Hurst exponent, ADX, ROC, CMF — todos vectorizados con pandas/numpy
- `lab/regime_detector.py`: Clasificador VIX → CALM / SLOW_BEAR / FAST_CRASH
- `lab/rule_engine.py`: Motor de reglas dinámico por régimen
- `lab/tournament_short.py`: Torneo de indicadores con hit rate, edge, p-value
- `lab/monte_carlo.py`: GBM con volatilidad real, 10k simulaciones
- `lab/backtest_combined.py`: Backtest con Kelly, CVaR, max racha pérdidas

**Resultados validados del torneo de indicadores a 1 día (S&P 500, 2015-2024):**
```
Indicador      | Edge FULL | Edge BEAR_2022 | Edge COVID | P-Value
RSI_14 ≤ 20    |  +2.79pp  |    +9.27pp     |  -8.98pp   | 3.3e-09
EMA200_disc 15%|  +3.25pp  |    +9.65pp     |  +1.97pp   | sig.
BB_pctB ≤ 0.2  |  +2.03pp  |    +3.66pp     |  +0.51pp   | sig.
Stoch_K ≤ 20   |  +1.77pp  |    +6.86pp     |  +1.37pp   | sig.
MFI_14 ≤ 20    |  +2.33pp  |   +12.83pp     |  -9.29pp   | sig.
```

**Métricas de riesgo de RSI≤20 con horizonte 1 día:**
```
Período    | Edge  | Kelly % | CVaR 5% | Max consec. losses
bear_2022  | 8.26pp|  26.8%  |  -5.1%  |         6
full       | 2.72pp|  13.7%  |  -7.0%  |         8
```

**Stack tecnológico disponible:**
- Python 3.12, pandas, numpy, scipy, yfinance
- Streamlit (web), Plotly (gráficos)
- Computadora de alta gama disponible para torneos masivos (8+ cores, RAM generosa)
- Datos: Yahoo Finance gratuito (sin límite de tickers, con rate limiting manejado)
- Sin APIs de pago requeridas

---

## 🔷 BLOQUE 5: TORNEO DE ESTRATEGIAS — La Capa de Validación Empírica

La Fase 2 del documento anterior define las métricas (Win Rate, EV, Profit Factor, etc.).
Este bloque define **cómo encontrar empíricamente qué combinación de esas métricas
produce la estrategia más rentable**, usando torneos masivos con datos reales.

### Principio fundamental del torneo

Un torneo de estrategias responde: **¿Qué configuración de parámetros,
aplicada sobre el S&P 500 completo en datos históricos reales,
produce el mayor Expected Value con riesgo controlado?**

No es optimización manual. Es búsqueda exhaustiva validada estadísticamente.

### Arquitectura del torneo de estrategias

```
UNIVERSO: S&P 500 (~500 tickers) — datos yfinance 2015-2024

ESPACIO DE BÚSQUEDA:
  ├── Indicadores de entrada (señal de compra)
  │     ├── RSI threshold: [15, 18, 20, 22, 25, 28, 30]
  │     ├── EMA200 discount: [8, 10, 12, 15, 18, 20]%
  │     ├── BB %B threshold: [0.05, 0.10, 0.15, 0.20, 0.25]
  │     ├── Stoch K: [10, 15, 20, 25]
  │     ├── Confirmadores: [ninguno, +MACD, +MFI, +volumen]
  │     └── Régimen requerido: [cualquiera, SLOW_BEAR only, no FAST_CRASH]
  │
  ├── Horizonte de salida
  │     └── [1, 2, 3, 5, 7, 10] días
  │
  ├── Stop Loss (en ATR múltiples)
  │     └── [1.0×ATR, 1.5×ATR, 2.0×ATR, sin SL]
  │
  └── Take Profit (en ATR múltiples)
        └── [1.5×ATR, 2.0×ATR, 3.0×ATR, sin TP]

TOTAL COMBINACIONES ESTIMADAS: ~8,000-15,000 estrategias
```

### Métricas de evaluación por estrategia

Para cada combinación de parámetros, calcular sobre el S&P 500 completo:

```python
METRICAS_POR_ESTRATEGIA = {
    # Del documento de Fase 2:
    "win_rate":         float,   # % operaciones ganadoras
    "expected_value":   float,   # EV = P_win×avg_win - P_loss×avg_loss
    "profit_factor":    float,   # suma_ganancias / suma_pérdidas
    "sharpe_90d":       float,   # Sharpe últimos 90 días de señales
    "sortino":          float,   # Sortino ratio
    "calmar":           float,   # retorno_anualizado / max_drawdown

    # Del sistema existente:
    "edge_pp":          float,   # ventaja sobre baseline (pp)
    "p_value":          float,   # significancia estadística
    "kelly_pct":        float,   # tamaño óptimo de posición
    "cvar_5pct":        float,   # pérdida peor 5% escenarios
    "max_consec_loss":  int,     # racha perdedora máxima
    "n_signals":        int,     # número de señales generadas
    "coverage_pct":     float,   # % de tickers que generaron señal

    # Nuevo — estabilidad temporal:
    "edge_bull":        float,   # edge en régimen CALM
    "edge_bear":        float,   # edge en régimen SLOW_BEAR
    "edge_crash":       float,   # edge en régimen FAST_CRASH
    "consistency":      float,   # % de sub-períodos donde edge > 0
}
```

### Score maestro del torneo (integración con Fase 2)

```python
def score_estrategia(m: dict) -> float:
    """
    Combina las métricas del documento de Fase 2 con las del torneo existente.
    Retorna score 0-100 para ranking.
    """
    # Normalización min-max sobre todo el espacio de estrategias testeadas
    # (cada valor se normaliza contra el mejor y peor de todas las estrategias)

    score = (
        normalizar(m["win_rate"])          * 0.20 +
        normalizar(m["expected_value"])    * 0.20 +
        normalizar(m["profit_factor"])     * 0.12 +
        normalizar(m["sharpe_90d"])        * 0.12 +
        normalizar(m["sortino"])           * 0.08 +
        normalizar(m["edge_pp"])           * 0.10 +
        normalizar(m["consistency"])       * 0.10 +
        normalizar(m["calmar"])            * 0.08
    )

    # Penalizaciones duras (no negociables):
    if m["expected_value"]    <= 0:   score *= 0.0   # EV negativo = descalificado
    if m["win_rate"]          < 0.52: score *= 0.3   # por debajo del azar
    if m["p_value"]           > 0.05: score *= 0.5   # no significativo
    if m["max_consec_loss"]   >= 10:  score *= 0.7   # racha perdedora peligrosa
    if m["n_signals"]         < 50:   score *= 0.4   # muestra estadística insuficiente
    if m["edge_crash"]        < -3.0: score *= 0.6   # destruye capital en crashes

    # Bonus:
    if m["edge_bull"] > 0 and m["edge_bear"] > 0 and m["edge_crash"] > 0:
        score *= 1.15  # all-weather premium

    return min(score * 100, 100)
```

### Protocolo anti-overfitting del torneo

```
SPLIT TEMPORAL OBLIGATORIO:
  IN-SAMPLE  (búsqueda):   2015-01-01 → 2020-12-31  (6 años)
  OUT-SAMPLE (validación): 2021-01-01 → 2024-12-31  (4 años, NO TOCAR hasta final)

PROCEDIMIENTO:
  1. Correr torneo completo sobre IN-SAMPLE → top 20 estrategias
  2. Para cada top-20: correr sobre OUT-SAMPLE
  3. Estrategia válida si:
     a. Aparece en top 20 de IN-SAMPLE
     b. EV > 0 en OUT-SAMPLE
     c. Score OUT-SAMPLE ≥ 60% del score IN-SAMPLE
     d. Edge consistente en al menos 2 de 3 regímenes en OUT-SAMPLE

  REGLA DE ORO: si un modelo gana en IN pero pierde en OUT,
  es overfitting al período de entrenamiento. Descartarlo sin excepciones.
```

### Implementación técnica del torneo (para computadora de alta gama)

```python
# lab/strategy_tournament.py

from itertools import product
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np

# Definición del espacio de búsqueda
PARAM_SPACE = {
    "rsi_threshold":    [15, 18, 20, 22, 25, 28, 30, 35, 999],  # 999=sin check
    "ema_discount":     [0, 5, 8, 10, 12, 15, 18, 20],          # 0=sin check
    "bb_pctB":          [0.05, 0.10, 0.15, 0.20, 0.25, 999],
    "stoch_k":          [10, 15, 20, 25, 999],
    "exit_days":        [1, 2, 3, 5, 7, 10],
    "sl_atr_mult":      [1.0, 1.5, 2.0, 999],                   # 999=sin SL
    "tp_atr_mult":      [1.5, 2.0, 3.0, 999],                   # 999=sin TP
    "regime_filter":    ["none", "no_calm", "bear_only", "no_crash"],
    "require_volume":   [False, True],                           # vol > 1.5× MA20
}

def run_strategy(params: dict, price_data: dict, regime_data: pd.Series) -> dict:
    """
    Simula una estrategia sobre todos los tickers con SL/TP reales.
    Retorna todas las métricas de evaluación.

    La simulación es event-driven:
    - DÍA 0: señal activa → entrada al precio de cierre
    - DÍA 1 a N: evaluar SL, TP, o exit_days
    - Registrar P&L exacto por operación
    """
    all_trades = []

    for ticker, df in price_data.items():
        signals = generate_signals(df, params, regime_data)
        for signal_date in signals:
            trade = simulate_trade(df, signal_date, params)
            if trade: all_trades.append(trade)

    return calculate_all_metrics(all_trades)

def run_tournament_parallel(price_data: dict, regime_data: pd.Series,
                            n_workers: int = None) -> pd.DataFrame:
    """
    Corre todas las combinaciones de parámetros en paralelo.
    Con 8 cores y ~10,000 estrategias sobre 500 tickers:
    Tiempo estimado: 2-4 horas.
    """
    if n_workers is None:
        n_workers = cpu_count() - 1  # dejar 1 core libre

    all_params = [dict(zip(PARAM_SPACE.keys(), combo))
                  for combo in product(*PARAM_SPACE.values())]

    print(f"Estrategias a evaluar: {len(all_params):,}")
    print(f"Tickers: {len(price_data)}")
    print(f"Workers: {n_workers}")

    worker = partial(run_strategy,
                     price_data=price_data,
                     regime_data=regime_data)

    with Pool(n_workers) as pool:
        results = pool.map(worker, all_params)

    df_results = pd.DataFrame(results)
    df_results["score"] = df_results.apply(score_estrategia, axis=1)
    return df_results.sort_values("score", ascending=False)
```

### Output del torneo: tabla de campeones

```
TOP 10 ESTRATEGIAS — Torneo completo (IN-SAMPLE 2015-2020)
┌───────────────────────────────────────────────────────────────────────────┐
│ # │ RSI │ EMA% │ BB  │ Exit │ SL   │ Regime  │ WR% │ EV%  │ PF  │ Score │
├───────────────────────────────────────────────────────────────────────────┤
│ 1 │ ≤20 │ 12%  │0.15 │ 3d   │1.5×  │no_crash │67.3 │+2.1% │2.84 │ 91.2  │
│ 2 │ ≤22 │ 10%  │ —   │ 5d   │2.0×  │any      │65.1 │+1.8% │2.61 │ 87.4  │
│ 3 │ ≤18 │ 15%  │0.10 │ 2d   │1.0×  │bear_only│71.2 │+2.4% │3.12 │ 85.1  │
│...│ ... │ ...  │ ... │ ...  │ ...  │ ...     │ ... │ ...  │ ... │ ...   │
└───────────────────────────────────────────────────────────────────────────┘

VALIDACIÓN OUT-OF-SAMPLE (2021-2024):
  Estrategia #1: WR 64.8% (+/-2.5pp vs in-sample) ✅ VÁLIDA
  Estrategia #2: WR 61.2% ✅ VÁLIDA
  Estrategia #3: WR 58.1% ⚠️ Degradación > 10pp vs in-sample
```

---

## 🔷 BLOQUE 6: STOP LOSS Y TAKE PROFIT BASADOS EN ATR

Integración del ATR Ratio (del documento de Fase 2) en la gestión táctica de cada operación.

### Cálculo del ATR dinámico por operación

```python
def calcular_niveles_operacion(precio_entrada: float,
                                atr_14: float,
                                params: dict,
                                direccion: str = "long") -> dict:
    """
    Calcula SL y TP exactos basados en el ATR del día de entrada.
    El ATR se recalcula cada día con datos reales de Yahoo Finance.
    """
    atr_pct = atr_14 / precio_entrada * 100

    # Ajuste de tamaño de posición por volatilidad
    if atr_pct < 1.5:
        size_mult = 1.20   # baja volatilidad → más capital
    elif atr_pct < 2.5:
        size_mult = 1.00   # volatilidad normal
    elif atr_pct < 4.0:
        size_mult = 0.80   # alta volatilidad → reducir
    else:
        size_mult = 0.60   # volatilidad extrema → mínimo

    if params["sl_atr_mult"] < 999:
        sl_precio = precio_entrada - (params["sl_atr_mult"] * atr_14)
        sl_pct    = params["sl_atr_mult"] * atr_pct
    else:
        sl_precio = None
        sl_pct    = None

    if params["tp_atr_mult"] < 999:
        tp_precio = precio_entrada + (params["tp_atr_mult"] * atr_14)
        tp_pct    = params["tp_atr_mult"] * atr_pct
        rr_ratio  = params["tp_atr_mult"] / params["sl_atr_mult"] if sl_pct else None
    else:
        tp_precio = None
        tp_pct    = None
        rr_ratio  = None

    return {
        "entrada":    precio_entrada,
        "sl":         sl_precio,
        "tp":         tp_precio,
        "sl_pct":     sl_pct,
        "tp_pct":     tp_pct,
        "rr_ratio":   rr_ratio,
        "atr_pct":    atr_pct,
        "size_mult":  size_mult,
    }
```

### Regla de oro RR Ratio (del documento de Fase 2, implementada)

```
CRITERIOS DE RECHAZO DE OPERACIÓN:
  Si RR Ratio < 1.5 → NO operar (ganancia potencial no justifica el riesgo)
  Si ATR_pct > 5.0% → reducir tamaño al 50% independientemente del Kelly
  Si SL se activa → cerrar inmediatamente, sin excepción
  Si TP se activa → cerrar el 70%, mantener el 30% con trailing stop
```

---

## 🔷 BLOQUE 7: INTEGRACIÓN DEL SCORE MAESTRA CON KELLY Y CONVEXIDAD

Unificación de la fórmula maestra (Fase 2) con el Kelly Criterion ya calculado
y el método de convexidad para asignación de capital.

### Pipeline completo de una sesión de trading

```python
def pipeline_sesion_diaria(capital_total: float,
                            tickers_sp500: list,
                            perfil: str = "moderado") -> dict:
    """
    Ejecuta el pipeline completo para una sesión:
    1. Detectar régimen (VIX real)
    2. Filtrar candidatos (Fase 1: fundamentales + técnico)
    3. Calcular Score Fase 2 (Win Rate, EV, PF, Sharpe, RS, Sortino)
    4. Penalizar por correlación, ATR, Beta
    5. Asignar capital con convexidad
    6. Calcular SL/TP exactos por ATR
    7. Retornar órdenes listas para ejecutar

    Output: lista de órdenes con monto exacto en USD, SL, TP por cada activo
    """

    # PASO 1: Régimen actual
    regime = detect_current_regime()

    # PASO 2: Candidatos (Fase 1 existente)
    candidatos = []
    for ticker in tickers_sp500:
        señal = evaluate_entry_signal(ticker, regime)
        if señal["signal"]:
            candidatos.append({**señal, "ticker": ticker})

    if not candidatos:
        return {"señales": 0, "ordenes": [], "mensaje": "Sin señales hoy"}

    # PASO 3: Score Fase 2 para cada candidato
    for c in candidatos:
        historico = obtener_retornos_señales_historicas(c["ticker"], regime)
        c["score_fase2"] = calcular_score_fase2(historico, c)

    # PASO 4: Penalizaciones inter-activos
    candidatos = aplicar_penalizacion_correlacion(candidatos, threshold=0.85)
    candidatos = aplicar_penalizacion_atr(candidatos)
    candidatos = aplicar_penalizacion_beta(candidatos, perfil)

    # PASO 5: Asignación convexo
    alpha = {"conservador": 1.2, "moderado": 1.5, "agresivo": 2.0}[perfil]
    pesos = asignar_capital_convexo(candidatos, alpha=alpha)

    # PASO 6: SL/TP por ATR
    ordenes = []
    for c, peso in zip(candidatos, pesos):
        monto = capital_total * peso
        # Kelly adjusts monto down if strategy has lower kelly %
        kelly_adj = c.get("kelly_pct", 0.137)  # default: 13.7% del full kelly
        monto_kelly = min(monto, capital_total * kelly_adj)

        niveles = calcular_niveles_operacion(
            precio_entrada=c["precio_actual"],
            atr_14=c["atr_14"],
            params=c["best_strategy_params"]
        )
        ordenes.append({
            "ticker":       c["ticker"],
            "monto_usd":    round(monto_kelly, 2),
            "peso_pct":     round(peso * 100, 1),
            "entrada":      niveles["entrada"],
            "stop_loss":    niveles["sl"],
            "take_profit":  niveles["tp"],
            "rr_ratio":     niveles["rr_ratio"],
            "score":        round(c["score_fase2"], 1),
            "win_rate":     round(c["win_rate"] * 100, 1),
            "ev":           round(c["ev"] * 100, 2),
            "regime":       regime,
        })

    return {"señales": len(ordenes), "ordenes": ordenes, "capital_en_riesgo": sum(o["monto_usd"] for o in ordenes)}
```

---

## 🔷 BLOQUE 8: MONTE CARLO A NIVEL DE PORTAFOLIO

Extensión del Monte Carlo individual (ya implementado en `monte_carlo.py`)
al nivel de portafolio completo, considerando correlaciones reales.

```python
def simulate_portfolio_paths(positions: list,
                              correlation_matrix: np.ndarray,
                              horizon_days: int = 5,
                              n_simulations: int = 100_000) -> dict:
    """
    Simula N trayectorias para el portafolio completo.
    Usa matriz de correlaciones real entre los activos seleccionados.
    Con 100k simulaciones en computadora de alta gama: ~2-5 segundos.

    positions = [{"ticker": "AAPL", "weight": 0.56, "mu": 0.001, "sigma": 0.018}, ...]
    """
    n_assets = len(positions)
    mus    = np.array([p["mu"]    for p in positions])
    sigmas = np.array([p["sigma"] for p in positions])
    weights = np.array([p["weight"] for p in positions])

    # Cholesky decomposition para correlaciones reales
    L = np.linalg.cholesky(correlation_matrix)

    portfolio_returns = []

    for _ in range(n_simulations):
        # Shocks correlacionados
        Z = np.random.normal(0, 1, (horizon_days, n_assets))
        correlated_shocks = Z @ L.T

        # Retornos por activo
        daily_returns = mus + sigmas * correlated_shocks
        asset_paths = np.exp(np.cumsum(daily_returns, axis=0))

        # Retorno del portafolio al día H
        portfolio_return = (asset_paths[-1] - 1) @ weights
        portfolio_returns.append(portfolio_return)

    returns = np.array(portfolio_returns)

    return {
        "prob_positive":    float((returns > 0).mean()),
        "prob_gt_2pct":     float((returns > 0.02).mean()),
        "prob_gt_5pct":     float((returns > 0.05).mean()),
        "prob_loss_5pct":   float((returns < -0.05).mean()),
        "p10":              float(np.percentile(returns, 10)),
        "p25":              float(np.percentile(returns, 25)),
        "p50":              float(np.percentile(returns, 50)),
        "p75":              float(np.percentile(returns, 75)),
        "p90":              float(np.percentile(returns, 90)),
        "cvar_5pct":        float(returns[returns <= np.percentile(returns, 5)].mean()),
        "sharpe_5d":        float(returns.mean() / (returns.std() + 1e-8) * np.sqrt(252/horizon_days)),
    }
```

---

## 🔷 BLOQUE 9: RELATIVE STRENGTH RATING REAL

Implementación del RS Rating (Bloque 2 del documento de Fase 2) con datos reales.

```python
def calcular_rs_rating_sp500(tickers: list,
                              periodo_dias: int = 63) -> pd.Series:
    """
    Calcula el RS Rating de todos los tickers vs. el S&P 500 (^GSPC).
    Igual al sistema de IBD/Investor's Business Daily.

    RS > 1.20 → supera al índice un 20% en 63 días → sobreponderar
    RS < 0.80 → rezagado → penalizar en el score final

    Con yfinance, descarga ^GSPC y todos los tickers en paralelo.
    """
    spy_ret = descargar_retorno_periodo("SPY", periodo_dias)

    rs_ratings = {}
    for ticker in tickers:
        try:
            ticker_ret = descargar_retorno_periodo(ticker, periodo_dias)
            rs = (1 + ticker_ret) / (1 + spy_ret)
            rs_ratings[ticker] = round(rs, 3)
        except:
            rs_ratings[ticker] = 1.0  # neutro si falla

    return pd.Series(rs_ratings).sort_values(ascending=False)
```

---

## 🔷 BLOQUE 10: CORRELACIÓN INTER-ACTIVOS EN TIEMPO REAL

```python
def filtrar_por_correlacion(candidatos: list,
                             periodo_dias: int = 60,
                             threshold: float = 0.85) -> list:
    """
    Si dos candidatos tienen correlación > 0.85 en los últimos 60 días,
    mantener solo el de mayor Score y descartar el otro
    (o reducir ambos al 50% como dice la Fase 2).

    Importante: Tech vs. Tech correlación alta es normal.
    El filtro es más útil para reducir exposición dentro de un mismo sector.
    """
    if len(candidatos) < 2:
        return candidatos

    tickers = [c["ticker"] for c in candidatos]

    # Descargar retornos de los últimos N días con yfinance
    returns_df = yf.download(tickers, period=f"{periodo_dias}d",
                              progress=False)["Close"].pct_change().dropna()

    corr_matrix = returns_df.corr()

    # Greedy: ordenar por score, ir eliminando alta correlación
    candidatos_ordenados = sorted(candidatos, key=lambda x: x["score_fase2"],
                                   reverse=True)
    seleccionados = []
    descartados = set()

    for c in candidatos_ordenados:
        if c["ticker"] in descartados:
            continue
        seleccionados.append(c)
        # Descartar candidatos correlacionados con este
        for otro in candidatos_ordenados:
            if otro["ticker"] != c["ticker"] and otro["ticker"] not in descartados:
                if abs(corr_matrix.loc[c["ticker"], otro["ticker"]]) > threshold:
                    descartados.add(otro["ticker"])

    return seleccionados
```

---

## 🔷 RESUMEN: FLUJO COMPLETO DEL SISTEMA

```
┌──────────────────────────────────────────────────────────────────┐
│                    FRANCOTIRADOR TRADING SYSTEM                  │
│                    Flujo completo de una sesión                  │
└──────────────────────────────────────────────────────────────────┘

UNIVERSO: S&P 500 (~500 tickers via yfinance)
    │
    ▼
[RÉGIMEN] VIX real → CALM / SLOW_BEAR / FAST_CRASH
    │
    ▼
[FASE 1 — FILTRO] (ya construido y validado)
  Técnico: EMA200 disc + régimen-specific indicators
  Fundamental: 16 checks reales → ≥ 13/16 stars
    │
    ├──→ 0 candidatos → "Sin señales hoy, guardar cash"
    │
    ▼
[FASE 2 — SCORING] (a construir con este documento)
  Win Rate histórico condicional  (25%)
  Expected Value                  (20%)
  Profit Factor                   (12%)
  Sharpe 90d                      (12%)
  RS Rating vs SPY                (10%)
  Sortino                         (10%)
  Consistency (temporal)          (11%)
    │
  × penalización correlación
  × penalización ATR
  × penalización Beta
    │
    ▼
[TORNEO] top estrategia del torneo define parámetros de SL/TP
    │
    ▼
[CAPITAL] convexidad α + Kelly ajustado → peso exacto en %
    │
    ▼
[MONTE CARLO PORTAFOLIO] 100k sims → distribución de retornos
    │
    ▼
[OUTPUT FINAL]
  Para cada activo:
    ticker, monto_USD, entrada, SL, TP, RR ratio,
    win_rate, EV, score, prob_positivo_5d
```

---

## 🔷 PLAN DE INVESTIGACIÓN PARA GEMINI

Investiga y desarrolla en detalle estos puntos que el sistema aún no tiene:

1. **Win Rate condicional robusto**: ¿Cómo calcular el Win Rate histórico de
   una señal específica (RSI≤20 + EMA200≥12%) sobre el S&P 500 completo con
   datos de yfinance, distinguiendo por régimen VIX y período temporal?
   Incluir bootstrap para intervalos de confianza del Win Rate.

2. **Expected Value con distribución real**: El EV del documento usa ganancia/pérdida
   promedio. ¿Cómo mejorarlo usando la distribución empírica completa de retornos
   (no solo media), para capturar fat tails y asimetría?

3. **RS Rating dinámico en el radar**: ¿Cómo integrar el RS Rating en tiempo real
   en el radar de Streamlit sin ralentizarlo? (ya tiene ThreadPoolExecutor con 8 workers)

4. **Correlación en tiempo real sin lag**: La correlación de 60 días es retrospectiva.
   ¿Existe un estimador de correlación más reactivo (ej: correlación rodante exponencial)
   que sea más útil para sesiones diarias de trading?

5. **Torneo de ~10,000 estrategias en la computadora de alta gama**:
   ¿Cuántos workers de multiprocessing son óptimos para un torneo con
   500 tickers × 10,000 estrategias × 10 años de datos?
   ¿Es mejor usar multiprocessing.Pool, concurrent.futures, o joblib?
   ¿Qué chunking de tickers maximiza el throughput?

6. **Integración de SL/TP en el backtest existente**: El `backtest_combined.py`
   actual mide retorno a N días fijos. ¿Cómo adaptarlo para simular SL/TP reales
   (que pueden activarse antes de que se cumpla el horizonte)?
