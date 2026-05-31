# ⚡ Francotirador de Liquidez — Trading Radar

Motor cuantitativo de selección de acciones S&P 500 con régimen VIX + señales técnicas + filtro fundamental.

## 🚀 Correr en Streamlit Cloud

1. Haz fork o sube la carpeta `REVOLUTION_ENTREGABLE/` como repositorio GitHub.
2. En [share.streamlit.io](https://share.streamlit.io), apunta a `app.py` como archivo principal.
3. ¡Listo! No se necesita configuración adicional.

## 💻 Correr localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Estructura

```
REVOLUTION_ENTREGABLE/
├── app.py                  # App principal Streamlit
├── lab_tickers.py          # Obtiene lista S&P 500 desde Wikipedia
├── requirements.txt        # Dependencias Python
├── .gitignore
├── lab/                    # Motor cuantitativo
│   ├── regime_detector.py  # Clasificador de régimen VIX
│   ├── indicators.py       # Indicadores técnicos
│   ├── rule_engine.py      # Motor de reglas por régimen
│   └── monte_carlo.py      # Simulación GBM
└── data/
    └── fetcher.py          # Descarga OHLCV con yfinance
```

## 📊 Funcionalidades

- **Radar S&P 500**: Escanea ~500 acciones con filtros técnicos + fundamentales por régimen de mercado
- **Análisis Individual**: Evalúa cualquier ticker con Monte Carlo a 5 días (10,000 trayectorias)
- **Régimen de Mercado**: Historial VIX 90 días con clasificación automática (CALM / SLOW_BEAR / FAST_CRASH)
- **Correlación**: Matriz de covarianza cruzada Top 30 S&P 500
