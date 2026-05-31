# Francotirador de Liquidez - Trading App ⚡

![Motor Cuantitativo](https://img.shields.io/badge/Motor-Cuantitativo-7C3AED?style=for-the-badge)
![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge)
![Régimen VIX](https://img.shields.io/badge/Regimen-VIX-00C8E0?style=for-the-badge)

Aplicación de trading cuantitativo basada en un motor de reglas dinámico que adapta los umbrales de indicadores según el régimen del mercado (medido a través del VIX).

## 🗂️ Estructura del Proyecto

El proyecto está diseñado para funcionar como una aplicación web en Streamlit, manteniendo todos los módulos necesarios en la raíz y aislando las pruebas locales en su propia carpeta, listo para ser desplegado desde GitHub.

- `app.py`: El punto de entrada principal de la aplicación Streamlit. Contiene toda la interfaz visual.
- `lab/`: Módulo core con los motores lógicos.
  - `indicators.py`: Cálculo de indicadores técnicos (EMA, BB, Stoch, RSI, MACD).
  - `regime_detector.py`: Detección del régimen del mercado (CALM, SLOW_BEAR, FAST_CRASH) basado en el VIX.
  - `rule_engine.py`: Motor de reglas dinámicas según el régimen actual.
  - `monte_carlo.py`: Simulación Monte Carlo para proyección de precios.
- `data/`: Módulo encargado de la obtención y procesamiento de datos históricos usando `yfinance`.
- `lab_tickers.py`: Script auxiliar para obtener los tickers del S&P 500.
- `experiments/`: Resultados de torneos, backtesting y archivos de texto residuales de las iteraciones de desarrollo (NO es necesario para que la app web funcione).
- `requirements.txt`: Dependencias del proyecto.

## 🚀 Instalación y Uso

1. **Clonar el repositorio:**
```bash
git clone <URL_DEL_REPOSITORIO>
cd trading_project
```

2. **Instalar las dependencias:**
```bash
pip install -r requirements.txt
```

3. **Ejecutar la Aplicación Streamlit:**
```bash
streamlit run app.py
```

## ⚙️ Características

- **Detección de Régimen:** Analiza el VIX de los últimos 90 días para clasificar el mercado y aplicar distintos umbrales de exigencia técnica.
- **Análisis de Componentes:** Permite ejecutar un escáner masivo del S&P 500 para encontrar empresas que cumplan con la señal completa.
- **Análisis Individual:** Evaluación detallada de un ticker específico mostrando qué reglas cumple y cuáles le faltan.
- **Monte Carlo:** Simulación probabilística de 10,000 trayectorias a 5 días vista para activos analizados.

## 🧠 Cómo Funciona el Motor (Deep Dive de app.py)

El archivo `app.py` orquesta todo el proceso desde que se analiza un ticker hasta que se emite una recomendación final. Aquí tienes el paso a paso detallado:

### 1. Detección del Régimen de Mercado
- **Obtención de datos globales:** El sistema consulta el VIX (`^VIX`) de los últimos 3 meses y la tasa de bonos a 10 años (`^TNX`).
- **Clasificación (Regime Detection):** Evalúa el comportamiento del VIX y su cambio en los últimos 10 días para establecer el régimen actual del mercado:
  - `CALM`: Mercado tranquilo.
  - `SLOW_BEAR`: Mercado en tendencia bajista progresiva.
  - `FAST_CRASH`: Caída rápida o pánico.
- **Reglas Dinámicas:** Según el régimen, se cargan distintos umbrales técnicos (definidos en `RULE_SETS`).

### 2. Recolección de Datos por Ticker
- **Listado S&P 500:** Se obtienen los tickers actuales del índice.
- **Descarga Histórica:** Para cada ticker, se descargan mínimo 252 días (1 año) de cotizaciones diarias vía `yfinance`.
- **Cálculo de Indicadores Técnicos:**
  - **Primario:** `EMA200_disc` (Descuento porcentual respecto a la Media Móvil Exponencial de 200 días).
  - **Secundarios (Corto Plazo):** Bandas de Bollinger (`BB_pctB`), Oscilador Estocástico (`Stoch_K`), RSI, MFI, MACD, Williams %R, ADX.

### 3. Evaluación de Señales (Motor Cuantitativo)
- **Filtro Técnico Primario:** El activo debe alcanzar el descuento mínimo exigido frente a su EMA200 (ej. 8% en CALM, 12% en SLOW_BEAR, 15% en FAST_CRASH).
- **Confirmación Secundaria:** Si el régimen lo requiere (ej. `SLOW_BEAR`), los osciladores (`BB_pctB`, `Stoch_K`) deben confirmar sobreventa extrema.
- **Filtro Fundamental (Quality Check):** Si el componente técnico está cerca de cumplirse o si el régimen lo exige estrictamente (CALM o FAST_CRASH), se activa el filtro fundamental. Este evalúa hasta 16 puntos de calidad del negocio (usando rentabilidad, deuda, márgenes, etc.) otorgando un puntaje en "estrellas" (hasta 16 ⭐).

### 4. Puntuación de Confianza (Confidence Score)
El `app.py` calcula una puntuación de proximidad y confianza (0 a 100) combinando:
- **Puntaje Técnico (máx 40 pts):** Qué tan cerca o cuánto superó el umbral de la EMA200.
- **Puntaje Fundamental (máx 40 pts):** Ratio de estrellas obtenidas vs estrellas totales requeridas.
- **Bonus (máx 20 pts):** Si tanto lo técnico como lo fundamental superan el 80% del requisito.
- *Nota en la señal:* El motor de reglas (`rule_engine.py`) añade ponderaciones por confirmaciones adicionales para obtener un *Confidence Score* final normalizado sobre 100.

### 5. Emisión de la Señal Final
Según el *Confidence Score*, el dashboard web muestra el veredicto:
- **STRONG BUY (🟢):** Puntuación ≥ 75. Condiciones técnicas óptimas y fundamentales sólidos confirmados por múltiples indicadores.
- **BUY (🟩):** Puntuación ≥ 50. Condiciones buenas, señal activa pero sin confirmadores absolutos.
- **VIGILAR (🔵):** Puntuación < 50. El activo no tiene señal activa, pero está cerca en el radar (evaluado por descuento EMA).

### 6. Simulación de Escenarios (Monte Carlo)
Para los activos analizados, el sistema ejecuta **10,000 trayectorias** estocásticas proyectando el precio a 5 días vista, calculando las probabilidades estadísticas de que el retorno sea positivo o supere ciertas métricas.

## 🛡️ Despliegue en Github

Este proyecto está optimizado para su uso en plataformas cloud que soportan Streamlit (ej: Streamlit Community Cloud). Únicamente se necesitan los archivos de la raíz (`app.py`, `lab/`, `data/`, `lab_tickers.py` y `requirements.txt`).
