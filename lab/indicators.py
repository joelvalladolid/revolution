import pandas as pd
import numpy as np

def rsi(df: pd.DataFrame, period=14) -> pd.Series:
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd_hist(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.Series:
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

def bollinger_pctB(df: pd.DataFrame, period=20, std=2) -> pd.Series:
    sma = df['Close'].rolling(window=period).mean()
    rolling_std = df['Close'].rolling(window=period).std()
    upper_band = sma + (rolling_std * std)
    lower_band = sma - (rolling_std * std)
    return (df['Close'] - lower_band) / (upper_band - lower_band)

def mfi(df: pd.DataFrame, period=14) -> pd.Series:
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    
    positive_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
    
    positive_mf = pd.Series(positive_flow, index=df.index).rolling(window=period).sum()
    negative_mf = pd.Series(negative_flow, index=df.index).rolling(window=period).sum()
    
    mfi_ratio = positive_mf / negative_mf
    return 100 - (100 / (1 + mfi_ratio))

def williams_r(df: pd.DataFrame, period=14) -> pd.Series:
    highest_high = df['High'].rolling(window=period).max()
    lowest_low = df['Low'].rolling(window=period).min()
    return -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))

def cci(df: pd.DataFrame, period=20) -> pd.Series:
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: pd.Series(x).mad(), raw=True)
    # Pandas >= 2.0 removed mad(). Alternative: (x - x.mean()).abs().mean()
    # Let's use standard deviation or alternative
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad)

def stochastic_k(df: pd.DataFrame, k=14, d=3) -> pd.Series:
    lowest_low = df['Low'].rolling(window=k).min()
    highest_high = df['High'].rolling(window=k).max()
    stoch_k = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
    # We smooth it with 'd' period SMA
    return stoch_k.rolling(window=d).mean()

def atr_pct(df: pd.DataFrame, period=14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr / df['Close']

def volume_ratio(df: pd.DataFrame, period=20) -> pd.Series:
    sma_vol = df['Volume'].rolling(window=period).mean()
    return df['Volume'] / sma_vol

def ema_discount(df: pd.DataFrame, period=200) -> pd.Series:
    ema = df['Close'].ewm(span=period, adjust=False).mean()
    return ((ema - df['Close']) / ema) * 100

def obv_slope(df: pd.DataFrame, period=5) -> pd.Series:
    direction = np.where(df['Close'] > df['Close'].shift(1), 1, 
                         np.where(df['Close'] < df['Close'].shift(1), -1, 0))
    obv = (df['Volume'] * direction).cumsum()
    return obv.diff(period)

def vwap_discount_daily(df: pd.DataFrame) -> pd.Series:
    # Approximate daily VWAP using typical price. If intra-day data is not available, 
    # daily VWAP is just typical price. For true daily VWAP, we need intraday data.
    # Since we are using daily bars, we'll approximate rolling VWAP over N days
    # Wait, user asked for "vwap_discount_daily", let's use 20-day VWAP as in strategy
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    vp = tp * df['Volume']
    vwap = vp.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
    return ((vwap - df['Close']) / vwap) * 100

def hurst_exponent(df: pd.DataFrame, lags=range(2, 20)) -> pd.Series:
    """Detecta si la serie es tendencial (H>0.5), aleatoria (H=0.5) o mean-reverting (H<0.5)"""
    def calc_hurst(ts):
        if len(ts) < max(lags) + 1: return np.nan
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    return df['Close'].rolling(window=max(lags)*2).apply(calc_hurst, raw=True)

def adx(df: pd.DataFrame, period=14) -> pd.Series:
    """Average Directional Index — fuerza de la tendencia, no dirección"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr))
    
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx_series = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_series

def roc(df: pd.DataFrame, period=10) -> pd.Series:
    """Rate of Change — momentum puro"""
    return df['Close'].pct_change(periods=period) * 100

def vwap_deviation(df: pd.DataFrame) -> pd.Series:
    """Desviación del precio respecto al VWAP diario acumulado"""
    # Equivalent to vwap_discount_daily roughly, but let's implement standard daily VWAP deviation
    # In daily bars, we'll use a rolling VWAP of 20 days and compute percentage deviation.
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    vp = tp * df['Volume']
    vwap = vp.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
    return ((df['Close'] - vwap) / vwap) * 100

def atr_normalized(df: pd.DataFrame, period=14) -> pd.Series:
    """ATR como % del precio — volatilidad normalizada"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return (atr / close) * 100

def cmf(df: pd.DataFrame, period=20) -> pd.Series:
    """Chaikin Money Flow — presión compradora/vendedora con volumen"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    vol = df['Volume']
    
    mfv = ((close - low) - (high - close)) / (high - low + 1e-10) * vol
    cmf_series = mfv.rolling(window=period).sum() / vol.rolling(window=period).sum()
    return cmf_series

# Forzando recarga de Streamlit para que detecte adx
